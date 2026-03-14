from __init__ import logger

import abc

from config import CFG

from datetime import datetime

import json

import duckdb

from enum import Enum

from db.election_db import ElectionDB

from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

import pandas as pd
from pprint import pprint
from pydantic import BaseModel, Field
from prometheus_client import start_http_server

import re
import sqlparse

import time
from typing import Tuple, Literal, Optional
import traceback

from utils import get_security_counter

db_client = ElectionDB()

class QueryIntent(Enum):
    AGGREGATION = "aggregation"
    RANKING     = "ranking"
    CHART       = "chart"
    GENERAL     = "general" # we default this to RAG
    INVALID     = "invalid"


class ToolReasoningSchema(BaseModel):
    reasoning: str = Field(
        description="Explanation of why this tool is being called and what information is sought."
    )

class TableActionSchema(ToolReasoningSchema):
    table_name: str = Field(
        description="The name of the table to inspect."
    )

class BaseAgentException(Exception):
    """Parent class for all custom agent errors."""
    pass

class SecurityViolationError(BaseAgentException):
    """Raised when forbidden keywords (DROP, DELETE, etc.) are detected."""
    pass

class IntentClassificationError(BaseAgentException):
    """Raised when the LLM fails to provide a valid QueryIntent."""
    pass

class DatabaseConnectionError(BaseAgentException):
    """Raised when DuckDB fails to connect or a query is syntactically invalid."""
    pass


VIOLATIONS_COUNTER = get_security_counter()

try:
    start_http_server(port=8001, addr='0.0.0.0')    
except OSError:
    pass 

class Agent(abc.ABC):
    def __init__(
            self, 
            vllm_url:str=CFG.VLLM_BASE_URL,
        ):

        self.model_name = CFG.BASE_MODEL
        self.client     = ChatOpenAI(
                            openai_api_base=vllm_url, 
                            # base_url=f"http://localhost:{CFG.VLLM_PORT}/v1",
                            api_key="EMPTY",
                            max_completion_tokens=CFG.MAX_TOKENS,
                            temperature=CFG.GENERATION_TEMPERATURE,
                            model=CFG.BASE_MODEL,
                            extra_body={
                                "reasoning_effort": CFG.REASONING_EFFORT,
                                "chat_template_kwargs": {"enable_thinking": False},
                            },
                            streaming=CFG.IS_STREAM,
                            timeout=CFG.TIMEOUT,
                            max_retries=3
                        )
        # self.client.openai_api_base = vllm_url
        # logger.info(f"Agent::Setting {vllm_url=}")
                
        self._forbidden = [
            "DROP", 
            "DELETE", 
            "UPDATE", 
            "INSERT", 
            "ALTER", 
            "TRUNCATE", 
            "CREATE"
        ]
        
        self.results_limit = CFG.SQL_MAX_LIMIT

        # Schema Metadata for the LLM
        self.schema_context = """
            always use the available tools to preview the database before attempting anything
            """

        self.metrics = {
            "violations": 0,
            "total_latency": 0.0,
            "intents": {intent.value: 0 for intent in QueryIntent}
        }

        self.investigation_logs = [] # Reset logs for this run
        self.generation_prompt  = ""
        # self.messages = [] # will have to be moved to db for safety

        # Default agent tools
        self.tools = self._collect_tools()

        self.llm_with_tools = self.client.bind_tools(self.tools, tool_choice="auto")

    @property
    def forbidden(self) -> tuple:
        """Read-only access to forbidden keywords."""
        return self._forbidden

    def __setattr__(self, name, value):
        # Prevent any modification to 'forbidden' or '_forbidden' after initialization
        if name in ("forbidden", "_forbidden") and hasattr(self, "_forbidden"):
            raise AttributeError(f"The '{name}' attribute is sealed and cannot be modified.")
        super().__setattr__(name, value)
    
    @abc.abstractmethod
    def process_query(self, user_prompt: str, intent: QueryIntent, chat_history: list= None):
        """Subclasses define HOW they get the data (SQL, RAG, etc.)"""
        pass

    def __call__(self):
        logger.info("Agent Initialized")

    def _collect_tools(self) -> list:
        """Scans for any attribute that is an instance of a LangChain Tool."""
        found_tools = []
        for name in dir(self):
            if name.startswith('_'): continue
            try:
                member = getattr(self, name)
                # Check if the member is a LangChain Tool instance
                if isinstance(member, BaseTool):
                    found_tools.append(member)
            except Exception:
                continue
        return found_tools

    def get_answer(self, user_prompt: str, chat_history: list = None):
        """The standard execution pipeline for ALL agents."""
        start_time = time.time()
        try:
            yield {"type": "status", "content": "Identifying intent..."}
            intent = self._get_intent(user_prompt)
            yield {"type": "status", "content": f"✅ Intent: {intent.name}."}
            
            if intent == QueryIntent.INVALID:
                yield {"type": "status", "content": "This query does not appear to be election-related."}
                pass
                # return {"type": "error", "content": "The query does not appear to be election-related."}

            for out in self.process_query(user_prompt, intent, chat_history):
                yield out
            self.metrics["total_latency"] += (time.time() - start_time)
                                
        except SecurityViolationError as e:
            self.metrics["violations"] += 1
            VIOLATIONS_COUNTER.inc()
            logger.error(f"SECURITY SHIELD: {e}")
            yield {
                "type": "error", 
                "content": f"Security violation: Operation {str(e)} is strictly prohibited."
            }
            return 

        except (DatabaseConnectionError, Exception) as e:
            logger.critical(f"SYSTEM ERROR: {e}", exc_info=True)
            return 
        except Exception as e:
            logger.error(f"Could not get agent's answer. {e}", exc_info=True)
            yield {"type": "error", "content": f"Could not get agent's answer. {e}"}
            return

    def _get_intent(self, user_prompt:str)-> QueryIntent:
        logger.info("Formatting input to LLM")

        intent_prompt = f"""
        Classify the user's election query into one category:\n
        - AGGREGATION: Sums, counts, averages (e.g., 'total votes/scores', 'turnout average').\n
        - RANKING: Comparisons, top/bottom lists (e.g., 'who won', 'best party', 'candidate with more votes').\n
        - CHART: Requests for distribution, trends, or visualizations.\n
        - GENERAL: Simple election-related fact retrieval from database. This category can also identify general conversation patterns like greetings or casual conversations before asking anything specific.\n
            - Use this for general conversation handling whenthe user starts a generic conversation asking "how are you"-like questions.\n
        - INVALID: Does not seem to be about elections or similar competitions (e.g. wheather-related questions) AND does not fall into previous categories.\n

        Query: "{user_prompt}"\n
        Expected output: Respond ONLY with the category name as string.\n
        """

        intent = None
        try:
            messages = [
                SystemMessage(content=intent_prompt), 
                HumanMessage(content=user_prompt)
            ]

            response = self.llm_with_tools.invoke(messages)

            # tool_requests = response.tool_calls
            # if not tool_requests and "<tool_call>" in response.content:
            #     pass # we ignore tools here
            
            label = response.content.strip()
            intent = QueryIntent[label] if label in [i.name for i in QueryIntent] else QueryIntent.INVALID
            
            self.metrics["intents"][intent.value] += 1
        except KeyError:
            logger.error("KeyError exception...defaulting to invalid query")
            return QueryIntent.INVALID
        
        except Exception as e:
            logger.error(f"Could not get intent. {e}", exc_info=True)
        
        return intent


    def _interpret_results(
            self, 
            user_prompt: str,
            df:pd.DataFrame, 
            intent: QueryIntent
        ) -> str:
        """
        Summarizes data into a natural language response based on intent.
        """

        logger.info("Interpreting results...")

        if df is None or df.empty:
            return "I found no data matching your request."

        # Convert dataframe to a readable Markdown table for the LLM
        data_table = df.to_markdown(index=False)
        
        # Context-aware instructions based on intent
        intent_context = {
            QueryIntent.AGGREGATION: "Focus on the totals and key metrics.",
            QueryIntent.RANKING: "Highlight the top performers and the gaps between them.",
            QueryIntent.CHART: "Describe the distribution or trend. Mention the highest and lowest points.",
            QueryIntent.GENERAL: "Provide a direct and concise answer."
        }

        interp_prompt = f"""
        User Query: "{user_prompt}"\n
        Intent: {intent.value}\n
        Data Results:\n
        {data_table}\n
        Task: {intent_context.get(intent, "Summarize the data.")}\n
        Provide a short natural language summary of these results. \n
        If there are many rows, highlight the most significant ones.\n
        Expected Output: Only output the interpretation.
        """

        messages = [
            SystemMessage(content="You are an expert election data analyst."), 
            HumanMessage(content=interp_prompt)
        ]

        try:
            logger.info("Prompting LLM")
            response = self.llm_with_tools.invoke(messages)

            tool_requests = response.tool_calls
            
            if not tool_requests and "<tool_call>" in response.content:
                pass # we ignore tools here
            
            return response.content.strip()
        except Exception as e:
            logger.error(f"Could not interpret results. {e}", exc_info=True)

    def _format_not_found(self, prompt:str, logic_path):
        """Standardized failure response per your requirements."""
        return (
            f"**Not found in the provided PDF dataset.**\n\n"
            f"User asked: {prompt}\n\n"
            f"*Attempted:* {logic_path}\n"
        )

    def _parse_xml_tool_calls(self, content: str) -> list:
        """Extracts tool calls from <tool_call> XML tags."""
        # Find all content between <tool_call> tags
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)
        
        parsed_calls = []
        for match in matches:
            try:
                parsed_calls.append(json.loads(match))
            except json.JSONDecodeError:
                continue
        return parsed_calls


class SQLAgent(Agent):
    def __init__(
        self,
        vllm_url:str=CFG.VLLM_BASE_URL
    ):
        super().__init__(vllm_url)
        self.client.openai_api_base = vllm_url
        # logger.info(f"SQLAgent::Setting {vllm_url=}")
        
        self.tools = self._collect_tools()
        self.llm_with_tools = self.client.bind_tools(self.tools, tool_choice="auto")

    @tool
    @staticmethod
    def get_db_schema(reasoning:str) -> str:
        """Returns the schema (table names and column definitions) of the database."""
        logger.info(f"LLM Reasoning (get_db_schema): {reasoning}")
        return db_client.get_schema_snapshot()

    @tool(args_schema=ToolReasoningSchema)
    def list_tables(reasoning: str) -> str:
        """Get a list of all available tables in the election database."""
        logger.info(f"LLM Reasoning (list_tables): {reasoning}")
        
        with duckdb.connect(str(CFG.DB_PATH), read_only=True) as conn:
            tables = conn.execute("SHOW TABLES").df()
            return tables.to_json()
    
    @tool(args_schema=TableActionSchema)
    @staticmethod
    def describe_table(table_name: str, reasoning: str) -> str:
        """Get column names, types, and descriptions for a specific table."""
        logger.info(f"LLM Reasoning (describe_table [{table_name}]): {reasoning}")
        
        try:
            with duckdb.connect(str(CFG.DB_PATH), read_only=True) as conn:
                # We use a string return so the LLM can 'read' the error
                schema = conn.execute(f"DESCRIBE {table_name}").df()
                return schema.to_string()
        except Exception as e:
            # Return the actual DuckDB error message to the LLM
            logger.warning(f"Database Error: {str(e)}")
            return f"Error: {str(e)}. Please check the table name and try again."


    @tool(args_schema=TableActionSchema)
    @staticmethod
    def sample_data(
        table_name: str, 
        reasoning: str,
        num_samples:int=5
    ) -> str:
        """Fetch the first 3 rows of a table to understand data formats (e.g. date styles, category names)."""
        logger.info(f"LLM Reasoning (sample_data [{table_name}]): {reasoning}")

        with duckdb.connect(str(CFG.DB_PATH), read_only=True) as conn:
            # Always limit samples to prevent data dumps
            sample = conn.execute(f"SELECT * FROM {table_name} USING SAMPLE {num_samples}").df()
            return sample#.to_markdown(index=False)
        
    @tool
    @staticmethod
    def validate_sql(
        sql:str,
        metrics:dict,
        forbidden:list,
        reasoning:str
    )->Tuple[bool, str]:
        """
            Validating SQL syntax and logic before execution

            Args:
                - sql (str): query to be validated
                - metrics (dict): list of SQL-related metrics to track (number of violations identified and blocked)
                - forbidden (list): list of forbidden SQL statements
                - reasoning (str): Why this tool has to be used in the current call

            Returns:
                tuple:
                    - is_query_valid: whether or not the input query is valid
                    - out_message: message accompagnying SQL query 
        """
        logger.info(f"LLM Reasoning (validate_sql): {reasoning}")
        
        is_query_valid  = False
        out_message     = ""

        sql = sql.upper()

        try:
            parsed = sqlparse.parse(sql)
            if len(parsed) > 1:
                metrics["violations"] += 1
                get_security_counter().inc()                
                out_message = "Security Violation: Multiple queries detected."
                return is_query_valid, out_message
            
            clean_sql = parsed[0]

            # Checking for the Command Type
            # We ensure the very first Data Manipulation Language (DML) token is 'SELECT'
            if clean_sql.get_type() != "SELECT":
                metrics["violations"] += 1
                get_security_counter().inc()                
                out_message = f"Security Violation: Forbidden command type '{clean_sql.get_type()}'."
                return is_query_valid, out_message
            
            # Ensuring only allowed tables are used
            if not any(t in clean_sql.value.lower() for t in CFG.ALLOWED_TABLES):
                metrics["violations"] += 1
                get_security_counter().inc()
                out_message = f"Security Violation: Unauthorized table access detected.\nSQL: {clean_sql.value}"
                return is_query_valid, out_message            

            # Deep Token Inspection (No hidden JOINs to sensitive tables)
            # We check every token for forbidden keywords that might bypass get_type()
            for token in clean_sql.flatten():
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() in forbidden:
                    metrics["violations"] += 1
                    get_security_counter().inc()
                    out_message = f"Security Violation: Forbidden keyword '{token.value}' detected."
                    return is_query_valid, out_message
            
            if "LIMIT" not in clean_sql:
                clean_sql = clean_sql.value.strip().rstrip(";") + f" LIMIT {CFG.SQL_MAX_LIMIT}"
                is_query_valid = True
                out_message = f"{clean_sql}"
            return is_query_valid, out_message
        
        except Exception as e:
            logger.error(f"Parsing Error: {str(e)}")
            return False, f"Parsing Error: {str(e)}"
    
    @tool
    @staticmethod
    def execute_read_query(sql_query: str, reasoning:str):
        """Executes a SQL SELECT query and returns the results. Use only for data retrieval."""
        
        logger.info(f"LLM Reasoning (execute_read_query): {reasoning}")

        results = None
        out_msg = "OK"

        try: 
            with duckdb.connect(str(CFG.DB_PATH), read_only=True) as conn:
                results = conn.execute(sql_query).df().drop_duplicates()
                
                yield {
                    "type": "status",
                    "content": results,
                    "message": out_msg
                }
        except Exception as e:
            out_msg = f"Could not execute query.\n{str(e)}"
            logger.error(out_msg)

            yield {
                    "type": "error",
                    "content": out_msg
                    }
            return

    def _generate_sql_streaming(
            self,
            messages
        ):
        try:
            for step in range(CFG.MAX_ITERATIONS):
                yield {"type": "status", "content": f"Iteration {step} starting...\n"}
                
                response = None
                
                # STREAMING PHASE
                for chunk in self.llm_with_tools.stream(messages):
                    # print(f"DEBUG: content='{chunk.content}' tools={chunk.tool_calls}")
                    
                    if response is None: response = chunk
                    else: response += chunk
                    
                    if chunk.content:
                        # Yield RAW content so the stream is visible!
                        yield {"type": "token", "content": chunk.content}

                # TOOL DETECTION PHASE
                tool_requests = response.tool_calls
                logger.info(f"{tool_requests=}")

                # Filter out the "Ghost" calls (empty names)
                valid_tools = [t for t in tool_requests if t.get('name')]
            
                if not valid_tools and "<tool_call>" in (response.content or ""):
                    yield {"type": "status", "content": "Parsing XML tool calls...\n"}
                    valid_tools = self._parse_xml_tool_calls(response.content)

                if valid_tools:
                    messages.append(response)

                    for tool_req in valid_tools:
                        name = tool_req.get("name") or tool_req.get("function", {}).get("name")
                        args = tool_req.get("args") or tool_req.get("arguments") or tool_req
                        call_id = tool_req.get("id", f"step_{step}")
                        
                        logger.info(f"Executing: {name}...")
                        yield {"type": "action", "content": f"Executing: {name}..."}

                        # Tool Execution
                        selected_tool = {t.name: t for t in self.tools}[name]
                        observation = selected_tool.invoke(args)    
                        yield {"type": "status", "content": observation}

                        messages.append(ToolMessage(content=str(observation), tool_call_id=call_id))
                    
                    continue # Re-enter loop to let LLM react to tool results
                else:
                    logger.info(f"No tools found: {tool_requests=}")
                    yield {"type": "status", "content": f"No tools found: {tool_requests=}"}
                    
                # If it's the 3rd time without a tool call, tell the LLM to stop thinking
                if step > 1 and (not response.tool_calls or not tool_requests):
                    messages.append(
                        HumanMessage(content="You are overthinking. If you have the data, provide the SQL now. If not, use a tool.")
                    )
                
                # FINALIZATION PHASE
                # If we are here, there are no more tool calls.
                if response.content and "SELECT" in response.content.upper():
                    final_sql = self._sanitize_sql(response.content)
                    logger.info(f"Capturing final SQL query {final_sql}")
                    yield {
                        "type": "final_sql", 
                        "content": response.content,
                        "query":final_sql}
                    return # Exit generator successfully
                else:
                    logger.warning(f"No SQL query found")

                
            yield {"type": "error", "content": "Max iterations reached."}

        except Exception as e:
            logger.critical(f"Query generation error: {e}", exc_info=True)
            yield {"type": "error", "content": str(e)}

    def generate_sql(
            self, 
            user_prompt:str, 
            intent:QueryIntent,
            stream_output:bool=CFG.IS_STREAM
        ):
        """Restricted SQL Generation"""

        log_msg = f"Attempting SQL generation [Max iterations={CFG.MAX_ITERATIONS}]"
        logger.info(log_msg)
        yield {"type": "status", "content": log_msg}

        intent_instructions = {
            QueryIntent.AGGREGATION: "Use GROUP BY and SUM/AVG. Ensure results are numeric and in desc order.",
            QueryIntent.RANKING: f"Use ORDER BY votes DESC and LIMIT {CFG.SQL_MAX_LIMIT}.",
            QueryIntent.CHART: "Return exactly two columns: a label (e.g., party) and a numeric value."
        }

        intent_instruction = intent_instructions.get(intent, "")

        self.generation_prompt = f"""
        Today is {datetime.now().strftime("%Y—%m-%d")}\n
        You are a Data Scientist with deep expertise in elections data management, exploration, and an expert in SQL query construction and optimization. \n
        Your purpose is to transform natural language requests into precise, efficient SQL queries that deliver exactly what the user requests.\n
        User Intent: {intent.value}. {intent_instruction}\n
        Instructions:
            - Devise your own strategic plan to explore and understand the database before constructing queries.\n
            - Determine the most efficient sequence of database investigation steps based on the specific user request.\n
            - Perform one investigation step at a time. Wait for the tool result before deciding on the next step.\n
            - Use the native tool calling capability.\n
            - Independently identify which database elements require examination to fulfill the query requirements.\n
            - You only have a maximum of {CFG.MAX_ITERATIONS} steps to return the SQL query, so make sure to minimize database exploration and only explore tables that are relevant to the user's request.\n
                - Do not revisit a table twice unless you have found new insights.\n
                - If you reach the {CFG.MAX_ITERATIONS}th iteration (step), return the SQL query you think of.\n
            - If you are unsure about table names or column types, do not guess and use the provided tools:\n
                - You can use list_tables to find the correct data source and table names. \n
                - Only use the allowed tables as specified in this list: {CFG.ALLOWED_TABLES}\n
                - You can use describe_table to see exact column names in previously listed tables.\n
                - You can use sample_data to understand how values are formatted and connected between the identified tables.\n
                - The user can ask about data for a mispelled region or constituency or a candidate. Be flexible about these typos/errors by checking the relevant tables/fields for similar names/texts in case no direct match is found.\n
            - HARD CONSTRAINT: \n
                - Use execute_read_query to ensure the SQL query actually works before returning it.\n
            - Once you have gathered sufficient evidence, stop exploring the database and generate the final SQL SELECT query. \n
            - Make sure to add any useful filed/column that may ease later interpretation of results\n
            - Formulate your query approach based on your professional judgment of the database structure.\n
            - Balance comprehensive exploration with efficient tool usage to minimize unnecessary operations.\n
            - For every tool call, include a detailed reasoning parameter explaining your strategic thinking.\n
            - Be sure to specify every required parameter for each tool call.\n
            - Use JOIN and ORDER BY where necessary.\n
            - HARD CONSTRAINT: Never use {self.forbidden} statements!\n
            - Always include a LIMIT {CFG.SQL_MAX_LIMIT}. \n
        Expected output: Only the final SQL query.\n
            - Your responses should be formatted as Markdown. 
        """

        messages = [
            SystemMessage(content=self.generation_prompt), 
            HumanMessage(content=user_prompt)
        ]

        if stream_output:
            self._generate_sql_streaming(messages)
        else:
            try: 
                steps = []
                for it in range(CFG.MAX_ITERATIONS):
                    step_counter = f"{it+1}/{CFG.MAX_ITERATIONS}"
                    
                    log_msg = f"\n[SQL generation step {step_counter}]"
                    logger.info(log_msg)
                    yield {"type": "status", "content": log_msg}
                    response = self.llm_with_tools.invoke(messages)
                    
                    tool_requests = response.tool_calls
                    
                    if not tool_requests and "<tool_call>" in response.content:
                        logger.info("Extracting tool calls")
                        tool_requests = self._parse_xml_tool_calls(response.content)
                        for i, tr in enumerate(tool_requests):
                            tr['id'] = f"manual_{i}" 
                    
                    if tool_requests:
                        messages.append(response)
                        for tool_req in tool_requests:
                            # Handle both native dict and parsed dict structures
                            name = tool_req.get("name") or tool_req.get("function", {}).get("name")
                            args = tool_req.get("args") or tool_req.get("arguments") or tool_req
                            call_id = tool_req.get("id", "manual")

                            selected_tool = {t.name: t for t in self.tools}[name]
                            
                            start_call = time.time()
                            observation = selected_tool.invoke(args)
                            call_duration = time.time() - start_call
                            
                            log_msg = f"Using Tool {name}. Reasoning: {args["reasoning"]} [Completed in {call_duration:.5f}s]"
                            logger.info(log_msg)  
                            steps.append(log_msg)
                            
                            yield {"type": "status", "content": f"Used Tool {name}."}
                            
                            # logger.info(f"Tool output: {str(observation)}")

                            messages.append(ToolMessage(content=str(observation)+f"\nIteration: {step_counter}", tool_call_id=call_id))
                        continue
                    else:
                        msg = "No tool call identified" 
                        if len(steps) > 0:
                            msg = "No more tool call identified" 
                        logger.info(msg)

                    if response.content:
                        # If it's just the final SQL string
                        yield {
                            "type": "final_sql", 
                            "content": response.content,
                            "sanitized_query": self._sanitize_sql(response.content),
                            "steps": steps
                            }
                        
                        return
                        
                    
                yield {
                    "type": "error", 
                    "content": "Maximul iterations reached.",
                    "steps": steps
                    }
                return

            except Exception as e:
                logger.critical(f"Query generation error: {e}", exc_info=True)
                yield {"type": "error", "content": str(e)}
                return
            
    def _sanitize_sql(self, text: str) -> str:
        """
        Strictly extracts ONLY the SQL SELECT statement.
        Removes <think> blocks, Markdown, and all conversational prose.
        """
        if not text:
            return ""

        # Stripping the Internal Monologue (<think> tags)
        clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Extracing content between ```sql and ``` if it exists
        # This handles the most common Markdown case
        md_match = re.search(r"```sql\s*(.*?)\s*```", clean, re.DOTALL | re.IGNORECASE)
        if md_match:
            clean = md_match.group(1)
        else:
            # Fallback: Find the first 'SELECT' and everything after it
            select_match = re.search(r"(SELECT\s+.*)", clean, re.DOTALL | re.IGNORECASE)
            if select_match:
                clean = select_match.group(1)

        # Cleaning up the extracted string
        # Removing any remaining markdown backticks that weren't caught
        clean = clean.replace("```", "").strip()

        # Using sqlparse for structural cleaning
        statements = sqlparse.split(clean)
        if not statements:
            return ""
        
        # We take the first statement that contains SELECT
        final_sql = ""
        for stmt in statements:
            if "SELECT" in stmt.upper():
                # Format: strip comments, flatten whitespace
                final_sql = sqlparse.format(stmt, strip_comments=True, keyword_case='upper').strip()
                # Remove trailing semicolon for DuckDB compatibility
                final_sql = final_sql.rstrip(";")
                break

        # Global Constraint: Ensure LIMIT is present
        if final_sql and "LIMIT" not in final_sql.upper():
            final_sql = f"{final_sql} LIMIT {self.results_limit}"

        return final_sql

    def process_query(
            self, 
            user_prompt: str, 
            intent: QueryIntent,
            chat_history:list=None
        ):

        try:
            for response in self.generate_sql(user_prompt, intent):
                yield response
                if response["type"] == "final_sql":
                    # Structural check before hitting the DB
                    sql     = response["sanitized_query"]
                    steps   = response["steps"]
                    
                    logger.info(f"Generated SQL: {sql}")
                    
                    is_safe, out_message = self.validate_sql.invoke({
                        "reasoning": "validating SQL query",
                        "sql": sql,
                        "metrics": self.metrics,
                        "forbidden": self.forbidden
                    })
                    
                    if not is_safe:
                        logger.error(f"SQL Violation Attempted: {out_message}")
                        # Increment the Prometheus counter
                        yield {
                            "type": "error", 
                            "content": f"Security violation. {out_message}"
                        }   
                        return 

                    results = None  
                    for out in self.execute_read_query.invoke({
                            "sql_query": sql,
                            "reasoning": "Retrieving data from database"
                        }):

                        if out.get("type") == "status":
                            results = out["content"]
                        
                        if out.get("type") == "error":
                            fix_prompt= f"""
                            Based on this message: {out["content"]}, what can you do to fix this error? \n
                            Answer directly with the fixed query keeping the original parameters (e.g. order or limit) unchanged.
                            Original query: {sql}
                            """

                            results =  self.client.invoke([HumanMessage(content=fix_prompt)]).content                        
                            
                            return 

                    # if results.empty or results is None:
                    #     yield {"type": "error", "content": "No data found. Data retrieval query came back empty."}
                    #     return 
                    if results is not None and not results.empty:
                        yield {
                            "type": "data",
                            "intent": intent,
                            "data": results,
                            "steps": steps,
                            "final_sql": sql,
                            "interpretation": self._interpret_results(user_prompt, results, intent)
                        }
        except Exception as e:
            logger.error(f"Query generation error: {e}", exc_info=True)
            yield {"type": "error", "content": str(e)}
            return

class RAGAgent(Agent):
    def __init__(
            self, 
            vllm_url:str=CFG.VLLM_BASE_URL, 
            ):
        super().__init__(vllm_url)
        self.client.openai_api_base = vllm_url
        # logger.info(f"RAGAgent::Setting {vllm_url=}")
        
        self.tools = self._collect_tools()
        self.llm_with_tools = self.client.bind_tools(self.tools, tool_choice="auto")

    @tool
    @staticmethod
    def search_database(query: str) -> str:
        """
        Performs hybrid (Full-Text and Vector)s search to find relevant topics for disambiguation.
        Use this when a query is vague (e.g., 'Who won in Tiapoum?').
        """
        # Use your existing weight logic
        w_vs, w_fts = (0.3, 0.7) if len(query.split()) <= 2 else (0.7, 0.3)
        
        return db_client.hybrid_search(query, weight_vs=w_vs, weight_fts=w_fts)

    def process_query(
            self, 
            user_prompt: str, 
            intent: QueryIntent,
            chat_history:list=None
        ):

        """Standard RAG Pipeline: Retrieve -> Augment -> Generate"""
        
        yield {"type": "status", "content": "reparing for RAG..."}       
        
        limited_history = chat_history[-5:] if chat_history else []

        history_str = ""
        for msg in limited_history:
            role = "User" if msg.type == "human" else "Assistant"
            # Extract text from the message object
            content = msg.content 
            history_str += f"{role}: {content}\n"

        sys_prompt = f"""
        You are an Election Specialist. \n
        Instructions:
        - Use the following conversation history to understand context:\n
        {history_str}\n
        - Use the search tools to find relevant facts that can help answer the user's request based on the identified intent.\n
        - Intent: {intent.value}\n
        - Exceptions:
            - If the query is too broad or ambiguous, 1) Call search_database. 2) Ask the user for clarification. 3) Present the most relevant matches from the search results as options."
            Example of vague queries:
                - "Who won in Tiapoum?" (ambiguous; a win is equivalent to being elected)
                - "Show turnout in Abidjan." (scope ambiguity but looking for all related circonscription)
                - "Top 5 in Grand-Bassam."(potential naming collisions)
        """
        
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:            
            response = self.llm_with_tools.invoke(messages)
            
            tool_requests = response.tool_calls
            
            if not tool_requests and "<tool_call>" in response.content:
                logger.info("Extracting tool calls")
                yield {"type": "status", "content": "Extracting tool calls"}

                tool_requests = self._parse_xml_tool_calls(response.content)
                for i, tr in enumerate(tool_requests):
                    tr['id'] = f"manual_{i}" 
            
            if tool_requests:
                messages.append(response)
                for tool_req in tool_requests:
                    # Handle both native dict and parsed dict structures
                    name = tool_req.get("name") or tool_req.get("function", {}).get("name")
                    args = tool_req.get("args") or tool_req.get("arguments") or tool_req
                    call_id = tool_req.get("id", "manual")

                    selected_tool = {t.name: t for t in self.tools}[name]
                    logger.info(f"Executing tool: {name} with args {args}")
                    yield {"type": "status", "content": f"Executing tool: {name} with args {args}"}

                    observation = selected_tool.invoke(args)

                    logger.info(f"Executing out: {str(observation)}")

                    if 'search' in name:
                        clarification_prompt = f"""
                        The user asked: '{user_prompt}'. 
                        We found these matching entities: {str(observation)}.
                        If there are multiple matches, ask the user to specify which one they meant.
                        Format your response as a polite question with the options as bullet points.
                        """
                    else:
                        clarification_prompt = str(observation)


                    messages.append(
                        ToolMessage(content=clarification_prompt, tool_call_id=call_id)
                    )
                
                yield {"type": "status", "content": "Synthesizing final answer..."}               
                
                final_answer = self.llm_with_tools.invoke(messages)
                yield {
                    "type": "text",
                    "intent": intent,
                    "content": final_answer.content,
                    # "options": search_results
                }
            else:
                logger.info(f"No tool call identified")      
                # No tools needed, just yield the direct response
                yield {
                    "type": "text",
                    "intent": intent,
                    "content": response.content
                }     
                return     
        except Exception as e:
            yield {
                "type": "error", 
                "content": f"Could not retrieve response. {e}",
                "attempt": response.content
            }
            return

class RouteValidation(BaseModel):
    decision: Literal["execute", "clarify"] = Field(
        description="Whether to proceed with the tool or ask for more info."
    )
    route: Literal["CHAT", "SQL", "RAG"]  = Field(description="The chosen route")
    clarification_question: Optional[str] = Field(
        description="If clarify, ask the clarifying question directly to the user?"
    )
    reasoning: str = Field(description="Why is this decision being made?")

class HybridAgent(Agent):    
    def __init__(
        self,
        vllm_url:str=CFG.VLLM_BASE_URL
    ):
        super().__init__(vllm_url=vllm_url)
        # agents
        self.client.openai_api_base = vllm_url
        self.sql_expert = SQLAgent(vllm_url=vllm_url)
        self.rag_expert = RAGAgent(vllm_url=vllm_url)

        sql_tools = self.sql_expert.tools
        rag_tools = self.rag_expert.tools
        
        all_tools = {t.name: t for t in (self.tools + sql_tools + rag_tools)}
        self.tools = list(all_tools.values())  

        self.router_llm = self.client.with_structured_output(
            RouteValidation, 
            method="json_schema",  # or "function_calling"
            tools=None, 
            include_raw=False,
            strict=True
        )

    def rule_based_routing(
            self, 
            user_prompt:str,
            intent:QueryIntent
    ):
        log_msg = "Identifying decision route using rule-based routing..."
        logger.info(log_msg)
        yield {"type": "status", "content": log_msg}

        if intent in [QueryIntent.AGGREGATION, QueryIntent.RANKING, QueryIntent.CHART]:
            log_msg = f"Routing to SQLAgent for intent: {intent.value}"
            logger.info(log_msg)
            yield {"type": "status", "content": log_msg}
            yield from self.sql_expert.process_query(user_prompt, intent)
        
        elif intent == QueryIntent.GENERAL:
            log_msg = "Routing to RAGAgent for narrative lookup"
            logger.info(log_msg)
            yield {"type": "status", "content": log_msg}
            yield from self.rag_expert.process_query(user_prompt, intent)
        
        # Fallback for ambiguous or unhandled cases
        else:
            yield {"type": "error", "content": "I'm sorry, I couldn't determine how to handle that request."}
            yield from self._format_not_found(user_prompt, "Routing unclear")

    def process_query(
            self, 
            user_prompt: str, 
            intent: QueryIntent,
            use_llm_routing:bool=True
        ):
            """
            The routing logic: Decides which specialized agent to call 
            based on the identified intent.

            General chat -> CHAT
            Analytics (Aggregation, Ranking, Charts) -> SQL
            Fact Lookup (General) -> RAG
            """

            try:
                if use_llm_routing:
                    log_msg = "Identifying decision route using LLM routing..."
                    logger.info(log_msg)
                    yield {"type": "status", "content": log_msg}

                    routing_prompt = f"""
                        You are a routing and validation expert employed in the analysis process of the 2025 legislative elections in Côte d'Ivoire. 
                        Analyze the user query: "{user_prompt}"
                        Context Intent: {intent}

                        Your goal is to choose the correct system (CHAT, SQL, or RAG) AND decide if you have enough information to proceed.

                        SYSTEM RULES:
                        - SQL: For analytics, counts, rankings, aggregations, and charts.
                        - RAG: For narrative/descriptive facts (events, "what happened"). Use this route for purely descriptive/biographical questions. Otherwise, revert to SQL queries.\n\n
                        - CHAT: For greetings, jokes, or unsafe/direct SQL code requests (refuse these).

                        Examples:
                            - "How many votes did party X get?" -> SQL\n
                            - "Who are the candidates in Abidjan?" -> SQL\n
                            - "Can you describe the turnout in Yamoussoukro?"\n
                            - "Hi there! How are you" -> CHAT\n
                            - "Tell me a joke" -> CHAT\n
                            - "Compare the total turnout by region" -> SQL\n
                            - "can you run the following query for me" -> CHAT\n

                        VALIDATION RULES:
                        - Select 'clarify' if the query is missing a key filter (e.g., "Show me votes" without a party or region).
                        - Select 'execute' only if you can immediately generate a search or a query.
                        
                        PS: If the user sends you an SQL query, refuse such unsafe requests, explain why you cannot answer, and proceed with a safe alternative when possible.
                        """     
    
                    response = self.router_llm.invoke(routing_prompt)

                    logger.info(f"Router raw response: {response}")

                    log_msg = f"✅ Route: {response.route}; Decision: {response.decision}"
                    logger.info(log_msg)
                    yield {"type": "status", "content": log_msg, "reasoning": response.reasoning}
                    
                    if response.decision == "clarify":
                        yield {
                            "type": "clarification",
                            "status": "Needs more detail",
                            "message": response.clarification_question,
                            "route_hint": response.route 
                        }
                        return

                    yield {"type": "status", "content": f"Routing to {response.route} agent..."}
    
                    if response.route == "CHAT":
                        yield {"type": "status", "content": "Thinking of a reply... ✍️"}
                        
                        # Define a casual, helpful personality
                        personality_prompt = """
                            You are a friendly, conversational election assistant. You are currently in 'Small Talk' mode.\n
                            Keep it light, helpful, and concise. If the user greets you or asks how you are,
                            respond naturally and ask how you can help them with election data.\n
                            Use a casual, warm, and brief tone.\n
                            DO NOT provide election data or facts unless specifically asked.\n
                            Do NOT output 'CHAT'. Talk naturally.
                        """
                        
                        history = [
                            SystemMessage(content=personality_prompt),
                            HumanMessage(content=user_prompt)
                        ]
                        
                        chat_resp =  self.client.invoke(history)

                        logger.info(f"CHAT response: {chat_resp.content}")
                        
                        yield {"type": "final", "content": chat_resp.content}
                        return

                    elif response.route =="SQL":
                        yield from self.sql_expert.process_query(user_prompt, intent)
                    else:
                        yield from self.rag_expert.process_query(user_prompt, intent)
                else:
                    yield from self.rule_based_routing(user_prompt, intent)

            except Exception as e: 
                error_trace = traceback.format_exc()
                log_msg = f"⚠️ Routing failed. {str(e)}\n{error_trace}"
                logger.error(log_msg)
                yield {"type": "error", "content": log_msg}
                return

if __name__ == "__main__":
    agent = HybridAgent(vllm_url=f"http://vllm:{CFG.VLLM_PORT}/v1")
