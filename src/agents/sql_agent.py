from __init__ import logger

from agents.agent import (
    Agent,  
    SecurityViolationError, 
    db_client, 
    QueryIntent, 
    ToolReasoningSchema, 
    TableActionSchema
)

from config import CFG

import duckdb

from langchain_core.tools import tool, BaseTool
from langchain_core.messages import (
    SystemMessage, 
    HumanMessage, 
    ToolMessage, 
    AIMessage
)
from langsmith import traceable

import re

import sqlparse

import time
from typing import Tuple

from utils import get_security_counter


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
    @traceable
    @staticmethod
    def get_db_schema(reasoning:str) -> str:
        """Returns the schema (table names and column definitions) of the database."""
        logger.info(f"LLM Reasoning (get_db_schema): {reasoning}")
        return db_client.get_schema_snapshot()
    
    @tool(args_schema=ToolReasoningSchema)
    @traceable
    def list_tables(reasoning: str) -> str:
        """Get a list of all available tables in the database."""
        logger.info(f"LLM Reasoning (list_tables): {reasoning}")
        
        with duckdb.connect(str(CFG.DB_PATH), read_only=True) as conn:
            tables = conn.execute("SHOW TABLES").df()
            return tables.to_json()
    
    @tool(args_schema=TableActionSchema)
    @traceable
    @staticmethod
    def describe_table(table_name: str, reasoning: str) -> str:
        """
            Use this to get column names, types, and descriptions for a specific table.
            REQUIRED: You must provide both the 'table_name' and your 'reasoning' arguments.
        
        """
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
    @traceable
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
    @traceable
    @staticmethod
    def validate_sql(
        sql:str,
        forbidden:list,
        reasoning:str
    )->Tuple[bool, str]:
        """
            Validating SQL syntax and logic before execution

            Args:
                - sql (str): query to be validated
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
                get_security_counter().inc()                
                out_message = "Multiple queries detected."
                return is_query_valid, out_message
            
            clean_sql = parsed[0]

            # Checking for the Command Type
            # We ensure the very first Data Manipulation Language (DML) token is 'SELECT'
            if clean_sql.get_type() != "SELECT":
                get_security_counter().inc()                
                out_message = f"Forbidden command type '{clean_sql.get_type()}'."
                return is_query_valid, out_message
            
            # Ensuring only allowed tables are used
            if not any(t in clean_sql.value.lower() for t in CFG.ALLOWED_TABLES):
                get_security_counter().inc()
                out_message = f"Unauthorized table access detected.\nSQL: {clean_sql.value}"
                return is_query_valid, out_message            

            # Deep Token Inspection (No hidden JOINs to sensitive tables)
            # We check every token for forbidden keywords that might bypass get_type()
            for token in clean_sql.flatten():
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() in forbidden:
                    get_security_counter().inc()
                    out_message = f"Forbidden keyword '{token.value}' detected."
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
    @traceable
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

    @traceable(run_type="llm")
    def generate_sql(
            self, 
            user_prompt:str, 
            intent:QueryIntent,
            stream_output:bool=CFG.IS_STREAM,
            chat_history:list=None
        ):
        """Restricted SQL Generation"""

        log_msg = f"Attempting SQL generation [Max iterations={CFG.MAX_ITERATIONS}]"
        logger.info(log_msg)
        yield {"type": "status", "content": log_msg}

        intent_instructions = {
            QueryIntent.AGGREGATION: "Use GROUP BY and SUM/AVG. Ensure results are numeric and in desc order.",
            QueryIntent.RANKING: f"Use statements like ORDER BY votes DESC and LIMIT {CFG.SQL_MAX_LIMIT}, use SUM/AVG when necessary.",
            QueryIntent.CHART: "Return exactly two columns: a label (e.g., party) and a numeric value."
        }

        intent_instruction = intent_instructions.get(intent, "")

        self.generation_prompt = f"""
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
                - Constituency, party, region, and candidate names will be kept or manipulated in uppercase form for convenience.\n
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
        if chat_history is not None:
            logger.info("updated chat history provided !")
            history_context = ""
            for m in chat_history[-3:]:
                role = "USER" if isinstance(m, HumanMessage) else "ASSISTANT" if isinstance(m, AIMessage) else "SYSTEM"
                history_context += f"{role}: {m.content}\n" 

            msg = HumanMessage(content=f"Consider the following context while generating your SQL query: {history_context}")
        else:
            msg =  HumanMessage(content=user_prompt)

        messages = [
            SystemMessage(content=self.generation_prompt), 
            msg
        ]

        if stream_output:
            self._generate_sql_streaming(messages)
        else:
            try: 
                steps = []
                for it in range(CFG.MAX_ITERATIONS):
                    step_counter = f"{it+1}/{CFG.MAX_ITERATIONS}"
                    
                    log_msg = f"\n[🚶🏼‍➡️ Walking down the SQL path...step {step_counter}]"
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
                            start_call = time.time()
                            # Handle both native dict and parsed dict structures
                            name = tool_req.get("name") or tool_req.get("function", {}).get("name")
                            args = tool_req.get("args") or tool_req.get("arguments") or tool_req
                            call_id = tool_req.get("id", "manual")

                            selected_tool = {t.name: t for t in self.tools}[name]
                            log_msg = f"Using Tool {name}. Args: {args}"
                            logger.info(log_msg) 
                            
                            observation = selected_tool.invoke(args)
                            
                            call_duration = time.time() - start_call 
                            steps.append(log_msg+f" [Completed in {call_duration:.5f}s]")
                            
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

    @traceable
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

    @traceable
    def process_query(
            self, 
            user_prompt: str, 
            intent: QueryIntent,
            chat_history:list=None
        ):

        try:
            for response in self.generate_sql(user_prompt=user_prompt, intent=intent, chat_history=chat_history):
                yield response
                if response["type"] == "final_sql":
                    # Structural check before hitting the DB
                    sql     = response["sanitized_query"]
                    steps   = response["steps"]
                    
                    logger.info(f"Generated SQL: {sql}")
                    
                    is_safe, out_message = self.validate_sql.invoke({
                        "reasoning": "validating SQL query",
                        "sql": sql,
                        "forbidden": self.forbidden
                    })
                    
                    if not is_safe:
                        log_msg = f"SQL Violation Attempted: {out_message}"
                        logger.error(log_msg)
                        yield {
                            "type": "error", 
                            "content": log_msg
                        }   

                        raise SecurityViolationError

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

                            results =  self.llm_with_tools.invoke([HumanMessage(content=fix_prompt)]).content                        
                            
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
                            "interpretation": self._interpret_results(
                                user_prompt=user_prompt, 
                                data=results, 
                                intent=intent,
                                chat_history=chat_history
                                )
                        }
        except Exception as e:
            logger.error(f"Query generation error: {e}", exc_info=True)
            yield {"type": "error", "content": str(e)}
            return
