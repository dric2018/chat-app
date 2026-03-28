from __init__ import logger

import abc

from config import CFG

from datetime import datetime

import json

from enum import Enum

from db.election_db import ElectionDB

from langchain_core.tools import tool, BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langsmith import traceable

import pandas as pd
from pprint import pprint
from pydantic import BaseModel, Field
from prometheus_client import start_http_server

import re

from unidecode import unidecode

from utils import get_security_counter, get_entity_context

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
        description="The exact name of the table to inspect. Mandatory."
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

        # Default agent tools
        self.tools = self._collect_tools()
        self.curr_messages = []

        self.model_name = CFG.BASE_MODEL

        date = datetime.now().strftime("%Y—%m-%d")
        year = date.split("-")[0]

        self.init_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Today is {date}. We are in the year {year}.\n
                You are an expert employed in the analysis process of the 2025 legislative elections in Côte d'Ivoire. 
                We assume any reference to 'elections' refers to this 2025 legislative elections in Cote d'Ivoire. 
                Use this assumption if user does not specify which elections they are interested in.  
                All your knowledge about Côte d'Ivoire will be crucial here including demographics, the overall political situation in the recent years.
                Do not guess if your are unsure about information related to that country.
                The user can ask about data for a mispelled region or constituency or a candidate. 
                Be flexible about these typos/errors by checking for similar names/texts in the corresponding candidate and constituency tables.
                """),
            ("human", "Hello, how are you doing?"),
            ("ai", "I'm doing well, thanks! How can I help you with the analysis of the 2025 legislative elections in Cote d'Ivoire ?"),
            ("human", "{user_input}"),
        ])

        # chain
        self.client = ChatOpenAI(
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
        self.llm_with_tools = self.client.bind_tools(self.tools, tool_choice="auto")
        self.chain     = self.init_prompt | self.llm_with_tools
                
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

    @traceable(run_type="llm")
    def get_answer(self, user_prompt: str, chat_history: list = None):
        """The standard execution pipeline for ALL agents."""
        try:
            yield {"type": "status", "content": "Identifying intent..."}
            intent = self._get_intent(user_prompt)
            yield {"type": "status", "content": f"✅ Intent: {intent.name}."}
            
            if intent == QueryIntent.INVALID:
                yield {"type": "status", "content": "This query does not appear to be election-related or safe."}
                pass

            for out in self.process_query(
                user_prompt=user_prompt, 
                intent=intent, 
                chat_history=chat_history
                ):
                yield out
                                
        except SecurityViolationError as e:
            get_security_counter().inc()                
            logger.error(f"SECURITY SHIELD: {e}")
            yield {
                "type": "error", 
                "content": f"Operation {str(e)} is strictly prohibited."
            }
            return 

        except (DatabaseConnectionError, Exception) as e:
            logger.critical(f"SYSTEM ERROR: {e}", exc_info=True)
            yield {"type": "error", "content": f"Could not connect to DB. {e}"}
            return 
        
        except Exception as e:
            logger.error(f"Could not get agent's answer. {e}", exc_info=True)
            yield {"type": "error", "content": f"Could not get agent's answer. {e}"}
            return

    @traceable
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

            response = self.client.invoke(messages)

            # tool_requests = response.tool_calls
            # if not tool_requests and "<tool_call>" in response.content:
            #     pass # we ignore tools here
            
            label = response.content.strip()
            intent = QueryIntent[label] if label in [i.name for i in QueryIntent] else QueryIntent.INVALID
            
        except KeyError:
            logger.error("KeyError exception...defaulting to invalid query")
            return QueryIntent.INVALID
        
        except Exception as e:
            logger.error(f"Could not get intent. {e}", exc_info=True)
        
        return intent

    @traceable
    def _interpret_results(
            self, 
            user_prompt: str,
            data:pd.DataFrame, 
            intent: QueryIntent,
            chat_history:list=None
        ):
        """
        Summarizes data into a natural language response based on intent.
        """
        log_msg = "Interpreting results..."
        logger.info(log_msg)

        # yield {
        #     "type": "status",
        #     "content": "Interpreting results..."
        # }

        if data is None or data.empty:
            return "I found no data matching your request."

        # Convert dataframe to a readable Markdown table for the LLM
        data_table = data.to_markdown(index=False)
        
        # Context-aware instructions based on intent
        intent_context = {
            QueryIntent.AGGREGATION: "Focus on the totals and key metrics.",
            QueryIntent.RANKING: "Highlight the top performers and the gaps between them.",
            QueryIntent.CHART: "Describe the distribution or trend. Mention the highest and lowest points.",
            QueryIntent.GENERAL: "Provide a direct and concise answer."
        }

        if chat_history is not None:
            history_context = ""
            for m in chat_history[-3:]:
                role = "USER" if isinstance(m, HumanMessage) else "ASSISTANT" if isinstance(m, AIMessage) else "SYSTEM"
                history_context += f"{role}: {m.content}\n" 

        interp_prompt = f"""
        User Query: "{user_prompt}"\n
        Conversation context: {history_context}\n
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
            response = self.client.invoke(messages)

            tool_requests = response.tool_calls
            
            if not tool_requests and "<tool_call>" in response.content:
                pass # we ignore tools here

            return unidecode(response.content.strip())
        
        except Exception as e:
            logger.error(f"Could not interpret results. {e}", exc_info=True)

    @traceable
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
