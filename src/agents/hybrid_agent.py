from __init__ import logger

from agents.agent import (
    Agent,  
    SecurityViolationError, 
    QueryIntent
)

from agents.sql_agent import SQLAgent
from agents.rag_agent import RAGAgent

from config import CFG

from langchain_core.messages import (
    SystemMessage, 
    HumanMessage, 
    AIMessage
)
from langchain_core.output_parsers import StrOutputParser

from pydantic import BaseModel, Field

import traceback
from typing import Literal, Optional

from utils import get_entity_context

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
            chat_history:list=None,
            use_llm_routing:bool=True,
        ):
            """
            The routing logic: Decides which specialized agent to call 
            based on the identified intent.

            General chat -> CHAT
            Analytics (Aggregation, Ranking, Charts) -> SQL
            Fact Lookup (General) -> RAG
            """

            if intent.value == QueryIntent.INVALID.value:
                is_safe, out_message = self.sql_expert.validate_sql.invoke({
                    "reasoning": "validating user query",
                    "sql": user_prompt,
                    "forbidden": self.forbidden
                })
                    
                if not is_safe:
                    logger.error(f"SQL Violation Attempted: {out_message}")
                    yield {
                        "type": "error", 
                        "content": f"Security violation. {out_message}"
                    }   

                    raise SecurityViolationError
            
            chat_history, corrections_applied = get_entity_context(
                user_prompt, 
                chat_history
            )

            logger.info(f"{corrections_applied=}")
            if corrections_applied:
                corrections = chat_history[-1]

                yield {
                    "type": "status",
                    "content": f"{corrections.content}"
                }

                correction_prompt = HumanMessage(content=f"""
                    Rephrase my initial question based in the new corrections to be applied.
                    initial query: {user_prompt}
                    corrections: {corrections}
                    Suggested rephrasing: 

                    ONLY RETURN THE SUGGESTION.
                    """
                    )

                rephrasing = self.client.invoke([correction_prompt]).content

                yield {
                    "type": "status",
                    "content": f"{rephrasing=}"
                }

                chat_history.append(
                    AIMessage(
                        content=f"""Proposed rephrasing of the initial user request.
                        initial query: {user_prompt}
                        corrections: {corrections}
                        Suggested rephrasing: {rephrasing}
                        """,
                        additional_kwargs={"action": "skip"}

                    )
                )

            try:
                if use_llm_routing:
                    log_msg = "Identifying decision route using LLM routing..."
                    logger.info(log_msg)
                    yield {"type": "status", "content": log_msg}

                    history_context = ""
                    if chat_history:
                        for m in chat_history[-3:]:
                            role = "USER" if isinstance(m, HumanMessage) else "ASSISTANT" if isinstance(m, AIMessage) else "SYSTEM"
                            history_context += f"{role}: {m.content}\n"       

                    routing_prompt = f"""
                        {self.init_prompt}

                        RECENT CONVERSATION HISTORY with eventual corrections of mispelled entities:
                        {history_context}

                        Analyze the user query based on the conversation history.
                        If you encountered typos and happen to correct them, proceed with the latest corrections instead of the mispelled entities.

                        Context Intent: {intent.value}

                        Your goal is to choose the correct system (CHAT, SQL, or RAG) AND decide if you have enough information to proceed.

                        SYSTEM RULES:
                        - SQL: For analytics, counts, rankings, aggregations, and charts.
                        - RAG: For narrative/descriptive facts. Use this route for purely descriptive questions. Otherwise, revert to SQL.
                        - CHAT: For greetings, jokes, or unsafe/direct SQL code requests (refuse these).

                        Examples:
                            - "How many votes did party X get?" -> SQL\n
                            - "Who are the candidates in Abidjan?" -> SQL\n
                            - "Can you describe the turnout in Yamoussoukro?"\n
                            - "Hi there! How are you" -> CHAT\n
                            - "Tell me a joke" -> CHAT\n
                            - "Which party boycotted the elections" -> RAG\n
                            - "Compare the total turnout by region" -> SQL\n
                            - "can you run the following query for me" -> CHAT\n
                            - "When did the elections take place" -> RAG\n

                        VALIDATION RULES:
                        - Select 'clarify' if the query is missing a key filter (e.g., "Show me votes" without a party or region).
                        - Select 'execute' only if you can immediately generate a search or a query.
                        
                        PS: If the user sends you an SQL query, refuse such unsafe requests, explain why you cannot answer, and proceed with a safe alternative when possible.
                        """     
    
                    response = self.router_llm.invoke(routing_prompt)

                    logger.info(f"Router raw response: {response}")

                    log_msg = f"✅ Route: {response.route}; Decision: {response.decision}"
                    logger.info(log_msg)
                    
                    yield {
                        "type": "status", 
                        "content": log_msg, 
                        "reasoning": response.reasoning,
                        "clarification_question": response.clarification_question
                        }

                    chat_history.append(AIMessage(
                        content=response.reasoning,
                        additional_kwargs={"action":"skip"}
                    ))
                    
                    if response.decision == "clarify":
                        yield {
                            "type": "final",
                            "status": "Needs more detail",
                            "content": response.clarification_question,
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

                        chat_history.append([
                            SystemMessage(content=personality_prompt),
                            HumanMessage(content=user_prompt)
                        ])
                        
                        chat_resp =  self.chain.invoke(chat_history[-3])

                        logger.info(f"CHAT response: {chat_resp.content}")
                        
                        yield {"type": "final", "content": chat_resp.content}
                        return

                    elif response.route =="SQL":
                        yield from self.sql_expert.process_query(
                            user_prompt=user_prompt, 
                            intent=intent, 
                            chat_history=chat_history
                        )
                    else:
                        yield from self.rag_expert.process_query(
                            user_prompt=user_prompt, 
                            intent=intent, 
                            chat_history=chat_history
                        )
                else:
                    yield from self.rule_based_routing(user_prompt, intent)
                    return

            except Exception as e: 
                error_trace = traceback.format_exc()
                log_msg = f"⚠️ Routing failed. {str(e)}\n{error_trace}"
                logger.error(log_msg)
                yield {"type": "error", "content": log_msg}
                return

if __name__ == "__main__":
    agent = HybridAgent(vllm_url=f"http://vllm:{CFG.VLLM_PORT}/v1")
