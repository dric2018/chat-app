from __init__ import logger

from agents.agent import (
    Agent,  
    db_client, 
    QueryIntent
)

from config import CFG

from datetime import datetime

from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage, 
    HumanMessage, 
    ToolMessage, 
    AIMessage
)

class RAGAgent(Agent):
    def __init__(
            self, 
            vllm_url:str=CFG.VLLM_BASE_URL, 
            ):
        super().__init__(vllm_url)
        self.client.openai_api_base = vllm_url
        
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
        
        limited_history = chat_history[-3:] if chat_history else []

        history_str = ""
        for msg in limited_history:
            role = "User" if msg.type == "human" else "Assistant"
            # Extract text from the message object
            content = msg.content 
            history_str += f"{role}: {content}\n"

        date = datetime.now().strftime("%Y—%m-%d")
        year = date.split("-")[0]
        sys_prompt = f"""
        Today is {date}. We are in the year {year}.\n
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
                    print(observation)
                    
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
                
                final_answer = self.chain.invoke(messages)
                yield {
                    "type": "text",
                    "intent": intent,
                    "content": final_answer.content,
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
