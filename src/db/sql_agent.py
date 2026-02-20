from __init__ import logger
from config import CFG

import duckdb

from enum import Enum

from src.db.election_db import ElectionDB

import sqlparse

import time

class QueryIntent(Enum):
    AGGREGATION = "aggregation"
    RANKING     = "ranking"
    CHART       = "chart"
    GENERAL     = "general"
    INVALID     = "invalid"

class SQLAgent:
    def __init__(
            self, 
            db_path:str,
            model_endpoint:str=CFG.VLLM_BASE_URL
        ):

        self.db_path    = db_path
        self.model_name = CFG.BASE_MODEL
        self.client     = CFG.client
        self.db_client  = ElectionDB()
        
        self.forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
        self.results_limit = CFG.SQL_MAX_LIMIT

        # Schema Metadata for the LLM
        self.schema_context = """
            Tables:
            - Table: turnout (columns: region, circonscription, commune, registered, votants, expressed, invalid_ballots, participation_rate, abstention_rate)
            - Table: results (columns: candidate_name, party_group, votes_count, votes_pct, is_winner, turnout_id)
            
            Views
            - vw_results (RESULT_ID, REGION_NAME, CIRCONSCRIPTION_TITLE, CANDIDATE_NAME, PARTY_NAME, SCORES, PCT_SCORE, IS_WINNER)
            - vw_turnout (REGION_NAME, CIRC_NAME, REGISTERED, BALLOTS_CAST, TURNOUT_PCT)
            - vw_rag_descriptions (Used for semantic search, contains text_chunk)
            """

        self.metrics = {
            "violations": 0,
            "total_latency": 0.0,
            "intents": {intent.value: 0 for intent in QueryIntent}
        }

    def _get_intent(self, user_prompt:str)-> QueryIntent:
        intent_prompt = f"""
        Classify the user's election query into one category:
        - AGGREGATION: Sums, counts, averages (e.g., 'total votes/scores', 'turnout average').
        - RANKING: Comparisons, top/bottom lists (e.g., 'who won', 'best party').
        - CHART: Requests for distribution, trends, or visualizations.
        - GENERAL: Simple fact retrieval.
        - INVALID: Not about the election.

        Query: "{user_prompt}"
        Respond ONLY with the category name.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": intent_prompt}],
                temperature=CFG.GENERATION_TEMPERATURE
            )
            
            label = response.choices[0].message.content.strip().upper()
            intent = QueryIntent[label] if label in [i.name for i in QueryIntent] else QueryIntent.INVALID
            
            self.metrics["intents"][intent.value] += 1
            
            return intent        

        except KeyError:
            return QueryIntent.INVALID

    def generate_sql(self, user_prompt:str, intent:QueryIntent, history):
        """Restricted SQL Generation"""

        intent_instructions = {
            QueryIntent.AGGREGATION: "Use GROUP BY and SUM/AVG. Ensure results are numeric.",
            QueryIntent.RANKING: f"Use ORDER BY votes DESC and LIMIT {CFG.SQL_MAX_LIMIT}.",
            QueryIntent.CHART: "Return exactly two columns: a label (e.g., party) and a numeric value."
        }

        generation_prompt = f"""
        You are a Data Scientist with deep expertise in elections data management, exploration, and an SQL Expert. 
        Given this schema as context:
        {self.schema_context}
        Intent: {intent.value}. {intent_instructions.get(intent, "")}
        Generate a single valid SQL SELECT statement. 
        - Use JOINs where necessary.
        - Never use {self.forbidden} statements. HARD CONSTRAINT!
        - If the question is too vague, return: SELECT 'NO_DATA'
        - Always include a LIMIT {self.results_limit}.
        - Output ONLY the SQL string.
        
        Exceptions:
            - If the query is ambiguous, you may ask for clarifications
            Examples:
                - "Who won in Tiapoum?" (ambiguous; a win is equivalent to being elected)
                - "Show turnout in Abidjan." (scope ambiguity but looking for all related circonscription)
                - "Top 5 in Grand-Bassam."(potential naming collisions)
        """
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": generation_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=CFG.GENERATION_TEMPERATURE,
            stream=CFG.IS_STREAM,
            max_tokens=CFG.MAX_TOKENS
        )
        return response.choices[0].message.content.strip().replace("```sql", "").replace("```", "").split(';')[0]

    def validate_sql(self, sql:str):
        """
            Strict Structural Guardrails. 
            Additional validation layer after system prompt
        """
        sql = sql.upper()
        try:
            parsed = sqlparse.parse(sql)
            if len(parsed) > 1:
                return False, "Security Violation: Multiple queries detected."
            
            clean_sql = parsed[0]
            
            # Checking for the Command Type
            # We ensure the very first 'DML' (Data Manipulation Language) token is 'SELECT'
            if clean_sql.get_type() != "SELECT":
                return False, f"Security Violation: Forbidden command type '{clean_sql.get_type()}'."

            # Deep Token Inspection (No hidden JOINs to sensitive tables)
            # We check every token for forbidden keywords that might bypass get_type()
            for token in clean_sql.flatten():
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() in self.forbidden:
                    self.metrics["violations"] += 1
                    return False, f"Security Violation: Forbidden keyword '{token.value}' detected."
            
            if "LIMIT" not in clean_sql:
                clean_sql += f" LIMIT {CFG.SQL_MAX_LIMIT}"

            return True, "Valid"
        
        except Exception as e:
            return False, f"Parsing Error: {str(e)}"

    def execute(self, user_prompt:str):
        start_time = time.time()
        
        intent = self._get_intent(user_prompt)

        if intent == QueryIntent.INVALID:
            return self._format_not_found(user_prompt, "Invalid Intent")

        generated_sql = self.generate_sql(user_prompt, intent)
        
        if not self.validate_sql(generated_sql):
            logger.error(f"SQL Violation Attempted: {generated_sql}")
            return {"type": "text", "content": "Security violation."}

        try:
            with duckdb.connect(self.db_path, read_only=True) as conn:
                df = conn.execute(generated_sql).df()
                
            if df.empty:
                return {"type": "text", "content": "No data found."}
                            
            self.metrics["total_latency"] += (time.time() - start_time)

            return {
                "type": "data",
                "intent": intent,
                "data": df,
                "interpretation": self._interpret_results(user_prompt, df.values.tolist(), df.columns.tolist())
            }  
              
        except Exception as e:
            logger.error(f"Execution Error: {e}")
            return f"I encountered an error analyzing the data: {e}"

    def _interpret_results(self, user_prompt: str, rows):
        if not rows:
            return "I found no data matching your request."
        
        interp_prompt = f"User asked: {user_prompt}. Data results: {rows}. Summarize as a natural response."
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": interp_prompt}]
        )
        return response.choices[0].message.content.strip()

    def _format_not_found(self, prompt:str, logic_path):
        """Standardized failure response per your requirements."""
        return (
            f"**Not found in the provided PDF dataset.**\n\n"
            f"User asked: {prompt}\n\n"
            f"*Attempted:* {logic_path}\n"
        )

class HybridAgent(SQLAgent):
    
    def __init__(self, db_path:str, model_endpoint:str = CFG.VLLM_BASE_URL):
        super().__init__(db_path, model_endpoint)
    
        self.db_client = ElectionDB(db_path=db_path)

    def route(self, user_prompt: str):
        """Decide between SQL (analytics) and RAG (narrative/fuzzy)."""
        routing_prompt = f"""
        Determine the best system for this query: 'SQL' for analytics (counts, rankings, aggregations, chart, any analytics-based query) or 'RAG' for narrative fact lookup (who won, what happened).
        Query: "{user_prompt}"
        Respond ONLY with 'SQL' or 'RAG'.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": routing_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=CFG.GENERATION_TEMPERATURE,
            stream=CFG.IS_STREAM,
            max_tokens=CFG.MAX_TOKENS
        )

        path = response.choices.message.content.strip().upper()

        if path == 'SQL':
            return self.execute(user_prompt) # Uses the SQL path

        elif path == 'RAG':
            context_chunks = self.db_client.vector_search(user_prompt)
            # Feed chunks to an LLM for interpretation
            return self._interpret_rag_results(user_prompt, context_chunks)
        else:
            return self._format_not_found(user_prompt, "Routing Error")
