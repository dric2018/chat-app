from __init__ import logger
from config import CFG

from openai import OpenAI

import pandas as pd
from prometheus_client import Counter, Histogram, start_http_server

import sqlite3
import sqlparse
from enum import Enum

class QueryIntent(Enum):
    AGGREGATION     = "aggregation" # e.g., "Total votes in Region X"
    RANKING         = "ranking"           # e.g., "Top 3 candidates by percentage"
    CHART           = "chart"               # e.g., "Show me the distribution of votes"
    GENERAL         = "general"           # Standard lookup
    INVALID         = "invalid"           # Out of scope


class SQLAgent:
    """
    Example Usage:
        executor = SQLAgent("elections.db")
        Adversarial Test: executor.execute("DROP TABLE election_results;") -> Blocks

    Example Usage:
        agent = SQLAgent("elections.db")
        Valid queries:
        print(agent.execute("Who won in Commune X?"))       # Triggers SQL generation
        Adversarial Tests: 
        executor.execute("DROP TABLE election_results;") -> Blocks
        print(agent.execute("What is the weather in Paris?")) # Triggers topic block
        Adversarial Test: agent.execute("Delete or modify results table") -> # Triggers topic block


    """
    def __init__(
            self, 
            db_path:str,
            embedding_model_name:str="google/gemma-2b"
        ):

        self.forbidden              = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE"]
        self.db_path                = db_path
        self.embedding_model_name   = embedding_model_name
        self.client                 = OpenAI(base_url="http://vllm:8000/v1", api_key="token")
        self.results_limit          = CFG.SQL_MAX_LIMIT

        # Schema Metadata for the LLM
        self.schema_context = """
            Table: turnout (columns: region, circonscription, commune, registered, votants, expressed, invalid_ballots, participation_rate, abstention_rate)
            Table: results (columns: candidate_name, party_group, votes_count, votes_pct, is_winner, turnout_id)
            Relationship: results.turnout_id = turnout.id
            """

        # Metrics to track
        self.SQL_VIOLATIONS = Counter(
            'rag_sql_security_violations_total', 
            'Total number of blocked malicious or invalid SQL queries',
            ['reason'] # e.g., 'multiple_statements', 'forbidden_keyword'
        )

        self.QUERY_LATENCY = Histogram(
            'rag_query_duration_seconds', 
            'Time spent generating and executing SQL'
        )


        self.INTENT_COUNTER = Counter(
            'rag_query_intent_total', 
            'Total queries by intent type', 
            ['intent']
        )


    def _get_intent(self, user_prompt:str):
        intent_prompt = f"""
        Classify the user's election query into one category:
        - AGGREGATION: Sums, counts, averages (e.g., 'total votes', 'turnout average').
        - RANKING: Comparisons, top/bottom lists (e.g., 'who won', 'best party').
        - CHART: Requests for distribution, trends, or visualizations.
        - GENERAL: Simple fact retrieval.
        - INVALID: Not about the election.

        Query: "{user_prompt}"
        Respond ONLY with the category name.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": intent_prompt}]
        )
        label = response.choices[0].message.content.strip().upper()
        try:
            return QueryIntent[label]
        except KeyError:
            return QueryIntent.INVALID


    def _is_relevant(self, user_prompt:str):
        """Intent Classification (Gatekeeper)"""
        check_prompt = f"""
        Determine if this query is about election results, turnout, or political candidates.
        Query: "{user_prompt}"
        Respond ONLY with 'YES' or 'NO'.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": check_prompt}]
        )
        return "YES" in response.choices[0].message.content.upper()

    def generate_sql(self, user_prompt:str, intent:QueryIntent):
        """Restricted SQL Generation"""

        intent_instructions = {
            QueryIntent.AGGREGATION: "Use GROUP BY and SUM/AVG. Ensure results are numeric.",
            QueryIntent.RANKING: "Use ORDER BY votes DESC and LIMIT 10.",
            QueryIntent.CHART: "Return exactly two columns: a label (e.g., party) and a numeric value."
        }
        generation_prompt = f"""
        You are an Election Data and SQL Expert. Given this schema:
        {self.schema_context}
        Intent: {intent.value}. {intent_instructions.get(intent, "")}
        Generate a single valid SQL SELECT statement. 
        - Use JOINs where necessary.
        - Never use {self.forbidden} statements!
        - If the question is too vague, return: SELECT 'NO_DATA'
        - Always include a LIMIT {self.results_limit}.
        - Output ONLY the SQL string.
        """
        
        response = self.client.chat.completions.create(
            model=self.embedding_model_name,
            messages=[
                {"role": "system", "content": generation_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content.strip().replace("```sql", "").replace("```", "")

    def validate_sql(self, sql:str):
        """
            Strict Structural Guardrails. 
            Additional validation layer after system prompt
        """
        sql = sql.upper()
        try:
            # Checking for multiple statements (Piggybacking attack)
            # e.g., "SELECT * FROM results; DROP TABLE turnout;"
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
                    return False, f"Security Violation: Forbidden keyword '{token.value}' detected."

            return True, "Valid"
        
        except Exception as e:
            return False, f"Parsing Error: {str(e)}"

    def execute(self, user_prompt:str):
        # Relevance Check
        if not self._is_relevant(user_prompt):
            return self._format_not_found(user_prompt, "Topic not related to election data.")

        # Generate and Validate SQL
        intent = self._get_intent(user_prompt)
        self.INTENT_COUNTER.labels(intent=intent.value).inc()

        if intent == QueryIntent.CHART:
            # Ensure SQL always returns ['label', 'value']
            # This allows a single Grafana Bar Chart panel to stay "fixed" 
            # while the data inside it changes based on the user's prompt.
            return {"type": "chart_data", "data": df.to_dict(orient='records')}

        
        sql = self.generate_sql(user_prompt, intent)

        is_safe, message = self.validate_sql(sql)

        if not is_safe:
            # Log this as a security event
            logger.info(f"ALERT: {message}")
            
            # Increment the Prometheus counter
            self.SQL_VIOLATIONS.labels(reason=message.split(":")[0]).inc()
            
            return self._format_not_found(
                user_prompt, 
                f"I attempted to generate a query, but it failed security validation.\nSecurity Block: {message}"
            )
        
        # Execute query
        try:
            # Open db in Read-Only mode
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            cursor = conn.cursor()
            
            # Auto-append LIMIT if missing
            if "LIMIT" not in sql.upper():
                sql = sql.rstrip(';') + f" LIMIT {self.results_limit}"
                
            cursor.execute(sql)
            rows = cursor.fetchall()
            # pd.read_sql_query handles the cursor and column names automatically
            df = pd.read_sql_query(sql, conn)
            conn.close()

            if not rows or (len(rows) == 1 and rows[0][0] == 'NO_DATA'):
                return self._format_not_found(user_prompt, f"Query executed: {sql}")

            return self._interpret_results(user_prompt, rows)

        except Exception as e:
            return self._format_not_found(user_prompt, f"Database Error: {str(e)}")

    def _interpret_results(self, user_prompt:str, rows):
        """Converts raw SQLite rows into a natural language answer."""
        interp_prompt = f"User asked: {user_prompt}\nData found: {rows}\nAnswer concisely:"
        res = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": interp_prompt}]
        )
        return res.choices[0].message.content.strip()

    def _format_not_found(self, prompt:str, logic_path):
        """Standardized failure response per your requirements."""
        return (
            f"**Not found in the provided PDF dataset.**\n\n"
            f"*Attempted:* {logic_path}\n"
            f"*Suggestions:* \n"
            f"- Try specifying a specific Commune or Region.\n"
            f"- Use candidate last names instead of full names.\n"
            f"- Ask specifically about 'votes', 'turnout', or 'participation'."
        )
