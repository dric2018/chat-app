from __init__ import logger, args
import pandas as pd
import pdfplumber
import requests

import streamlit as st

import sqlite3
from prometheus_client import start_http_server, Summary, Counter

def check_stack_health():
    services = {
        "vLLM": "http://localhost:8000/v1/models",
        "Streamlit": "http://localhost:8501/_stcore/health",
        "Prometheus": "http://localhost:9090/-/healthy",
        "MLflow": "http://localhost:5000/health"
    }
    
    up = False

    for name, url in services.items():
        try:
            res = requests.get(url, timeout=2)
            status = "✅" if res.status_code == 200 else "⚠️"
            logger.info(f"{status} {name}: {res.status_code}")

            up = True
        except:
            logger.error(f"❌ {name}: Unreachable")

    return up

# Measuring how long SQL execution takes
SQL_TIME = Summary('sql_execution_seconds', 'Time spent executing safe SQL')


# Starting the metrics server on a separate port inside the container
start_http_server(8000)

@st.cache_resource
def start_metrics_server():
    # Start server on port 8000 inside the container
    start_http_server(8000)
    return {
        "sql_latency": Summary('sql_op_seconds', 'Time spent on SQLite ops'),
        "attacks": Counter('sql_attacks_total', 'Blocked adversarial queries')
    }


@SQL_TIME.time()
def run_safe_query(query, db_path="data/election_results.db"):
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT"]
    if any(cmd in query.upper() for cmd in forbidden):
        return None, "⚠️ Security Violation: Unauthorized Command."
    
    try:
        # Connect in Read-Only mode
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df, None
    except Exception as e:
        return None, f"SQL Error: {str(e)}"


def ingest_election_pdf(pdf_path, db_path):
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                all_tables.append(df)
    
    # Merge and clean (Ensure numeric types for SQL aggregations)
    final_df = pd.concat(all_tables)
    final_df['votes'] = pd.to_numeric(final_df['votes'].str.replace(',', ''), errors='coerce')
    
    conn = sqlite3.connect(db_path)
    final_df.to_sql("election_results", conn, if_exists="replace", index=False)
    
    # Create the Semantic View (Bonus A)
    conn.execute("CREATE VIEW vw_participation AS SELECT region, AVG(participation_rate) as rate FROM election_results GROUP BY region")
    conn.close()
    logger.info("✅ Ingestion complete: SQLite DB created.")
