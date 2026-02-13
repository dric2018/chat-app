from __init__ import logger, args, client
from config import CFG
import pandas as pd
import pdfplumber
import requests

import sqlite3
import sqlite_vec
from sqlite_vec import serialize_float32

def check_stack_health():
    services = {
        "MLflow": f"http://{CFG.SERVER_IP}:{CFG.MLFLOW_PORT}/health",
        "Nginx": f"http://{CFG.SERVER_IP}:{CFG.NGINX_PORT}",
        "Prometheus": f"http://{CFG.SERVER_IP}:{CFG.PROMETHEUS_PORT}/-/healthy",
        "vLLM": f"http://{CFG.SERVER_IP}:{CFG.VLLM_PORT}/health",
    }
    
    up = False

    for name, url in services.items():
        try:
            res = requests.get(url, timeout=2)
            status = "✅" if res.status_code == 200 else "⚠️"
            logger.info(f"{status} {name}: \t{res.status_code} (url: {url})")

            up = True
        except:
            logger.error(f"❌ {name}: \tUnreachable via url {url}")

    return up

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


def add_embedding(chunk_text:str, db_conn):
    embedding = get_embedding(chunk_text)

    cur = db_conn.execute("INSERT INTO docs (content) VALUES (?)", (chunk_text,))
    doc_id = cur.lastrowid

    db_conn.execute(
        "INSERT INTO vec_index(embedding, content_id) VALUES (?, ?)",
        (sqlite_vec.serialize_float32(embedding), doc_id)
    )
    db_conn.commit()

def get_embedding(text:str):
    """Fetches embedding from vLLM (Ensure vLLM is running an embedding model or --task embed)"""
    response = client.embeddings.create(
        model=CFG.BASE_MODEL,
        input=[text]
    )
    return response.data[0].embedding


def retrieve_context(
    db:sqlite3.Connection, 
    user_query:str, 
    k:int=5
    ):
    """Finds top K relevant chunks from SQLite"""
    query_vector = get_embedding(user_query)
    
    # Search using cosine distance
    cursor = db.execute("""
        SELECT d.content 
        FROM vec_index v
        JOIN docs d ON v.content_id = d.id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY distance
    """, [serialize_float32(query_vector), k])
    
    results = cursor.fetchall()

    return "\n".join([r[0] for r in results])


