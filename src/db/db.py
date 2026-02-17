from __init__ import logger
from config import CFG
import duckdb

import pdfplumber
import pandas as pd

import tomllib

from sentence_transformers import SentenceTransformer


def get_connection():
    return duckdb.connect(
        CFG.DB_PATH,
        read_only=True
    )


class ElectionDB:
    def __init__(
            self, 
            embedding_model_name:str="google/gemma-2b"
        ):
        
        self.embedding_model_name = embedding_model_name
        
        with open("pyproject.toml", "rb") as f:
            self.db_config = tomllib.load(f)["tool"]["db"]

    def init_db(db_path:str):
        conn = duckdb.connect(CFG.DB_PATH)

        conn.execute(f"""
            CREATE OR REPLACE TABLE dim_region AS
            SELECT * FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/dim_region.parquet')
        """)

        conn.execute(f"""
            CREATE OR REPLACE TABLE dim_candidate AS
            SELECT * FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/dim_candidate.parquet')
        """)

        conn.execute(f"""
            CREATE OR REPLACE TABLE dim_party AS
            SELECT * FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/dim_party.parquet')
        """)

        conn.execute(f"""
            CREATE OR REPLACE TABLE fact_results AS
            SELECT * FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/fact_results.parquet')
        """)

        conn.execute("""
            ALTER TABLE fact_results ADD COLUMN IF NOT EXISTS is_winner BOOLEAN;
        """)

        conn.execute("""
            UPDATE fact_results
            SET is_winner = (rank = 1);
        """)

        with open("src/db/views.sql") as f:
            conn.execute(f.read())

        conn.close()
        return conn
    
    def get_data_from_pdf(self, pdf_path:str):
        all_tables      = []
        all_tables_df   = []
        settings        = {
            "vertical_strategy": "lines",   # Excel usually exports with clear vertical lines
            "horizontal_strategy": "lines", # Use "text" if the rows don't have borders
            "snap_tolerance": 4,            # Merge lines that are nearly touching
            "join_tolerance": 4,            # Join broken lines
            "intersection_tolerance": 10,   # Helps with merged cells
        }
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables(table_settings=settings)
                for table in tables:
                    all_tables.append(table)
                    df = pd.DataFrame(table[2:], columns=table[:2])
                    all_tables_df.append(df)
        
        try:
            # Merge and clean (Ensure numeric types for SQL aggregations)
            final_df    = pd.concat(all_tables_df)
            totals      = final_df.iloc[0]
            final_df    = final_df.drop(index=0).reset_index(drop=True)
            
            # Fix headers
            raw_headers = [h.lstrip().strip() if isinstance(h, str) else h for h in tables[0][:2]]
            new_headers = []

            for (r1, r2) in zip(raw_headers[0], raw_headers[1]):
                if r1 is None or r1=='':
                    r1 = '-'
                if r2 is None or r2=='':
                    r2 = '-'
                new_header = "".join([r1, r2])
                new_headers.append(new_header)
            
            final_df.columns = new_headers
        
            return totals, final_df, all_tables, all_tables_df
        
        except Exception as e:
            print(f" Error while merging data: {e}")
            return None, None, all_tables, all_tables_df
        
    def ingest_data(self, conn, df):
        
        model = SentenceTransformer(self.embedding_model_name)
        
        for _, row in df.iterrows():
            # Insert into turnout
            cur = conn.execute(
                "INSERT INTO turnout (region, commune, participation_rate, abstention_rate) VALUES (?, ?, ?)",
                (row.get('REGION'), row.get('COMMUNE'), row.get('PARTICIPATION'), row.get('ABSTENTION'))
            )
            turnout_id = cur.lastrowid
            
            # Insert into results
            conn.execute(
                "INSERT INTO results (turnout_id, candidate_name, party_group, votes_count, votes_percent) VALUES (?, ?, ?, ?)",
                (turnout_id, row.get('CANDIDATE_NAME'), row.get('PARTY_GROUP'), row.get('VOTES_COUNT'), row.get('VOTES_PERCENT'))
            )
            
            # Create RAG Chunk & Embedding
            chunk_text = f"In {row['REGION']}, {row['CANDIDATE_NAME']} ({row['PARTY_GROUP']}) got {row['VOTES_COUNT']} votes ({row['VOTES_PERCENT']}%)."
            embedding = model.encode(chunk_text).tolist()
            
            # Insert into Vector Table (using row_id to link back to the relational data)
            conn.execute(
                "INSERT INTO results_vec(row_id, embedding) VALUES (?, ?)",
                (turnout_id, str(embedding))
            )
        
        conn.commit()

    def populate_db(
            self, 
            db_path:str, 
            data_path:str
        ):
        conn = self.init_db(db_path)

        try:
            totals, df, _, _ = self.ingest_pdf(data_path)

        except Exception as e:
            logger.error(f"Error while processing PDF: {e}")

        try:
            self.ingest_data(conn, df)
            logger.info("Ingestion Complete! You can now perform SQL or Vector search.")
        except Exception as e:
            logger.error(f"Error during ingestion: {e}")