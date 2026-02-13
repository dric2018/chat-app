from __init__ import logger
from config import CFG
import sqlite3

import pdfplumber
import pandas as pd
import sqlite3
import sqlite_vec

import tomllib

from sentence_transformers import SentenceTransformer

class ElectionDB:
    def __init__(
            self, 
            embedding_model_name:str="google/gemma-2b"
        ):
        
        self.embedding_model_name = embedding_model_name
        
        with open("pyproject.toml", "rb") as f:
            self.db_config = tomllib.load(f)["tool"]["db"]

    def init_db(db_path:str):
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        
        # Create normalized relational tables
        conn.execute("DROP TABLE IF EXISTS turnout")
        conn.execute("DROP TABLE IF EXISTS results")
        conn.execute("""
            CREATE TABLE turnout (
                id INTEGER PRIMARY KEY, region TEXT, circonscription TEXT, 
                commune TEXT, registered INT, votants INT, expressed INT, 
                invalid_ballots INT, participation_rate REAL, abstention_rate REAL
            )""")
        conn.execute("""
            CREATE TABLE results (
                id INTEGER PRIMARY KEY, turnout_id INT, candidate_name TEXT, 
                party_group TEXT, votes INT, votes_pct REAL, is_winner INT,
                FOREIGN KEY(turnout_id) REFERENCES turnout(id)
            )""")
        
        # Create table with total numbers extracted from pdf data !!!
        
        # Creating indexed for query optimization
        conn.execute("""CREATE INDEX idx_results_votes ON results(votes DESC);""")
        conn.execute("""CREATE INDEX idx_results_party ON results(party);""")
        conn.execute("""CREATE INDEX idx_turnout_commune ON turnout(commune);""")
        
        # Create the Vector Search Table (using vec0 virtual table)
        conn.execute("DROP TABLE IF EXISTS results_vec")
        conn.execute("CREATE VIRTUAL TABLE results_vec USING vec0(embedding float[384], row_id INTEGER PRIMARY KEY)")
        
        # Create a table for text and a virtual table for vector search
        # conn.execute("CREATE TABLE docs (id INTEGER PRIMARY KEY, content TEXT)")
        # conn.execute("CREATE VIRTUAL TABLE vec_index USING vec0(embedding float[4096], content_id int)")

        return conn
    
    def ingest_pdf(self, pdf_path:str):
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