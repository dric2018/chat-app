from __init__ import logger
from config import CFG
import duckdb

import pdfplumber
import pandas as pd

import re

from sentence_transformers import SentenceTransformer
import sys

from tqdm import tqdm
import traceback


def get_connection():
    return duckdb.connect(
        CFG.DB_PATH,
        read_only=True
    )


def clean_region_name(region:str):
    if region is not None and "\n" in region:
        return region.replace("\n", "")[::-1]
    else:
        return region
    
def summarize_turnout(totals:pd.Series):
    turnout_summary = "These are the overall numbers from the 2025 legislative elections: "

    for k in totals.keys()[1:]:
        if totals[k] not in [None, ''] and totals[k]:
            entry=f"{k}={totals[k]}, "
            turnout_summary+=entry

    turnout_summary = turnout_summary.strip(' ,')

    return turnout_summary

class ElectionDB:
    def __init__(
            self, 
            embedding_model_name:str=CFG.EMBEDDING_MODEL_NAME,
            db_path:str=CFG.DB_PATH
        ):
        
        self.db_path = db_path
        assert embedding_model_name in ["google/embeddinggemma-300m", "sentence-transformers/all-MiniLM-L6-v2"], "Only supporting embeddinggemma-300m, and sentence-transformers/all-MiniLM-L6-v2 as embedding models at the moment."
        
        self.embedding_model_name = embedding_model_name
        if embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2":
            self.embedding_dim = 384
        else:
            self.embedding_dim = 768

        self.embedding_model = None # loaded as needed

    def init_db(
            self 
        ):
        
        logger.info(f"\nInitiating connection to DB: {self.db_path}")
        conn = duckdb.connect(self.db_path)

        try:
            conn.execute(f"""
                DROP SEQUENCE IF EXISTS seq_regions_id;                                                  
                CREATE SEQUENCE seq_regions_id START 1;

                CREATE OR REPLACE TABLE region AS
                SELECT 
                    nextval('seq_regions_id') AS REGION_ID,
                    p.* 
                FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/regions.parquet') p;
            """)    
            logger.info("Successfully created and populated table region")


            conn.execute(f"""
                DROP SEQUENCE IF EXISTS seq_candidates_id;                         
                CREATE SEQUENCE seq_candidates_id START 1;

                CREATE OR REPLACE TABLE candidate AS
                SELECT
                    nextval('seq_candidates_id') AS CANDIDATE_ID,
                    par.PARTY_ID,        
                    p.*                 
                FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/candidates.parquet') p
                JOIN party par ON p.PARTY_NAME = par.PARTY_NAME;
            """)
            logger.info("Successfully created and populated table candidate")

            conn.execute(f"""
                DROP SEQUENCE IF EXISTS seq_parties_id;                         
                CREATE SEQUENCE seq_parties_id START 1;

                CREATE OR REPLACE TABLE party AS
                SELECT
                    nextval('seq_parties_id') AS PARTY_ID,
                    p.*  
                FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/parties.parquet') p
            """)
            logger.info("Successfully created and populated table party")

            conn.execute(f"""
                DROP SEQUENCE IF EXISTS seq_results_id;                         
                CREATE SEQUENCE seq_results_id START 1;

                CREATE OR REPLACE TABLE result AS
                SELECT
                    nextval('seq_results_id') AS RESULT_ID,
                    c.CANDIDATE_ID,           
                    circ.CIRCONSCRIPTION_ID,    
                    par.PARTY_ID,        
                    p.*               
                FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/results.parquet') p
                JOIN candidate c ON p.CANDIDATE_NAME = c.CANDIDATE_NAME
                JOIN party par ON p.PARTY_NAME = par.PARTY_NAME
                JOIN circonscription circ ON p.CIRCONSCRIPTION_NUM = circ.CIRCONSCRIPTION_NUM;
            """)
            logger.info("Successfully created and populated table result")

            # conn.execute("""
            #     ALTER TABLE result ADD COLUMN IF NOT EXISTS is_winner BOOLEAN;
            # """)

            conn.execute(f"""
                DROP SEQUENCE IF EXISTS seq_turnouts_id;                         
                CREATE SEQUENCE seq_turnouts_id START 1;

                CREATE OR REPLACE TABLE turnout AS
                SELECT
                    nextval('seq_turnouts_id') AS TURNOUT_ID,
                    circ.CIRCONSCRIPTION_ID,
                    p.*  
                FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/turnout.parquet') p
                JOIN circonscription circ ON p.CIRCONSCRIPTION_NUM = circ.CIRCONSCRIPTION_NUM;
            """)
            logger.info("Successfully created and populated table turnout")

            conn.execute(f"""
                DROP SEQUENCE IF EXISTS seq_circonscriptions_id;
                CREATE SEQUENCE seq_circonscriptions_id START 1;

                CREATE OR REPLACE TABLE circonscription AS
                SELECT
                    nextval('seq_circonscriptions_id') AS CIRCONSCRIPTION_ID,
                    r.REGION_ID,
                    p.*  
                FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/circonscriptions.parquet') p
                JOIN region r on p.REGION = r.REGION_NAME;
            """)
            logger.info("Successfully created and populated table circonscription")

            conn.execute(f"""
                DROP SEQUENCE IF EXISTS seq_chunk_id;
                CREATE SEQUENCE seq_chunk_id START 1;
                         
                CREATE OR REPLACE TABLE embeddings (
                CHUNK_ID INTEGER PRIMARY KEY,
                ENTITY_TYPE TEXT,        -- result, turnout
                ENTITY_ID INTEGER,      -- result or turnout id (foreign)
                TEXT_CHUNK TEXT,         -- Descriptive text for RAG (created in vw_rag_descriptions)
                EMBEDDING FLOAT[{self.embedding_dim}],
            );
                         
            ALTER TABLE embeddings ALTER CHUNK_ID SET DEFAULT nextval('seq_chunk_id');
            
            INSTALL fts; LOAD fts;
            PRAGMA drop_fts_index('embeddings');
            PRAGMA create_fts_index(
                'embeddings', 'CHUNK_ID', 'TEXT_CHUNK', overwrite=1
            );  
                                     
            """)
            logger.info("Successfully created table embeddings")
            # self.initialize_fts()

            conn.execute("""
                CREATE OR REPLACE TABLE entity_alias (
                    ALIAS TEXT,
                    NORMALIAZED_ALIAS TEXT,
                    ENTITY_TYPE TEXT,
                    ENTITY_ID INTEGER
                );
                """)
            logger.info("Successfully created and populated table entity_alias.")

            logger.info("\nDeploying views...")
            self.deploy_views(conn, "src/db/views.sql")

            self.compute_embeddings()

            logger.info("Views successfully created...closing DB connection.")
            conn.close()

        except Exception as e:
            logger.error(f"Failed to create tables.\nReason: {e}")
            traceback.print_exc()

    def deploy_views(
            self,
            conn, 
            sql_file_path:str
        ):
        try:
            with open(sql_file_path, "r") as f:
                # Split the file by semicolons, but ignore semicolons inside strings/quotes
                statements = re.split(r';(?=(?:[^\'"]*[\'"][^\'"]*[\'"])*[^\'"]*$)', f.read())
                
            for statement in statements:
                stmt = statement.strip()
                if not stmt:
                    continue # Skip empty lines/whitespace
                
                match = re.search(r"VIEW\s+(\w+)", stmt, re.IGNORECASE)
                view_name = match.group(1) if match else "SQL Statement"

                try:
                    conn.execute(stmt)
                    logger.info(f"✅ OK: {view_name}")
                except Exception as e:
                    logger.error(f"❌ ERROR in {view_name}:")
                    logger.error(f"   {str(e).splitlines()[0]}") # Only show the first line of the error
                    # continue to next view to see if others work
                    
        except FileNotFoundError:
            logger.error(f"Could not find view file at {sql_file_path}")

    def get_data_from_pdf(
            self,
            pdf_path:str
        ):
        all_tables      = []
        all_tables_df   = []

        pdf_cols = [
            'REGION',
            'CIRCONSCRIPTION NUM',
            'CIRCONSCRIPTION TITLE',
            'NB BV',
            'REGISTERED',
            'VOTERS',
            'PART RATE',
            'NULL BALL.',
            'EXPRESSED VOTES',
            'NB BLANK',
            'PCT BLANK',
            'PARTY NAME',
            'CANDIDATE NAME',
            'SCORES',
            'PCT SCORE',
            'IS_WINNER'
        ]
        
        settings        = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines", 
            "snap_tolerance": 3,            
            "join_tolerance": 3,            
            "intersection_tolerance": 10, 
        }
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables(table_settings=settings)
                for table in tables:
                    all_tables.append(table)
                    df = pd.DataFrame(table[2:], columns=table[:2])
                    all_tables_df.append(df)
        
        try:
            for df in all_tables_df:
                df.columns = pdf_cols
                
            final_df            = pd.concat(all_tables_df, axis=0, ignore_index=True)
            totals              = final_df.iloc[0] # extract totals line from final table
            final_df            = final_df.drop(index=0).reset_index(drop=True)

            # fix region names
            final_df["REGION"]  = final_df["REGION"].apply(lambda r: clean_region_name(r))
            final_df            = final_df.ffill()

            return totals, final_df, all_tables, all_tables_df
        
        except Exception as e:
            print(f" Error while merging data: {e}")
            return None, None, all_tables, all_tables_df
    
    def load_embedding_model(self):
        self.embedding_model = SentenceTransformer(self.embedding_model_name)

    def compute_embeddings(
            self, 
            batch_size:int=100
        ):

        self.load_embedding_model()
        assert self.embedding_model is not None, "Failed to properly load mbedding model"
        
        conn = duckdb.connect(str(CFG.DB_PATH))

        logger.info("Fetching descriptions from vw_rag_descriptions...")
        data = conn.execute("SELECT text_chunk, entity_type, entity_id FROM vw_rag_descriptions").fetchall()

        if not data:
            logger.warning("No data to embed!")
            return

        conn.execute("DELETE FROM embeddings") # Clear old data

        logger.info(f"Embedding {len(data)} chunks in batches of {batch_size}...")
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            texts = [r[0] for r in batch]
            
            embeddings = self.embedding_model.encode(texts).tolist()

            insert_data = []
            for j, row in enumerate(batch):
                # row: (text, type, id) + embedding
                insert_data.append((row[1], row[2], row[0], embeddings[j]))

            conn.executemany("""
                INSERT INTO embeddings (ENTITY_TYPE, ENTITY_ID, TEXT_CHUNK, EMBEDDING)
                VALUES (?, ?, ?, ?)
            """, insert_data)

        logger.info(f"✅ RAG table synced with {len(data)} vectors.")

    def vector_search(
            self,
            query: str, 
            top_k: int = CFG.TOP_K 
        ):
        with duckdb.connect(str(CFG.DB_PATH)) as conn:
            self.load_embedding_model()
            assert self.embedding_model is not None, "Failed to properly load mbedding model"
            
            query_vector = self.embedding_model.encode(query).tolist()

            # Note: we cast the query_vector to FLOAT[384] to match the column
            results = conn.execute(f"""
                SELECT 
                    TEXT_CHUNK, 
                    array_cosine_similarity(EMBEDDING, ?::FLOAT[{self.embedding_dim}]) AS similarity
                FROM embeddings
                ORDER BY similarity DESC
                LIMIT ?
            """, [query_vector, top_k]).fetchall()
        

        return results

    def initialize_fts(self):
        """To enable Full-Text Search on embeddings table."""

        logger.info("Initializing FTS.")

        with duckdb.connect(str(CFG.DB_PATH)) as conn:
            conn.execute("INSTALL fts; LOAD fts;")
            
            # Drop if exists
            conn.execute("PRAGMA drop_fts_index('embeddings')")
            
            conn.execute("""
                PRAGMA create_fts_index(
                    'embeddings', 'CHUNK_ID', 'TEXT_CHUNK', overwrite=1
                )
            """)
        logger.info("FTS Index initialized successfully.")


    def full_text_search(
            self, 
            query: str, 
            top_k: int = CFG.TOP_K 
        ):
        """
        Direct keyword search using DuckDB's FTS extension.
        """
        with duckdb.connect(str(CFG.DB_PATH)) as conn:
            conn.execute("LOAD fts;")

            sql = f"""
                SELECT 
                    TEXT_CHUNK, 
                    fts_main_embeddings.match_bm25(CHUNK_ID, ?) AS score
                FROM embeddings
                WHERE score IS NOT NULL
                ORDER BY score DESC
                LIMIT ?
            """
            
            results = conn.execute(sql, [query, top_k]).fetchall()

        return results
    
    def hybrid_search(
            self, 
            query: str, 
            top_k: int = CFG.TOP_K, 
            k_rrf: int = 60
        ):
        self.load_embedding_model()
        assert self.embedding_model is not None, "Failed to properly load mbedding model"
        
        with duckdb.connect(str(CFG.DB_PATH)) as conn:
            query_vector = self.embedding_model.encode(query).tolist()

            # Unified query using Reciprocal Rank Fusion (RRF)
            sql = f"""
            WITH vector_results AS (
                SELECT CHUNK_ID, TEXT_CHUNK, 
                    ROW_NUMBER() OVER (ORDER BY array_cosine_similarity(EMBEDDING, ?::FLOAT[{self.embedding_dim}]) DESC) as rank
                FROM embeddings
                LIMIT 50
            ),
            fts_results AS (
                SELECT CHUNK_ID, TEXT_CHUNK, 
                    ROW_NUMBER() OVER (ORDER BY fts_main_embeddings.match_bm25(chunk_id, ?) DESC) as rank
                FROM embeddings
                WHERE fts_main_embeddings.match_bm25(chunk_id, ?) IS NOT NULL
                LIMIT 50
            )
            SELECT 
                COALESCE(v.TEXT_CHUNK, f.TEXT_CHUNK) as TEXT_CHUNK,
                (COALESCE(1.0 / (? + v.rank), 0.0) + COALESCE(1.0 / (? + f.rank), 0.0)) AS rrf_score
            FROM vector_results v
            FULL OUTER JOIN fts_results f ON v.CHUNK_ID = f.CHUNK_ID
            ORDER BY rrf_score DESC
            LIMIT ?
            """

            results = conn.execute(sql, [query_vector, query, query, k_rrf, k_rrf, top_k]).fetchall()
            
        return results


if __name__=="__main__":

    db_client = ElectionDB()
    db_client.init_db()