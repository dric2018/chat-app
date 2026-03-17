from __init__ import logger
from config import CFG
import duckdb

import pandas as pd

import re

from sentence_transformers import SentenceTransformer

from tqdm import tqdm
import traceback


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
            truncate_embedding_dim:bool=True,
            db_path:str=CFG.DB_PATH
        ):
        
        self.db_path = db_path
        assert embedding_model_name in ["google/embeddinggemma-300m", "sentence-transformers/all-MiniLM-L6-v2"], "Only supporting embeddinggemma-300m, and sentence-transformers/all-MiniLM-L6-v2 as embedding models at the moment."
        
        self.embedding_model_name = embedding_model_name

        if truncate_embedding_dim:
            self.embedding_dim = CFG.TARGET_EMBEDDING_DIM
        else:
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
            logger.info("✅ Successfully created and populated table region")

            conn.execute(f"""
                DROP SEQUENCE IF EXISTS seq_parties_id;                         
                CREATE SEQUENCE seq_parties_id START 1;

                CREATE OR REPLACE TABLE party AS
                SELECT
                    nextval('seq_parties_id') AS PARTY_ID,
                    p.*  
                FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/parties.parquet') p
            """)
            logger.info("✅ Successfully created and populated table party")


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
            logger.info("✅ Successfully created and populated table candidate")

            conn.execute(f"""
                DROP SEQUENCE IF EXISTS seq_constituencies_id;
                CREATE SEQUENCE seq_constituencies_id START 1;

                CREATE OR REPLACE TABLE constituency AS
                SELECT
                    nextval('seq_constituencies_id') AS CONSTITUENCY_ID,
                    r.REGION_ID,
                    p.*  
                FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/constituencies.parquet') p
                JOIN region r on p.REGION = r.REGION_NAME;
            """)
            logger.info("✅ Successfully created and populated table constituency")

            conn.execute(f"""
                DROP SEQUENCE IF EXISTS seq_results_id;                         
                CREATE SEQUENCE seq_results_id START 1;

                CREATE OR REPLACE TABLE result AS
                SELECT
                    nextval('seq_results_id') AS RESULT_ID,
                    c.CANDIDATE_ID,           
                    circ.CONSTITUENCY_ID,    
                    par.PARTY_ID,   
                    par.PARTY_NAME,   
                    p.*               
                FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/results.parquet') p
                JOIN candidate c ON p.CANDIDATE_NAME = c.CANDIDATE_NAME
                JOIN party par ON p.PARTY_NAME = par.PARTY_NAME
                JOIN constituency circ ON p.CONSTITUENCY_NUM = circ.CONSTITUENCY_NUM;
            """)
            logger.info("✅ Successfully created and populated table result")

            conn.execute(f"""
                DROP SEQUENCE IF EXISTS seq_turnouts_id;                         
                CREATE SEQUENCE seq_turnouts_id START 1;

                CREATE OR REPLACE TABLE turnout AS
                SELECT
                    nextval('seq_turnouts_id') AS TURNOUT_ID,
                    circ.CONSTITUENCY_ID,
                    p.*  
                FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/turnout.parquet') p
                JOIN constituency circ ON p.CONSTITUENCY_NUM = circ.CONSTITUENCY_NUM;
            """)
            logger.info("✅ Successfully created and populated table turnout")

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
            
            INSTALL fts;                         
            """)
            logger.info("✅ Successfully created table embeddings")
            
            # Alias table only for constituency
            conn.execute(f"""
                INSTALL rapidfuzz FROM community;
                LOAD rapidfuzz;
                         
                DROP SEQUENCE IF EXISTS seq_alias_id;                         
                CREATE SEQUENCE seq_alias_id START 1;
                                                  
                CREATE OR REPLACE TABLE entity_alias AS
                SELECT
                    nextval('seq_alias_id') AS ALIAS_ID,
                    circ.CONSTITUENCY_ID,
                    p.*  
                FROM read_parquet('{CFG.PROCESSED_DATA_DIR}/entity_alias.parquet') p
                JOIN constituency circ ON p.CONSTITUENCY_NUM = circ.CONSTITUENCY_NUM;
                """)

            logger.info("✅ Successfully created and populated table entity_alias.")

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
                    raise RuntimeError(f"   {str(e).splitlines()[0]}") # Only show the first line of the error
                    
        except FileNotFoundError:
            logger.error(f"Could not find view file at {sql_file_path}")

    def vector_search(
            self,
            query: str, 
            reasoning:str,
            top_k: int = CFG.TOP_K
        ):

        """
            Performing vector search over the client database
        """

        logger.info(f"LLM Reasoning (vector_search): {reasoning}")

        with duckdb.connect(str(CFG.DB_PATH)) as conn:
            self.load_embedding_model()
            assert self.embedding_model is not None, "Failed to properly load mbedding model"
            
            query_vector = self.embedding_model.encode(query).tolist()

            results = conn.execute(f"""
                SELECT 
                    TEXT_CHUNK, CHUNK_ID,
                    array_cosine_similarity(EMBEDDING, ?::FLOAT[{self.embedding_dim}]) AS similarity
                FROM embeddings
                ORDER BY similarity DESC
                LIMIT ?
            """, [query_vector, top_k]).fetchall()
        

        return results

    def full_text_search(
        self,
            query: str, 
            top_k: int = CFG.TOP_K 
        ):
        """
        Direct keyword search using DuckDB's Full-Text search extension.
        """
        with duckdb.connect(str(CFG.DB_PATH)) as conn:
            conn.execute("LOAD fts;")

            sql = f"""
                SELECT 
                    TEXT_CHUNK, CHUNK_ID,
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
            k_rrf: int = 60,
            weight_vs: float = 0.7,
            weight_fts: float = 0.3 
        ):
        """
            Performing hybrid (vector+full-text) search over the client database
        """
        self.load_embedding_model()
        assert self.embedding_model is not None, "Failed to properly load mbedding model"
        
        with duckdb.connect(str(CFG.DB_PATH)) as conn:
            conn.execute("LOAD fts;")
            
            query_vector = self.embedding_model.encode(query).tolist()

            # Unified query using Reciprocal Rank Fusion (RRF)
            sql = f"""
            WITH vector_results AS (
                SELECT CHUNK_ID, TEXT_CHUNK, 
                    ROW_NUMBER() OVER (ORDER BY array_cosine_similarity(EMBEDDING, ?::FLOAT[{self.embedding_dim}]) DESC) as rank
                FROM embeddings
                LIMIT 100
            ),
            fts_results AS (
                SELECT CHUNK_ID, TEXT_CHUNK, 
                    ROW_NUMBER() OVER (ORDER BY fts_main_embeddings.match_bm25(chunk_id, ?) DESC) as rank
                FROM embeddings
                WHERE fts_main_embeddings.match_bm25(chunk_id, ?) IS NOT NULL
                LIMIT 100
            )
            SELECT 
                COALESCE(v.TEXT_CHUNK, f.TEXT_CHUNK) as TEXT_CHUNK,
                COALESCE(v.CHUNK_ID, f.CHUNK_ID) as CHUNK_ID,
                (? * COALESCE(1.0 / (? + v.rank), 0.0) + ? * COALESCE(1.0 / (? + f.rank), 0.0)) AS weighted_rrf_score
            FROM vector_results v
            FULL OUTER JOIN fts_results f ON v.CHUNK_ID = f.CHUNK_ID
            ORDER BY weighted_rrf_score DESC
            LIMIT ?
            """

            params = [
                query_vector, 
                query, 
                query, 
                weight_vs, 
                k_rrf, 
                weight_fts, 
                k_rrf, 
                top_k
            ]
            results = conn.execute(sql, params).df().drop_duplicates(subset=['CHUNK_ID'])

                   
        return results       
    
    def load_embedding_model(self):
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            truncate_dim=self.embedding_dim
        )

    def compute_embeddings(
            self, 
            batch_size:int=100
        ):
        logger.info("Computing embeddings...")
        self.load_embedding_model()
        assert self.embedding_model is not None, "Failed to properly load mbedding model"
        
        with duckdb.connect(str(CFG.DB_PATH)) as conn:
            logger.info("Fetching descriptions from vw_rag_descriptions...")
            data = conn.execute("SELECT text_chunk, entity_type, entity_id FROM vw_rag_descriptions").fetchall()

            if not data:
                logger.warning("No data to embed!")
                return

            conn.execute("DELETE FROM embeddings") # Clear old data

            logger.info(f"Embedding {len(data)} chunks in batches of {batch_size}...")
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                texts = [r[0] for r in batch if r[0]] # Filter out None/empty strings
                
                if not texts:
                    continue

                # Get the raw output first to check for None
                raw_embeddings = self.embedding_model.encode(texts)
                
                if raw_embeddings is None:
                    logger.error(f"Embedding model returned None for batch {i}")
                    continue
                    
                embeddings = raw_embeddings.tolist()

                insert_data = []
                for j, row in enumerate(batch):
                    # Safety check: Ensure the embeddings list is indexed correctly
                    embedding_val = embeddings[j] if j < len(embeddings) else None
                    insert_data.append((row[1], row[2], row[0], embedding_val))
                try:
                    conn.executemany("""
                        INSERT INTO embeddings (ENTITY_TYPE, ENTITY_ID, TEXT_CHUNK, EMBEDDING)
                        VALUES (?, ?, ?, ?)
                    """, insert_data)
                except Exception as e:
                    logger.error(f"{e}")
                    print(f"Batch #{i}")
                
            self.initialize_fts(conn)

            logger.info(f"✅ RAG table synced with {len(data)} vectors.")

    def initialize_fts(self, conn):
        """To enable Full-Text Search on embeddings table."""

        logger.info("Initializing FTS.")

        conn.execute("LOAD fts;")
        
        # conn.execute("PRAGMA drop_fts_index('embeddings')")            
        conn.execute("""
            PRAGMA create_fts_index(
                'embeddings', 
                'CHUNK_ID', 
                'TEXT_CHUNK', 
                overwrite=1
            )
        """)
        logger.info("FTS Index initialized successfully.")


if __name__=="__main__":

    db_client = ElectionDB()
    db_client.init_db()