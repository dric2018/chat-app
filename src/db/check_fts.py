from config import CFG

import duckdb


if __name__=="__main__":

    with duckdb.connect(str(CFG.DB_PATH)) as conn:
        # Check if the FTS schema exists
        schemas = conn.execute("SELECT schema_name FROM information_schema.schemata").fetchall()
        print(f"Schemas: {schemas}") # Look for 'fts_main_embeddings'
        
        # Check if the table has data that FTS should be seeing
        count = conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
        print(f"Total rows in table: {count}")

    with duckdb.connect(str(CFG.DB_PATH)) as conn:
        # Test 1: Lowercase search
        res_low = conn.execute("SELECT fts_main_embeddings.match_bm25(CHUNK_ID, 'election') as score FROM embeddings WHERE score IS NOT NULL").fetchall()
        
        # Test 2: Uppercase search
        res_up = conn.execute("SELECT fts_main_embeddings.match_bm25(CHUNK_ID, 'ELECTION') as score FROM embeddings WHERE score IS NOT NULL").fetchall()
        
        print(f"Lowercase matches: {len(res_low)}")
        print(f"Uppercase matches: {len(res_up)}")
