from dotenv import load_dotenv
load_dotenv() # Loading vars from .env

import os
import os.path as osp

from pathlib import Path
from pprint import pprint

from openai import OpenAI

def get_project_root() -> Path:
    """Finds the root by looking for a marker file."""
    current_path = Path(__file__).resolve()

    for parent in current_path.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    return current_path.parent

class CFG:
    PROJECT_ROOT            = get_project_root()
    LOGS_DIR                = osp.join(PROJECT_ROOT, "logs")
    DATA_DIR                = osp.join(PROJECT_ROOT, "data")
    PROCESSED_DATA_DIR      = osp.join(DATA_DIR, "processed")

    # Server settings
    UI_PORT                 = os.getenv("UI_PORT", 8501)
    PROMETHEUS_PORT         = os.getenv("PROMETHEUS_PORT", "9090")
    NGINX_PORT              = os.getenv("NGINX_PORT", "8080")
    VLLM_PORT               = os.getenv("VLLM_PORT", "8000")
    GRAFANA_PORT            = os.getenv("GRAFANA_PORT", "3000")

    SERVER_IP               = str(os.getenv("SERVER_IP", "127.0.0.1"))
    DOCKER_CON_IP           = "http://host.docker.internal"
    VLLM_API_KEY            = os.getenv("VLLM_API_KEY", "token-is-ignored")
    HF_TOKEN                = os.getenv('HF_TOKEN', '')
    VLLM_BASE_URL           = f"http://{SERVER_IP}:{VLLM_PORT}/v1"

    # DB Paths
    DB_DIR                  = osp.join(PROJECT_ROOT, "storage/duckdb")
    DB_NAME                 = "elections.duckdb"
    DB_PATH                 = osp.join(DB_DIR, DB_NAME)
    
    # SQL Guardrails
    ALLOWED_TABLES          = ["vw_winners", 
                               "vw_party", 
                               "vw_turnout", 
                               "vw_results", 
                               "vw_rag_descriptions"
                               ]
    # DB
    SQL_MAX_LIMIT = TOP_K    = 20
    
    # LLM (vLLM) Settings
    IS_STREAM               = True
    BASE_MODEL              = os.getenv("BASE_MODEL", "Qwen/Qwen3-1.7B")
    EMBEDDING_MODEL_NAME    = "google/embeddinggemma-300m" #"sentence-transformers/all-MiniLM-L6-v2" (384d)
    MODEL_PROVIDER          = "openai"
    RELEVANCE_THRESHOLD     = 0.8 # For intent classification
    GENERATION_TEMPERATURE  = 0.0 # setting to 0 for consistent generations
    MAX_TOKENS              = 1024
    CHUNK_SIZE              = 256
    CHUNK_OVERLAP           = 100

    client = OpenAI(
        base_url=f"http://localhost:{VLLM_PORT}/v1",
        api_key=VLLM_API_KEY,
    )

os.chdir(CFG.PROJECT_ROOT)

if __name__=="__main__":
    pprint(vars(CFG))