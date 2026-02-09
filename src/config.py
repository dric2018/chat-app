from dotenv import load_dotenv
load_dotenv() # Loading vars from .env

import os
import os.path as osp

class CFG:
    BASE_MOSEL  = "openai/gpt-oss-20b"
    LOGS_DIR    = osp.join(os.path.abspath("."), "logs")
    DATA_DIR    = osp.join(os.path.abspath("."), "data")

    # Secrets (from .env)
    VLLM_PORT = os.getenv("VLLM_PORT", "8000")

    VLLM_API_KEY = os.getenv("VLLM_API_KEY", "default_token")
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", f"http://vllm:{VLLM_PORT}/v1")
    
    MLFLOW_PORT = os.getenv("MLFLOW_PORT", "5000")
    MLFLOW_TRACKING_URI = f"http://mlflow:{MLFLOW_PORT}/v1"

    # DB Paths
    DB_DIR      = osp.join(os.path.abspath("."), "storage")
    DB_NAME     = "elections.sqlite"
    DB_PATH     = os.getenv("DB_PATH", f"/{DB_DIR}/{DB_NAME}")
    
    # SQL Guardrails
    ALLOWED_TABLES = ["turnout", "results"]
    SQL_MAX_LIMIT = 50
    
    # LLM Settings
    MODEL_NAME = "google/gemma-2b"
    RELEVANCE_THRESHOLD = 0.8  # For intent classification