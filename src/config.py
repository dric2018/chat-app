from dotenv import load_dotenv
load_dotenv() # Loading vars from .env

import os
import os.path as osp

from pathlib import Path

class CFG:
    SRC_DIR                 = Path(__file__).resolve().parent
    PROJECT_ROOT            = SRC_DIR.parent    
    LOGS_DIR                = osp.join(PROJECT_ROOT, "logs")
    DATA_DIR                = osp.join(PROJECT_ROOT, "data")

    # Server settings
    UI_PORT                 = os.getenv("UI_PORT", 8501)
    PROMETHEUS_PORT         = os.getenv("PROMETHEUS_PORT", "9090")
    MLFLOW_PORT             = os.getenv("MLFLOW_PORT", "5000")
    NGINX_PORT              = os.getenv("NGINX_PORT", "8080")
    VLLM_PORT               = os.getenv("VLLM_PORT", "8000")
    GRAFANA_PORT            = os.getenv("GRAFANA_PORT", "3000")

    SERVER_IP               = str(os.getenv("SERVER_IP", "127.0.0.1")) # "vllm" is the name of the vllm docker container
    DOCKER_CON_IP           = "http://host.docker.internal"
    VLLM_API_KEY            = os.getenv("VLLM_API_KEY", "token-is-ignored")
    HF_TOKEN                = os.getenv('HF_TOKEN', '')
    VLLM_BASE_URL           = f"http://{SERVER_IP}:{VLLM_PORT}/v1"
    MLFLOW_TRACKING_URI     = f"http://{SERVER_IP}:{MLFLOW_PORT}"

    # DB Paths
    DB_DIR                  = osp.join(PROJECT_ROOT, "storage")
    DB_NAME                 = "elections.sqlite"
    DB_PATH                 = os.getenv("DB_PATH", f"/{DB_DIR}/{DB_NAME}")
    
    # SQL Guardrails
    ALLOWED_TABLES          = ["turnout", "results"]
    SQL_MAX_LIMIT           = 50
    
    # LLM (vLLM) Settings
    IS_STREAM               = True
    BASE_MODEL              = os.getenv("BASE_MODEL", "Qwen/Qwen3-1.7B")
    MODEL_PROVIDER          = "openai"
    RELEVANCE_THRESHOLD     = 0.8  # For intent classification
    GENERATION_TEMPERATURE  = 0.0 # setting to 0 for consistent SQL generations...can also be set to higher values (0.7) for variability
    MAX_TOKENS              = 1024
    CHUNK_SIZE              = 1024
    CHUNK_OVERLAP           = 100
