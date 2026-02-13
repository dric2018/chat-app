import argparse
from config import CFG
import logging
import os.path as osp
from openai import OpenAI
import sys
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(osp.join(CFG.LOGS_DIR, "run.log"))
    ]
)

parser = argparse.ArgumentParser(description="Local LLM Stack Orchestrator")
parser.add_argument("--reset", action="store_true", help="Full wipe and rebuild of the stack")
parser.add_argument("--refresh", action="store_true", help="Rebuild Streamlit UI only (keeps vLLM warm)")
parser.add_argument("--recreate", action="store_true", help="Full wipe and rebuild of the stack + recreate")

args = parser.parse_args()

client = OpenAI(
    base_url=f"http://{CFG.SERVER_IP}:{CFG.VLLM_PORT}/v1",
    api_key=CFG.VLLM_API_KEY,
)

logger = logging.getLogger("LocalLLMStack")
__all__ = ["logger", "args", "client"]
