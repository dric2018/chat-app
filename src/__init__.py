import argparse
from config import CFG
import logging
import os.path as osp
import sys

# Configure global logging for the stack initialization
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
args = parser.parse_args()

logger = logging.getLogger("LocalLLMStack")
__all__ = ["logger", "args"]
