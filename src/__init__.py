
import logging

import sys

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("LocalLLMStack")


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Local LLM Stack Orchestrator")
    parser.add_argument("--reset", action="store_true", help="Full wipe and rebuild of the stack")
    parser.add_argument("--refresh", action="store_true", help="Rebuild Streamlit UI only (keeps vLLM warm)")
    parser.add_argument("--recreate", action="store_true", help="Full wipe and rebuild of the stack + recreate")

    return parser.parse_args()


__all__ = ["logger", "get_args"]
