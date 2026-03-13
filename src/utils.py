from __init__ import logger
from config import CFG

import requests

from unidecode import unidecode
import re
import string

from prometheus_client import Counter, REGISTRY

    
def parse_llm_response(raw_response:str):
    """
    Regex to find the <think> block and the remaining text
    """
    # Try to extract Thinking Tags first (for Qwen3/DeepSeek-R1)
    think_match = re.search(r'<think>(.*?)</think>(.*)', raw_response, re.DOTALL)
    
    if think_match:
        thinking = think_match.group(1).strip()
        answer = think_match.group(2).strip()
    else:
        # If no tags, the thinking is empty, the whole thing is the answer
        thinking = "No internal reasoning provided by model."
        answer = raw_response.strip()

    # Clean up Markdown SQL blocks (Common in standard models)
    # by removing ```sql ... ``` or just ``` ... ```
    clean_sql = re.sub(r'```(?:sql)?\s*(.*?)\s*```', r'\1', answer, flags=re.DOTALL).strip()
    
    return thinking, clean_sql

def check_stack_health():
    services = {
        "Prometheus": f"http://{CFG.SERVER_IP}:{CFG.PROMETHEUS_PORT}/-/healthy",
        "vLLM": f"http://{CFG.SERVER_IP}:{CFG.VLLM_PORT}/health",
        "Nginx": f"http://{CFG.SERVER_IP}:{CFG.NGINX_PORT}",
    }
    
    up = False

    for name, url in services.items():
        try:
            if name=='Nginx':
                creds = (CFG.USERNAME, CFG.VLLM_API_KEY)
                res = requests.get(url, auth=creds, timeout=2)
            else:
                res = requests.get(url, timeout=2)
            
            status = "✅" if res.status_code == 200 else "⚠️"
            logger.info(f"{status} {name} (url: {url})")

            up = True
        except:
            logger.error(f"❌ {name}: \tUnreachable via url {url}")

    return up

def normalize_text(
        text:str, 
        remove_punctuation:bool=False, 
        lowercase:bool=True
    ):

    """
        Performs entity normalization for accents/casing/punctuations
    """

    punctuation_to_remove = '!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'
    
    if lowercase:
        text = text.lower()

    if remove_punctuation:
        translator = str.maketrans(punctuation_to_remove, ' ' * len(punctuation_to_remove))
        text = text.translate(translator)
    
    text = text.strip()
    
    return unidecode(text)


def get_security_counter():
    metric_name = 'rag_sql_security_violations_total'
    # Check if this metric is already in the global registry
    if metric_name not in REGISTRY._names_to_collectors:
        return Counter(
            metric_name, 
            'Total number of blocked SQL security violations'
        )
    # If it exists, return the existing collector
    return REGISTRY._names_to_collectors[metric_name]

if __name__=="__main__":
    check_stack_health()