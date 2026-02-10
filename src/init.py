from config import CFG
import platform
import tomllib
import subprocess
import os
from utils import check_stack_health
import sys
from __init__ import logger, args

def setup_configs():
    """Generates Nginx and Prometheus configs based on pyproject.toml."""
    
    logger.info("Calling setup_configs()...")
    logger.info("Generating Nginx and Prometheus configurations...")
    
    nginx_conf = f"""
    # 1. Define the shared memory zone for rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=5r/s;

    server {{
        listen 80;
        
        location / {{
            proxy_pass http://localhost:8501;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400;
        }}

        location /v1/ {{
            proxy_pass http://localhost:8000/v1/;
        }}
    
        location /api/query {{
            # Apply the rate limit (5 requests per second)
            limit_req zone=api_limit burst=10 nodelay;

            proxy_pass http://agent-api:8000;
            
            # 2. Advanced: Block based on response content 
            # (Usually handled via a fail2ban script watching Nginx logs)
            # If the backend returns a 403 or a specific 'INVALID' header:
            proxy_intercept_errors on;
            error_page 403 = /block_page;
        }};
    }}"""
    
    with open("nginx.conf", "w") as f:
        f.write(nginx_conf)
    logger.info("Configuration files written successfully.")
    
def setup_env_file(config, hw:dict):
    logger.info("Calling setup_env_file()...")
    
    logger.info("⚙️ Writing .env file for Docker Compose...")
    with open(".env", "w") as f:
        f.write("# --- LLM Backend (vLLM) ---\n")
        f.write(f"VLLM_MODEL={config['model']}\n")
        f.write(f"VLLM_PORT={config['port']}\n")
        f.write(f"VLLM_IMAGE={hw["image"]}\n")
        f.write(f"VLLM_DEVICE={hw["device"]}\n")
        f.write(f"HF_TOKEN={os.getenv('HF_TOKEN', '')}\n")

def detect_hardware():
    logger.info("Calling detect_hardware()...")
    
    sys_info    = platform.system()
    arch_info   = platform.machine()
    
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        logger.info("NVIDIA GPU detected. Using CUDA stack.")
        return {"image": "vllm/vllm-openai:latest", "device": "cuda", "is_mac": False}
    except Exception:
        logger.warning("🐢 No GPU found. Falling back to standard CPU.")
  
        if sys_info == "Darwin" and arch_info == "arm64":
            logger.info("🍎 Apple Silicon detected. Switching to CPU-optimized stack.")
            return {
                "image": "openeuler/vllm-cpu:latest", # Most stable ARM-compatible CPU image
                "device": "cpu",
                "is_mac": True
            }
        else:        
            return {"image": "vllm/vllm-cpu:latest", "device": "cpu", "is_mac": False}

def run_stack():
    logger.info("Calling run_stack()...")
    if not os.environ.get("VIRTUAL_ENV"):
        logger.warning("⚠️ You are not running in a virtual environment. Highly recommended!!!")

    try:
        with open("pyproject.toml", "rb") as f:
            config = tomllib.load(f)["tool"]
        
        # Hardware Detection
        hw = detect_hardware()
        nginx_external_port = config.get("nginx_port", "8080")
        os.environ["NGINX_PORT"]    = str(nginx_external_port)
        os.environ["VLLM_IMAGE"]    = hw["image"]
        os.environ["VLLM_DEVICE"]   = hw["device"]
        os.environ["VLLM_MODEL"]    = config["vllm-stack"]["model"]
        os.environ["VLLM_PORT"]     = str(config["vllm-stack"]["port"])
        os.environ["MLFLOW_PORT"]   = str(config["monitoring"]["mlflow_port"])        
        os.environ["DB_PATH"]       = CFG.DB_PATH   

        setup_env_file(config["vllm-stack"], hw)
        setup_configs()
    
        if hw["is_mac"]:
            os.environ["COMPOSE_PROFILES"] = "cpu"
            logger.info("🔩 Hardware reservations neutralized for CPU-only mode.")
        else:
            os.environ["COMPOSE_PROFILES"] = "gpu"
            logger.info("NVIDIA Profile: GPU hardware reservations active.")

        if not os.path.exists(CFG.DB_DIR):
            os.makedirs(CFG.DB_DIR, exist_ok=True)
            logger.info(f"📁 Created local directory: {CFG.DB_DIR}")
        else:
            logger.info(f"📁 DB Directory found at {CFG.DB_DIR}")


        # Standard Launch (Smart & Cached)
        logger.info(f"🐳 Launching stack...Device: {os.environ['VLLM_DEVICE']} ({platform.processor()})\n")
        
        if args.reset:
            logger.warning("🧹 Hard Reset: Wiping containers and volumes...")
            subprocess.run(["docker-compose", "up", "-d", "--remove-orphans"], check=True)
        
        elif args.refresh:
            logger.info("🔍 Checking backend health before refresh...")
            if check_stack_health():
                logger.info("✅ Backend healthy and warm. Proceeding with UI refresh.")
            else:
                logger.warning("⚠️ Part of backend not responding! A simple refresh might fail.")
                confirm = input("Force UI refresh anyway? (y/n): ")
                if confirm.lower() != 'y':
                    sys.exit(1)

            logger.info("♻️ Refreshing Frontend only...")
            subprocess.run(["docker-compose", "up", "-d", "streamlit-app", 
                "--no-recreate"
                ], check=True)
            
            return
    
        # subprocess.run([
        #     "docker-compose", "up", "-d", "vllm", #"mlflow",
        #     "--no-recreate" # Prevents vLLM restart if nothing changed
        # ], env=os.environ, check=True)  

        # DATA INGESTION
        # Use a flag to skip if data already exists, or always run if 'args.reset'
        # logger.info(f"💾 Preparing data ingestion...\n")
        
        # if args.reset or not os.path.exists(os.environ.get("DB_PATH", "elections.db")):
        #     logger.info("📥 Starting Data Ingestion (PDF to DB)...")
        #     subprocess.run([
        #         "docker-compose", "run", "--rm", "ingestion-job"
        #     ], env=os.environ, check=True)
        #     logger.info("✅ Ingestion complete.")
        # else:
        #     logger.info("⏭️ skipping ingestion (Database already exists).")

        logger.info("🐳 Backend is up!")

        # logger.info("📺 Building & launching Streamlit frontend...")
        # subprocess.run(["docker-compose", "build", "streamlit-app"], check=True)
        # subprocess.run(["docker-compose", "up", "-d", "streamlit-app"], check=True)
        # subprocess.run(["docker-compose", "up", "-d", "nginx"], check=True)

        # logger.info(f"🌐 Nginx will be available at http://localhost:{nginx_external_port}")       
        logger.info(f"🚀 Stack is fully operational! Access at http://0.0.0.0:{os.environ['NGINX_PORT']}")

    except Exception as e:
        logger.error(f"Critical failure during stack initialization: {e}")

if __name__ == "__main__":
    run_stack()
