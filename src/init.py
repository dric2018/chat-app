from config import CFG
import platform
import subprocess
import os
from utils import check_stack_health, init_logging, get_args
from src import logger
import sys

def setup_nginx_config():
    """Generates Nginx and Prometheus configs based on pyproject.toml."""
    
    logger.info("Calling setup_nginx_config()...")
    logger.info("Generating Nginx and Prometheus configurations...")
    
    nginx_conf = f"""
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=5r/s;

    server {{
        listen 80;
        
        location / {{
            
            auth_basic "Restricted Testing Area";
            auth_basic_user_file /etc/nginx/.htpasswd;

            proxy_pass http://streamlit-app:8501;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400;

            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;

            proxy_buffering off;
            proxy_cache off;

        }}

        location /_stcore/stream {{
            auth_basic "Restricted Testing Area";
            auth_basic_user_file /etc/nginx/.htpasswd;
                    
            proxy_pass http://streamlit-app:8501/_stcore/stream;
            proxy_http_version 1.1;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header Host $host;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400;
        }}

        
        # 'vllm' is the service name defined in docker-compse.yml
        # this will allow Docker's internal DNS to resolve it to the vLLM container's IP
        location /v1/ {{
            proxy_pass http://vllm:8000;
        }}
    }}"""
    
    with open("nginx.conf", "w") as f:
        f.write(nginx_conf)
    logger.info("Configuration files written successfully.")
    
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
            vllm_image = "openeuler/vllm-cpu:latest"
            logger.info("🍎 Apple Silicon detected. Switching to CPU-optimized stack.")
            logger.info(f"Using image: {vllm_image}")
            
            return {"image": vllm_image, "device": "cpu", "is_mac": True}
        else:      
            vllm_image = "vllm/vllm-cpu:latest"
            logger.info(f"Using image: {vllm_image}")
            return {"image": vllm_image, "device": "cpu", "is_mac": False}

def add_to_env_file(hw:dict):
    logger.info("Calling add_to_env_file()...")
    
    image_in_env = os.getenv("VLLM_IMAGE")
    logger.info(f"{image_in_env=}")

    if image_in_env is  None:
        with open(".env", "a") as f:
            f.write(f"VLLM_IMAGE={hw["image"]}\n")
            f.write(f"VLLM_DEVICE={hw["device"]}\n")

def run_stack():

    init_logging()
    args = get_args()

    logger.info("Calling run_stack()...")
    if not os.environ.get("VIRTUAL_ENV"):
        logger.warning("⚠️ You are not running in a virtual environment. Highly recommended!!!")

    try:
        # Hardware Detection
        hw = detect_hardware()
        
        # Set/update env vars
        os.environ["NGINX_PORT"]    = CFG.NGINX_PORT
        os.environ["VLLM_PORT"]     = CFG.VLLM_PORT
        os.environ["NGINX_PORT"]    = CFG.NGINX_PORT
        os.environ["GRAFANA_PORT"]  = CFG.GRAFANA_PORT
        os.environ["UI_PORT"]       = CFG.UI_PORT
        os.environ["VLLM_IMAGE"]    = hw["image"]
        os.environ["VLLM_DEVICE"]   = hw["device"]
        os.environ["BASE_MODEL"]    = CFG.BASE_MODEL
        os.environ["DB_PATH"]       = CFG.DB_PATH   
        os.environ["PROMETHEUS_PORT"]= CFG.PROMETHEUS_PORT
        os.environ["HF_TOKEN"]      = CFG.HF_TOKEN

        add_to_env_file(hw)
        setup_nginx_config()
    
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
        logger.info(f"🐳 Launching stack...Device: {os.environ['VLLM_DEVICE']} ({platform.processor()})")

        if args.reset:
            logger.warning("🧹 Hard Reset: Wiping containers and volumes...")            

            if args.recreate:
                logger.warning("Recreating containers and volumes...")            
                
                subprocess.run([
                    "docker-compose", "up", "-d", "vllm",
                    "--force-recreate",
                    "--remove-orphans"
                ], env=os.environ, check=True) 
            else:
                subprocess.run([
                    "docker-compose", "up", "-d", "vllm",
                    "--remove-orphans"
                ], env=os.environ, check=True) 

            
            logger.info("Successfully initiated vLLM container.")

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

        logger.info("📺 Building & launching Frontend...")
        subprocess.run(["docker-compose", "build", "streamlit-app"], check=True)
        subprocess.run(["docker-compose", "up", "-d", "streamlit-app"], check=True)
        subprocess.run(["docker-compose", "up", "-d", "grafana", "prometheus", "nginx"], check=True)

        up = check_stack_health()
        if up:
            logger.info("🐳 Backend is up!")
            logger.info(f"🚀 Stack is fully operational! Access at http://{CFG.SERVER_IP}:{os.environ['NGINX_PORT']}")

    except Exception as e:
        logger.error(f"Critical failure during stack initialization: {e}")
        logger.error(e.__traceback__.tb_frame)

if __name__ == "__main__":
    run_stack()
