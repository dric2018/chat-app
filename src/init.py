from config import CFG
import docker
import platform
import subprocess
import os
from utils import check_stack_health
import requests
import sys
import time
from __init__ import logger, args

def setup_nginx_config():
    """Generates Nginx and Prometheus configs based on pyproject.toml."""
    
    logger.info("Calling setup_nginx_config()...")
    logger.info("Generating Nginx and Prometheus configurations...")
    
    nginx_conf = f"""
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=5r/s;

    server {{
        listen 80;
        
        location / {{
            
        if ($http_authorization != "Bearer ")
        {{
            return 401;
        }}

            proxy_pass http://streamlit-app:8501;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400;

            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Crucial for LLM streaming responses
            proxy_buffering off;
            proxy_cache off;

        }}

        location /_stcore/stream {{
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

def wait_for_vllm(
        vllm_container:docker.models.container.Container=None,
        url=f"http://{CFG.SERVER_IP}:{CFG.VLLM_PORT}/health", 
        timeout=300,

    ):
    """Wait for vLLM to return 200 OK at the health endpoint."""
    logger.info("Waiting for vLLM to finish initializing...")

    if vllm_container:
        while True:
            vllm_container.reload()
            health = vllm_container.attrs['State'].get('Health', {}).get('Status')
            if health == 'healthy':
                logger.info("vLLM is ready!")
                break
            if health == 'unhealthy':
                logger.info("vLLM healthcheck failed!")
                logger.info(vllm_container.logs().decode())
                break
            time.sleep(5)
    else:
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    logger.info("vLLM is up and healthy!")
                    return True
            except requests.exceptions.ConnectionError:
                pass
            
            logger.info("Waiting for vLLM to initialize weights...")
            time.sleep(5)
        
        raise TimeoutError("vLLM failed to start within the timeout period.")


def run_stack():
    logger.info("Calling run_stack()...")
    if not os.environ.get("VIRTUAL_ENV"):
        logger.warning("⚠️ You are not running in a virtual environment. Highly recommended!!!")

    try:
        # Hardware Detection
        hw = detect_hardware()
        
        # Set/update env vars
        os.environ["NGINX_PORT"]    = CFG.NGINX_PORT
        os.environ["VLLM_PORT"]     = CFG.VLLM_PORT
        os.environ["MLFLOW_PORT"]   = CFG.MLFLOW_PORT
        os.environ["NGINX_PORT"]    = CFG.NGINX_PORT
        os.environ["GRAFANA_PORT"]  = CFG.GRAFANA_PORT
        os.environ["UI_PORT"]       = CFG.UI_PORT
        os.environ["VLLM_IMAGE"]    = hw["image"]
        os.environ["VLLM_DEVICE"]   = hw["device"]
        os.environ["BASE_MODEL"]    = CFG.BASE_MODEL
        os.environ["DB_PATH"]       = CFG.DB_PATH   
        os.environ["PROMETHEUS_PORT"]= CFG.PROMETHEUS_PORT
        os.environ["HF_TOKEN"]      = CFG.HF_TOKEN
        os.environ["MLFLOW_TRACKING_URI"] = CFG.MLFLOW_TRACKING_URI

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
        docker_client = docker.from_env()

        if args.reset:
            logger.warning("🧹 Hard Reset: Wiping containers and volumes...")            
            
            # device_requests = []
            # if hw["device"]=="gpu":
            #     device_requests = [
            #         docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
            #     ]
            
            # docker_client.containers.prune()
            # logger.info("Building vLLM container...")
            # vllm_container = docker_client.containers.run(
            #     image=hw["image"],
            #     name="vllm",
            #     detach=True,
            #     ports={'8000/tcp': CFG.VLLM_PORT},
            #     volumes={
            #         os.path.expanduser('~/.cache/huggingface'): {'bind': '/root/.cache/huggingface', 'mode': 'rw'},
            #         'models_volume_name': {'bind': '/root/.cache/huggingface', 'mode': 'rw'}
            #     },
            #      command=[
            #         "--model", os.getenv("BASE_MODEL"),
            #         "--host", "0.0.0.0",
            #         "--dtype", "auto",
            #         "--max-model-len", "1024",
            #         "--trust-remote-code"
            #     ],
            #     environment={
            #         "HF_TOKEN": os.getenv("HF_TOKEN")
            #     },
            #     shm_size="4gb",
            #     # Hardware reservations (see x-gpu-config part in docker-compose.yml)
            #     device_requests=device_requests,
            #     network="backend-net", 
            #     restart_policy={"Name": "unless-stopped"},
            #     healthcheck={
            #         "test": ["CMD", "curl", "-f", f"http://{CFG.SERVER_IP}:{CFG.VLLM_PORT}/health"],
            #         "interval": 10 * 10**9,
            #         "timeout": 30 * 10**9,
            #         "retries": 20,
            #         "start_period": 60 * 10**9
            #     }
            # )

            # logger.info(vllm_container)
            # if vllm_container is not None:
            #     wait_for_vllm(vllm_container=vllm_container)

            if args.recreate:
                logger.warning("Recreating containers and volumes...")            
                
                subprocess.run([
                    "docker-compose", "up", "-d", "mlflow", "vllm",
                    "--force-recreate",
                    "--remove-orphans"
                ], env=os.environ, check=True) 
            else:
                subprocess.run([
                    "docker-compose", "up", "-d", "mlflow", "vllm",
                    "--remove-orphans"
                ], env=os.environ, check=True) 

            
            logger.info("Successfully initiated fondations (MLflow and vLLM).")

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
