import os
import yaml
from dotenv import load_dotenv

#print("config.py")

# Load .env variables
load_dotenv()

def load_config():
    """Load configuration from YAML file."""
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

config = load_config()

# API Keys
VANNA_API_KEY = os.getenv("VANNA_API_KEY")

# Model Configurations
GPT_MODEL = config["llm"]["model"]
VANNA_MODEL_NAME = config["models"]["vanna"]

# Database Paths
DB_PATH_SQLITE = config["database"]["sqlite_uri"]
DB_PATH_RAW = config["database"]["local_path"]

#print("config.py loaded")