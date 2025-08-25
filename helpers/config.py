import os
import sys
import yaml
from dotenv import load_dotenv

#print("config.py")

# Load .env variables
load_dotenv()

def load_config():
    """Load configuration from YAML file."""
    with open("config.yaml", "r") as file:
        return yaml.safe_load(file)

def get_current_company():
    """Determine current company from various sources"""
    config = load_config()
    
    # Priority: command line > environment > config file default
    
    # Check command line arguments
    if '--company' in sys.argv:
        try:
            idx = sys.argv.index('--company')
            if idx + 1 < len(sys.argv):
                cmd_company = sys.argv[idx + 1].upper()
                if cmd_company in ['MOBILY', 'STC']:
                    print(f"Company set from command line: {cmd_company}")
                    return cmd_company
        except (ValueError, IndexError):
            pass
    
    # Check environment variable
    env_company = os.getenv('COMPANY_MODE', '').upper()
    if env_company in ['MOBILY', 'STC']:
        print(f"Company set from environment: {env_company}")
        return env_company
    
    # Use default from config file
    default_company = config.get("company", "MOBILY").upper()
    print(f"Company set from config default: {default_company}")
    return default_company

# Load configuration
config = load_config()
CURRENT_COMPANY = get_current_company()

# API Keys
VANNA_API_KEY = os.getenv("VANNA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model Configurations (shared between companies)
GPT_MODEL = config["llm"]["model"]
VANNA_MODEL_NAME = config["models"]["vanna"]

# Database Paths (shared between companies)
DB_PATH_SQLITE = config["database"]["sqlite_uri"]
DB_PATH_RAW = config["database"]["local_path"]

# Company Display Information (company-specific)
COMPANY_CONFIG = config["companies"][CURRENT_COMPANY]
COMPANY_NAME = COMPANY_CONFIG["name"]
COMPANY_DISPLAY_NAME = COMPANY_CONFIG["display_name"]
COMPANY_THREAD_ID = COMPANY_CONFIG["thread_id"]
COMPANY_GREETING = COMPANY_CONFIG["greeting"]
COMPANY_AUTHOR_NAME = COMPANY_CONFIG["author_name"]
COMPANY_STEP_NAME = COMPANY_CONFIG["step_name"]

print(f"=== Configuration loaded for: {COMPANY_DISPLAY_NAME} ===")
print(f"Database: {DB_PATH_RAW}")
print(f"Vanna Model: {VANNA_MODEL_NAME}")

#print("config.py loaded")