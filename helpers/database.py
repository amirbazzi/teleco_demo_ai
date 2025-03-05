import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helpers import config
import sqlite3
from langchain_community.utilities import SQLDatabase
from helpers import config


#print("database.py is running...")
def get_sqlite_connection():
    """Return a raw SQLite connection."""
    return sqlite3.connect(config.DB_PATH_SQLITE.replace("sqlite:///", ""))

def get_langchain_db(db_path: str = config.DB_PATH_SQLITE) -> SQLDatabase:
    """Return a LangChain SQLDatabase instance."""
    return SQLDatabase.from_uri(db_path)

#print("database.py is available in helpers:", "get_langchain_db" in globals())