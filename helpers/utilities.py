import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
# Use absolute import from the helpers package
from helpers import database as db
from helpers import config
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.language_models import BaseLanguageModel
from typing import Optional

import logging
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_core.language_models import BaseLanguageModel
from typing import Optional

# print("Helper functions loaded.")

def add_year_column_from_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a "Year" column to the DataFrame if it does not already exist
    and a "Date" column is present.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with a "Year" column (if applicable).
    """
    if isinstance(df, pd.DataFrame) and 'Date' in df.columns and 'Year' not in df.columns:
        try:
            df['Year'] = pd.to_datetime(df['Date'], errors='coerce').dt.year
        except Exception as e:
            print(f"Error converting 'Date' to datetime: {e}")
    return df


def get_combined_schema(llm: BaseLanguageModel, db_path: Optional[str] = None) -> str:
    """
    Retrieves and combines the schemas of all tables in a database.

    Args:
        llm (BaseLanguageModel): The language model instance for tool invocation.
        db_path (Optional[str]): The database path (default is loaded from config).

    Returns:
        str: A formatted string containing the combined schema of all tables.
    """
    # Load database path from config if not provided
    db_path = db_path or config.DB_PATH_SQLITE

    # Initialize the database and toolkit
    db = db.get_langchain_db(db_path)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = {tool.name: tool for tool in toolkit.get_tools()}  # Convert list to dictionary

    # Retrieve tools for listing tables and schemas
    list_tables_tool = tools.get("sql_db_list_tables")
    get_schema_tool = tools.get("sql_db_schema")

    if not list_tables_tool or not get_schema_tool:
        return "Error: Required database tools are missing."

    # Get the list of tables
    table_list_message = list_tables_tool.invoke("")
    table_names = [name.strip() for name in table_list_message.split(",") if name.strip()]

    # Retrieve schemas for all tables
    all_schemas = [
        f"Schema for {table_name}:\n{get_schema_tool.invoke(table_name)}"
        for table_name in table_names
    ]

    return "\n\n".join(all_schemas) if all_schemas else "No tables found in the database."



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_combined_schema(llm: BaseLanguageModel, db_path: Optional[str] = None) -> str:
    """
    Retrieves and combines the schemas of all tables in a database.

    Args:
        llm (BaseLanguageModel): The language model instance for tool invocation.
        db_path (Optional[str]): The file path to the database (default is loaded from config).

    Returns:
        str: A formatted schema of all tables in the database, or an error message.
    """
    # Use the default database path from config if not provided
    db_path = db_path or config.DB_PATH_SQLITE

    try:
        logging.info(f"Connecting to database: {db_path}")
        db = SQLDatabase.from_uri(db_path)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools_dict = {tool.name: tool for tool in toolkit.get_tools()}  # Store tools in a dictionary

        # Check if required tools exist
        if "sql_db_list_tables" not in tools_dict or "sql_db_schema" not in tools_dict:
            logging.error("Required database tools are missing.")
            return "Error: Required database tools (list tables, get schema) are missing."

        # Get the list of tables
        logging.info("Fetching list of tables...")
        table_list_message = tools_dict["sql_db_list_tables"].invoke("")
        table_names = [name.strip() for name in table_list_message.split(",") if name.strip()]

        if not table_names:
            logging.warning("No tables found in the database.")
            return "No tables found in the database."

        # Retrieve schemas for all tables
        logging.info(f"Fetching schema for {len(table_names)} tables...")
        all_schemas = [
            f"Schema for {table_name}:\n{tools_dict['sql_db_schema'].invoke(table_name)}"
            for table_name in table_names
        ]

        return "\n\n".join(all_schemas)

    except Exception as e:
        logging.error(f"Error retrieving database schema: {e}")
        return f"Error retrieving database schema: {e}"


# print("Helper functions loaded.")