
import sys
import os
import plotly
import json
import pandas as pd

# Navigate up two levels to the project root and add it to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import plotly.io as pio
import logging
from typing import List, Tuple
from langchain_core.tools import tool
from helpers.database import get_sqlite_connection
from helpers.config import DB_PATH_RAW, VANNA_MODEL_NAME, VANNA_API_KEY
from vanna.remote import VannaDefault
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Vanna Model
vn = VannaDefault(model=VANNA_MODEL_NAME, api_key=VANNA_API_KEY)
# Sometimes you may want to add documentation about your business terminology or definitions.


# --------------------------
# 📌 Generate SQL Query Tool
# --------------------------

@tool("generate_sql_query", return_direct=True)
def generate_sql_query(user_query: str) -> str:
    """
    Generates an SQL query based on a user query using a language model.

    Args:
        user_query (str): The user's natural language query (e.g., "What is the top spending customer?").

    Returns:
        str: The generated SQL query string.
    """
    try:
        logging.info("--- GENERATING SQL QUERY ---")
        
        # Connect to SQLite Database
        vn.connect_to_sqlite(DB_PATH_RAW)

        # Generate SQL query
        sql_query = vn.generate_sql(question=user_query, allow_llm_to_see_data=True)
        logging.info(f"Generated SQL Query: {sql_query}")

        return sql_query
    
    except Exception as e:
        logging.error(f"Error generating SQL query: {e}")
        return f"Error: Failed to generate SQL query - {str(e)}"

# --------------------------
# 📌 Execute SQL Query Tool
# --------------------------



@tool("execute_sql_query", return_direct=True)
def execute_sql_query(sql_query: str, user_query: str) -> List[Tuple]:
    """
    Executes a given SQL query on an SQLite database and fetches the results.
    
    Args:
        sql_query (str): The SQL query string to execute.
        user_query (str): The user's query for generating plotly code.
    
    Returns:
        List[Tuple]: A list of tuples containing the rows returned by the SQL query.
    """
    try:
        logging.info("--- EXECUTING SQL QUERY ---")
        
        # Connect to SQLite Database
        vn.connect_to_sqlite(DB_PATH_RAW)

        # # Execute the SQL query
        # rows = vn.run_sql(sql_query)

        # # Check if the returned table has more than one row.
        # # If it only has one row, skip plotting.
        # if len(rows) > 1:
        #     plotly_code = vn.generate_plotly_code(
        #         question=user_query,
        #         sql=sql_query,
        #         df=rows
        #     )
        #     figure_path = 'figure.json'
        #     fig = vn.get_plotly_figure(
        #         plotly_code=plotly_code,
        #         df=rows
        #     )

        # Execute the SQL query and get results as a DataFrame
        rows_df = vn.run_sql(sql_query)

         # Get the list of numeric columns from the DataFrame
        numeric_cols = rows_df.select_dtypes(include=['number']).columns

        # Apply conditional rounding: if a value has more than 3 digits, round with 0 decimals; otherwise, round with 1 decimals.
        for col in numeric_cols:
            rows_df[col] = rows_df[col].apply(
                lambda x: round(x, 0) if pd.notnull(x) and abs(x) >= 1000 else round(x, 1)
            )

        # Convert DataFrame to list of tuples for the return value
        rows = [tuple(row) for row in rows_df.to_numpy()]

        # Check if the returned table has more than one row.
        if len(rows_df) > 1:
            plotly_code = vn.generate_plotly_code(
                question=user_query,
                sql=sql_query,
                df=rows_df  # Use the rounded DataFrame for plotting
            )
            figure_path = 'figure.json'
            fig = vn.get_plotly_figure(
                plotly_code=plotly_code,
                df=rows_df
            )
            
            print("DEBUG GENERATE FIGURE CODE ============ 1")
            print(plotly_code)
            print("DEBUG GENERATE FIGURE ============ 1")
            print(fig)
            
            pio.write_json(fig, figure_path)
            print("DEBUG GENERATE FIGURE ============ 2")
        else:
            logging.info("Only one row returned. Skipping plot generation.")
        
        logging.info(f"Executed query successfully. Retrieved {len(rows)} rows.")
        return rows

    except Exception as e:
        logging.error(f"Error executing query: {e}")
        raise

# --------------------------
# 📌 LLM Binding with Tools
# --------------------------

# Tools list
tools = [generate_sql_query, execute_sql_query]

# Bind tools with LLM
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


#--------------------------------------------------#
#-------------------- SQL MAIN --------------------#
#--------------------------------------------------#


sql_agent_prompt = """
-You are a helpful assistant tasked with taking in a user query, correcting any spelling or grammar mistakes. 

-You have a tool that generates an SQL query and another tool that executes this query.

-You will be working with an SQLite database, so make sure to create the proper SQL syntax to query the database and obtain the required information.

-Additionally, STC (or stc) refers to the company that owns this database. It is considered as a client, service, segment, or sector within the database. When the user asks about revenue, billing, or other data related to any sector, segment, service, or product, you should **not** filter specifically for STC, as it is the company you work for.

-Sales and revenue are often used interchangeably; however, whenever the user refers to sales or revenue, ensure to reference the Revenue column in the Metrics table

-The currency is SAR

- You MUST ALWAYS compute percentage changes and mathematical operations within the SQL query itself. Never return raw values that require external calculations.

- For percentage differences, generate SQL that directly calculates:
  ((NewValue - OldValue)/OldValue)*100 AS Result

- Example: When asked "percentage difference between 2023-2024":
WITH Revenue2023 AS ( SELECT SUM(Revenue) AS Revenue2023 FROM Metrics WHERE Date LIKE '%2023%' ), Revenue2024 AS ( SELECT SUM(Revenue) AS Revenue2024 FROM Metrics WHERE Date LIKE '%2024%' ) SELECT (Revenue2024.Revenue2024 - Revenue2023.Revenue2023) / Revenue2023.Revenue2023 * 100 AS PercentageChange FROM Revenue2023, Revenue2024;

- Return ONLY numerical results. Never show formulas, calculations, or LaTeX.

- If the query returns a single percentage value, present it directly: "The percentage difference is X%"

-When asked 'which customers declined in revenue in 2024' or anything similar, NEVER use this query:
    - SELECT Clients.ClientName
    FROM Clients
    JOIN Metrics AS m2023 ON Clients.ClientID = m2023.ClientID AND m2023.Date LIKE '2023%'
    JOIN Metrics AS m2024 ON Clients.ClientID = m2024.ClientID AND m2024.Date LIKE '2024%'
    WHERE m2024.Revenue < m2023.Revenue

    USE THIS INSTEAD:
    - WITH Revenue2023 AS (
            SELECT ClientID, SUM(Revenue) AS Revenue2023
            FROM Metrics
            WHERE Date LIKE '%2023%'
            GROUP BY ClientID
        ),
        Revenue2024 AS (
            SELECT ClientID, SUM(Revenue) AS Revenue2024
            FROM Metrics
            WHERE Date LIKE '%2024%'
            GROUP BY ClientID
        )
        SELECT 
            Clients.ClientName,
            Revenue2023.Revenue2023,
            Revenue2024.Revenue2024,
            (Revenue2024.Revenue2024 - Revenue2023.Revenue2023) AS RevenueDecline
        FROM Revenue2023
        JOIN Revenue2024 ON Revenue2023.ClientID = Revenue2024.ClientID
        JOIN Clients ON Revenue2023.ClientID = Clients.ClientID
        WHERE Revenue2024.Revenue2024 < Revenue2023.Revenue2023
        ORDER BY RevenueDecline ASC;

        
"""

sys_msg = SystemMessage(content=sql_agent_prompt)


# Node
def database_assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

from typing import Dict, Any, Annotated, Literal

from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from IPython.display import Image, display

# Graph
sql = StateGraph(MessagesState)

# Define nodes: these do the work
sql.add_node("sql_agent", database_assistant)
sql.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
sql.add_edge(START, "sql_agent")
sql.add_conditional_edges(
    "sql_agent",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
sql.add_edge("tools", "sql_agent")
sql_graph = sql.compile()

# Show
#display(Image(sql_graph.get_graph(xray=True).draw_mermaid_png()))


# NODE 2
@tool
def database_tool(user_query: Annotated[str, "Database tool extraction tool"]):
    """ 
    Use this tool to extract data from database to help answering a user query about anything related to data.
    
    """
    print("---SQL PROCESS STARTED---")
    database_result = sql_graph.invoke({"messages": user_query})
    return database_result


vn.train(
    question="whats the percentage change in revenue between 2023 and 2024?", 
    sql="""
    SELECT
  (
    (
      SUM(CASE WHEN Date LIKE '%2024%' THEN Revenue ELSE 0 END) -
      SUM(CASE WHEN Date LIKE '%2023%' THEN Revenue ELSE 0 END)
    )
    / NULLIF(SUM(CASE WHEN Date LIKE '%2023%' THEN Revenue ELSE 0 END), 0)
  ) * 100 AS PercentageChange
FROM Metrics
WHERE Date LIKE '%2023%' OR Date LIKE '%2024%';
"""
)

vn.train(
    question="percentage change in revenue between 2023 and 2024?", 
    sql="""
    SELECT
  (
    (
      SUM(CASE WHEN Date LIKE '%2024%' THEN Revenue ELSE 0 END) -
      SUM(CASE WHEN Date LIKE '%2023%' THEN Revenue ELSE 0 END)
    )
    / NULLIF(SUM(CASE WHEN Date LIKE '%2023%' THEN Revenue ELSE 0 END), 0)
  ) * 100 AS PercentageChange
FROM Metrics
WHERE Date LIKE '%2023%' OR Date LIKE '%2024%';
"""
)

vn.train(
    question="% change in revenue between 2023 and 2024?", 
    sql="""
    SELECT
  (
    (
      SUM(CASE WHEN Date LIKE '%2024%' THEN Revenue ELSE 0 END) -
      SUM(CASE WHEN Date LIKE '%2023%' THEN Revenue ELSE 0 END)
    )
    / NULLIF(SUM(CASE WHEN Date LIKE '%2023%' THEN Revenue ELSE 0 END), 0)
  ) * 100 AS PercentageChange
FROM Metrics
WHERE Date LIKE '%2023%' OR Date LIKE '%2024%';
"""
)


# Show
# display(Image(sql_graph.get_graph(xray=True).draw_mermaid_png()))

# messages = [HumanMessage(content="whats the percentage change in revenue between 2023 and 2024?")]

# messages = sql_graph.invoke({"messages": messages})
# for m in messages['messages']:
#     m.pretty_print()