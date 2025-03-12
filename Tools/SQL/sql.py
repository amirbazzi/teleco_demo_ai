
import sys
import os
import plotly
import json
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

        # Execute the SQL query
        rows = vn.run_sql(sql_query)

        # Check if the returned table has more than one row.
        # If it only has one row, skip plotting.
        if len(rows) > 1:
            plotly_code = vn.generate_plotly_code(
                question=user_query,
                sql=sql_query,
                df=rows
            )
            figure_path = 'figure.json'
            fig = vn.get_plotly_figure(
                plotly_code=plotly_code,
                df=rows
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


sql_agent_prompt = """You are a helpful assistant tasked with taking in a user query, 
correct if you have spelling or grammar mistakes, 
you have a tool that generates an SQL query and another tool that executes this query.
whenever the user asks about data with dates, always sort the date by date so that when we plot a trendline, the x-axis is sorted properly. 
- whenever asked to find the total or overall of a metric, always return the sum of that metric over the speciified time period.
- whenever someones asks: what is the total revenue in 2024? DONT EVER ANSWER by stating each data point, instead SUM ALL in that time period

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
display(Image(sql_graph.get_graph(xray=True).draw_mermaid_png()))


# NODE 2
@tool
def database_tool(user_query: Annotated[str, "Database tool extraction tool"]):
    """ 
    Use this tool to extract data from database to help answering a user query about anything related to data
    """
    print("---SQL PROCESS STARTED---")
    database_result = sql_graph.invoke({"messages": user_query})
    return database_result


# Show
# display(Image(sql_graph.get_graph(xray=True).draw_mermaid_png()))

# messages = [HumanMessage(content="what are the overall sales of stc?")]

# messages = sql_graph.invoke({"messages": messages})
# for m in messages['messages']:
#     m.pretty_print()