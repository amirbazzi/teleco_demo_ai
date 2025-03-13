import sys
import os

# Navigate up two levels to the project root and add it to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from langchain_core.tools import tool


from Tools.RCA.query_process import query_processor
from Tools.RCA.args_mapping import args_processor
from Tools.RCA.rca_analysis import rca_answer_processor
from Tools.RCA.rca_base_function import filter_by_year, apply_filters, calculate_overall_kpi, calculate_kpi_changes, calculate_grouped_kpi_changes, filter_and_sort_by_threshold, calculate_ratio, perform_root_cause_analysis
from IPython.display import Image, display
from helpers.database import get_sqlite_connection




import logging
from helpers.database import get_sqlite_connection
from helpers.config import DB_PATH_RAW, VANNA_API_KEY, DB_PATH_SQLITE
from helpers.utilities import add_year_column_from_date
from vanna.remote import VannaDefault
from typing import Dict, Any, Annotated, Literal
import pandas as pd
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages

# Initialize Vanna AI instance
vn = VannaDefault(model="stc-demo",api_key=VANNA_API_KEY )  # Model name loaded dynamically if needed

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Graph Schema

class RCAState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        user_query: user basic question
        sql_query: db tool 
        rca_result: dataframe with RCA results
    """
    user_query: Annotated[list[AnyMessage], add_messages]
    rca_response: str
    rca_args: str
    formatted_user_query: str
    sql_query: str
    df_extracted: list
    rca_result: str

    
# Node 1
def process_user_query(state: RCAState):
    # Test the processor with the sample query
    print("---ANALYZING IF RCA NEEDED---")
    user_query = state["user_query"]
    queries = query_processor(user_query)
    return {"rca_response" : queries}


# Conditional Edge
def rca_judge(state: RCAState):
    """
    Judge that determines if we need an RCA or not, and if yes, it determines if we need more arguments to proceed.
    """
    #rca_decision = state['rca_response']

    if state['rca_response'].rca_required == 'Yes' and state['rca_response'].is_missing == 'Yes':
        return "END"
    elif state['rca_response'].rca_required == 'No':
        return "END"
    elif state['rca_response'].rca_required == 'Yes' and state['rca_response'].is_missing == 'No':
        print("---MAPPING RCA ARGS---")
        return "rca_args_map"


# Node 2
# Mapping the RCA arguments
def rca_args_map(state: RCAState):
    """
    Once we have all the RCA arguments needed, we should map them to the correct column names and return a structured output
    """
    updated_args = args_processor(state['rca_response'])
    return {"rca_args":updated_args}


# Node 3
def form_user_query(state: RCAState) -> str:
    """
    Forms a user query dynamically based on RCA arguments.

    Args:
        arguments (RCAArguments): The RCA arguments containing the required details.

    Returns:
        str: The dynamically formed user query.
    """
    
    args = state['rca_args']
    kpi = args.kpi or "KPI"
    levels = ", ".join(args.levels) if args.levels else "all levels"
    current_period = args.current_period or "current period"
    previous_period = args.previous_period or "previous period"
    user_query =str(f"Extract the {kpi} by {levels} in {current_period} and {previous_period} record by record without summing the {kpi}")

    return {'formatted_user_query': user_query}

import pandas as pd
from sqlalchemy import create_engine

# This assumes your SQLDatabase instance provides a SQLAlchemy engine.
engine = create_engine(DB_PATH_SQLITE)

# Load lookup tables from the database.
df_products = pd.read_sql("SELECT * FROM products", engine)
df_segments = pd.read_sql("SELECT * FROM segments", engine)
df_clients  = pd.read_sql("SELECT * FROM clients", engine)
df_sectors  = pd.read_sql("SELECT * FROM sectors", engine)

 

# Node 4
def generate_and_execute(state: RCAState):
    """
    generate a sql query once the arguments of the RCA are complete. take as an input the formatted message which start with "Extract, then execute it"
    """
    try:
        print("---GENERATE QUERY---")
        user_query = state['formatted_user_query']
        vn.connect_to_sqlite(DB_PATH_RAW)
        sql_query =  vn.generate_sql(question=user_query, allow_llm_to_see_data=True)
        
        print("---EXECUTE QUERY---")
        extracted_df = vn.run_sql(sql_query)
        extracted_df = add_year_column_from_date(extracted_df)



        print(f"{extracted_df.columns}")
        if 'SegmentID' in extracted_df.columns:
            # Assuming your segments table has columns "SegmentID" and "SegmentName"
            extracted_df = extracted_df.merge(
                df_segments[['SegmentID', 'SegmentName']],
                on='SegmentID',
                how='left'
            )
            extracted_df.drop('SegmentID', axis=1, inplace=True)


        if 'ProductID' in extracted_df.columns:
            # Assuming your products table has columns "ProductID" and "ProductName"
            extracted_df = extracted_df.merge(
                df_products[['ProductID', 'ProductName']],
                on='ProductID',
                how='left'
            )
            extracted_df.drop('ProductID', axis=1, inplace=True)
        
        if 'ClientID' in extracted_df.columns:
            # Assuming your clients table has columns "ClientID" and "ClientName"
            extracted_df = extracted_df.merge(
                df_clients[['ClientID', 'ClientName']],
                on='ClientID',
                how='left'
            )
            extracted_df.drop('ClientID', axis=1, inplace=True)

        if 'SectorID' in extracted_df.columns:
            extracted_df = extracted_df.merge(
                df_sectors[['SectorID', 'SectorName']],
                on='SectorID',
                how='left'
            )
            extracted_df.drop('SectorID', axis=1, inplace=True)

        print(extracted_df)
        return {"df_extracted": extracted_df.to_dict(orient="records")}

    except Exception as e:
        logging.error(f"Error in query execution: {e}")
        return {"error": f"Failed to execute query: {str(e)}"}
    

# Node 5
def perform_rca_analysis(state: RCAState):
    """
    Executes the RCA analysis by feeding `rca_args` and `extracted_data` to the RCA function
    and updates the state with the resulting DataFrame.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        GraphState: The updated state with RCA results stored in `rca_result`.
    """
    print("---PERFORMING ROOT CAUSE ANALYSIS---")
    
    # Retrieve arguments and data from the state
    rca_args = state["rca_args"]
    print(rca_args)

    # Define a mapping from ID column names to name column names
    id_to_name_mapping = {
        "SegmentID": "SegmentName",
        "ProductID": "ProductName",
        "ClientID": "ClientName",
        "SectorID": "SectorName"
    }
    
    # Update the levels: for each level in rca_args.levels,
    # if it is one of the ID columns, replace it with the corresponding name column
    rca_args.levels = [id_to_name_mapping.get(level, level) for level in rca_args.levels]
    print("DEBUG ============== AFTER MAPPING")
    print(rca_args)


    extracted_df = pd.DataFrame(state["df_extracted"])
    #extracted_df = state["df_extracted"]
    
    # Now check DataFrame emptiness
    if extracted_df is None or extracted_df.empty:  # <-- Valid for DataFrames
        raise ValueError("Extracted data is missing or empty.")
    


    # Extract arguments
    args = rca_args.copy()
    
    # Map thresholds dictionary to a list in the same order as the levels
    thresholds_list = [int(args.thresholds.get(level, 0)) for level in args.levels]
    
    # Build the function call
    result_df = perform_root_cause_analysis(
        df=extracted_df,
        current_period=int(args.current_period),  # Convert to integer if needed
        previous_period=int(args.previous_period),  # Convert to integer if needed
        kpi=args.kpi,
        levels=args.levels,
        calculation="sum",
        thresholds=thresholds_list,
        service_filter=None 
    )


    rca_answer_analysis = rca_answer_processor(result_df)


    return {"rca_result" : rca_answer_analysis}




#---------------------------------------------------#
#----------------- GRAPH BUILDING ------------------#
#---------------------------------------------------#

from langgraph.graph import END, StateGraph, START

from langgraph.graph import END, StateGraph, START

rcaflow = StateGraph(RCAState)

# Define the nodes
rcaflow.add_node("process_user_query", process_user_query)
rcaflow.add_node("rca_args_map", rca_args_map)
rcaflow.add_node("form_user_query", form_user_query)

rcaflow.add_node("generate_and_execute", generate_and_execute)

#rcaflow.add_node("generate_query", generate_query)
#rcaflow.add_node("execute_query", execute_query)
rcaflow.add_node("perform_rca_analysis", perform_rca_analysis)

# Build graph
rcaflow.add_edge(START,"process_user_query")

# Edges taken after the `action` node is called.
rcaflow.add_conditional_edges(
    "process_user_query",
    # Assess agent decision
    rca_judge,{
        "END":END,
        "rca_args_map": "rca_args_map"
    }
)

rcaflow.add_edge("rca_args_map","form_user_query")
rcaflow.add_edge("form_user_query", "generate_and_execute")
#rcaflow.add_edge("generate_query", "execute_query")
rcaflow.add_edge("generate_and_execute", "perform_rca_analysis")
#rcaflow.add_edge("perform_rca_analysis", END)

# Compile
rca = rcaflow.compile()


@tool
def rca_subgraph_tool(
    user_query: Annotated[str, "user query to process for RCA"]):
    """
    Use this to process if RCA is needed, gather arguments if needed then perform root cause analysis. 
    RCA helps identifying the reasons behind increase or increase in a certain KPI (sales, revenue, debt, etc..) over a period of time, 
    and the user can specify by what level (segment, service, etc..) he would like to carry on this analysis
    """
    rca_test = rca.invoke({"user_query":user_query})
    return rca_test

# rca_test = rca.invoke({"user_query":"tell me what cause the revenue incerase from 2023 to 2024 by segment and product "})


# print(rca_test)