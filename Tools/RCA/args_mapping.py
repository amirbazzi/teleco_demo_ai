import sys
import os

# Navigate up two levels to the project root and add it to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from pathlib import Path

# Calculate the root directory (stc_ai_insights)
current_file = Path(__file__).resolve()
root_dir = current_file.parent.parent.parent  # Go up 3 levels: RCA → Tools → stc_ai_insights
sys.path.append(str(root_dir))
# print(sys.path)

# # Debug: Confirm the path
# print(f"Root directory: {root_dir}")
# print(f"Helpers path exists: {os.path.exists(os.path.join(root_dir, 'helpers'))}")

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from helpers.utilities import get_combined_schema
from helpers.config import GPT_MODEL
# from helpers.database import get_langchain_db

import sys
import os

# Initialize LLM
llm = ChatOpenAI(temperature=0, model=GPT_MODEL, streaming=True)

class RCAArguments(BaseModel):
    current_period: Optional[str]
    previous_period: Optional[str]
    kpi: Optional[str]
    levels: Optional[List[str]]
    calculation_method: Optional[str]  # Removed "= 'sum'" default
    thresholds: Optional[Dict[str, str]] = None
    service_filter: Optional[str] = None


# Retrieve database schema dynamically
schema = get_combined_schema(llm)
# print(schema)

# Generate prompt with the retrieved schema
column_mapping_prompt = f"""
    You are given a database schema and a text reflecting user interests for performing Root Cause Analysis (RCA). Your task is to:

    1. Map the levels, KPI, and filters in the dictionary to the exact column names from the database schema.
    2. Update the dictionary's `arguments` to include these exact column names under the respective keys.
    3. Ensure the mappings are precise and can be directly used to generate an SQL query.
    4. If the user inserts none, not available, nan, or nothing, convert it to `None` in Python.
    5. The column names in the `levels` should match the column names in the `thresholds`.

    ### Database Schema:
    {schema}

    ### Examples:

    #### Example 1:
    **Input:**
    The RCA arguments are as follows:

    - **Current Period**: 2024
    - **Previous Period**: 2023
    - **KPI**: Sales
    - **Levels**: Segment
    - **Calculation Method**: Sum

    These arguments will be used to analyze the revenue increase by segment from 2023 to 2024.

    **Output:**
    {{{{
        "current_period": "2024",
        "previous_period": "2023",
        "kpi": "Sales",  # Column Name: "Sales"
        "levels": ["Segments"],  # Mapped Column Names
        "calculation_method": "sum",
        "thresholds": {{{{"Segments": "1"}}}},  # Mapped Columns
        "service_filter": None  
    }}}}
    """

# print(column_mapping_prompt)
# Set up the query processor
args_assessment = llm.with_structured_output(RCAArguments)

args_assessment_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", column_mapping_prompt),
        ("human", "Here are the user arguments that need processing: {user_interests}"),
    ]
)




def args_processor(rca_processed_response: Any):
    args_processor = args_assessment_prompt_template | args_assessment
    args_updated = args_processor.invoke({"user_interests": rca_processed_response})
    return args_updated


# print("args_processor is available in args_mappings:", "args_processor" in globals())
