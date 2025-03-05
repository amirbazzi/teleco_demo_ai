import sys
import os

# Navigate up two levels to the project root and add it to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# print("args_mappings.py is running...")
import sys
import os
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



# ai_message ="""
# [HumanMessage(content='tell me whta cause the revenue incerase in 2024 by segment', additional_kwargs={}, response_metadata={}, id='8c819ce2-dd58-4c9a-8c41-314e1b096fe5'),
#  AIMessage(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_voZEBS2eeoJEK7VWo3vrgWYy', 'function': {'arguments': '{"question":"What caused the revenue increase in 2024 by segment?"}', 'name': 'rca_judge'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_50cad350e4'}, id='run-b5b13988-60e7-4f1a-9312-624a0ece6afd-0', tool_calls=[{'name': 'rca_judge', 'args': {'question': 'What caused the revenue increase in 2024 by segment?'}, 'id': 'call_voZEBS2eeoJEK7VWo3vrgWYy', 'type': 'tool_call'}]),
#  ToolMessage(content="rca_required='Yes' reason='The query seeks to identify the causes for an increase in revenue, which requires a detailed analysis by segment.' arguments=RCAArguments(current_period='2024', previous_period=None, kpi='Revenue', levels=['Segment'], calculation_method='sum', thresholds=None, service_filter=None) missing_arguments=['Previous Period'] prompt_user='Please provide the previous year for comparison.' is_missing='Yes'", name='rca_judge', id='c7104f4a-558f-41e2-9632-cd7135875878', tool_call_id='call_voZEBS2eeoJEK7VWo3vrgWYy'),
#  AIMessage(content='Yes, an RCA (Root Cause Analysis) is required to determine what caused the revenue increase in 2024 by segment. However, I need additional information to proceed:\n\n- **Previous Period**: Please provide the previous year for comparison.\n\nOnce you provide this information, I can assist you further.', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_50cad350e4'}, id='run-5b989d8a-fca2-459b-a532-5add973c45fb-0'),
#  HumanMessage(content='2023', additional_kwargs={}, response_metadata={}, id='3f85778b-069f-498e-8b31-e5771d60e086'),
#  AIMessage(content="Thank you for providing the previous period. Now, we can proceed with the Root Cause Analysis to determine what caused the revenue increase in 2024 by segment. \n\nIf you have any specific thresholds or service filters you'd like to apply, please let me know. Otherwise, I will proceed with the analysis using the information provided.", additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_50cad350e4'}, id='run-3910252f-1875-4366-9974-7aab5d1da2d7-0'),
#  HumanMessage(content='what was the missing arguemtn', additional_kwargs={}, response_metadata={}, id='5f8322ae-5ff8-40b3-bd99-b71c897e1a5e'),
#  AIMessage(content='The missing argument was the **Previous Period**, which you have now provided as 2023. This allows us to compare the revenue between 2023 and 2024 by segment.', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'model_name': 'gpt-4o-2024-08-06', 'system_fingerprint': 'fp_50cad350e4'}, id='run-7620db8f-2d0f-4726-a0ba-6e5b1d8639e5-0')]"""



# updated_args = args_processor.invoke({
#         "user_interests": ai_message
#     })



def args_processor(rca_processed_response: Any):
    args_processor = args_assessment_prompt_template | args_assessment
    args_updated = args_processor.invoke({"user_interests": rca_processed_response})
    return args_updated


# print("args_processor is available in args_mappings:", "args_processor" in globals())
