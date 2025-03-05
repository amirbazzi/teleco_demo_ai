import sys
import os

# Navigate up two levels to the project root and add it to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# print("query_processor.py is running...")



from typing import Optional, List, Dict, Literal
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from helpers.config import GPT_MODEL  # Load LLM model config

# Initialize LLM
llm = ChatOpenAI(temperature=0, model=GPT_MODEL, streaming=True)

combined_prompt = """
Your task is to first determine whether the user query requires Root Cause Analysis (RCA). If RCA is required, extract the following arguments from the query. If any arguments are missing or unclear, ask the user to provide them.

---

### Step 1: Determine if RCA is Required
A query requires RCA if:
1. It focuses on diagnosing an issue, identifying causes, or solving a problem (e.g., "Why did sales decline in Q3?").
2. It asks for a detailed analysis of trends, inefficiencies, or failures.

Respond with:
- "Yes" if RCA is required, along with a reason.
- "No" if RCA is not required, along with a reason.

---

### Step 2: Extract Arguments for RCA (If Required)
If the query requires RCA, extract the following arguments:
1. **Current Period**: The year for the current period you are analyzing.
2. **Previous Period**: The year for the previous period you want to compare against.
3. **KPI**: The key performance indicator you want to analyze.
4. **Levels**: The grouping hierarchy for the analysis.
5. **Calculation Method** (optional): The aggregation method (default is "sum").
6. **Thresholds** (optional): Thresholds for filtering results. (default is 2).
7. **Service Filter** (optional): Any specific service you want to filter by.

---

### Step 3: Handle Missing Arguments
If any arguments are missing or ambiguous:
1. List the missing arguments.
2. Ask the user to provide them explicitly.

---

### Examples:

#### Example 1:
User Query: "Why are we falling short in revenue this year compared to last year?"
- **Step 1 (RCA Required)**: Yes  
  **Reason**: The query seeks to identify causes for a negative trend in revenue.
- **Step 2 (Extracted Arguments)**:
  - Current Period: Current Year (to be clarified by user)
  - Previous Period: Previous Year (to be clarified by user)
  - KPI: Revenue
  - Levels: None (to be clarified by user)
  - Calculation Method: Sum
  - Thresholds: 2
  - Service Filter: None

- **Step 3 (Missing Arguments)**:
  - Missing: Current Period, Previous Period, Levels.
  - is_missing: Yes
  - Ask: "Please provide the current year, previous year, and grouping levels for the analysis."

#### Example 2:
User Query: "What are the sales figures for Q3?"
- **Step 1 (RCA Required)**: No  
  **Reason**: The query asks for data retrieval without seeking underlying causes.

---

Now process the following query:
User Query: {query}
"""
    
class RCAArguments(BaseModel):
    current_period: Optional[str]
    previous_period: Optional[str]
    kpi: Optional[str]
    levels: Optional[List[str]]
    calculation_method: Optional[str]  # Removed "= 'sum'" default
    thresholds: Optional[Dict[str, str]] 
    service_filter: Optional[str] 


class RCACombinedResponse(BaseModel):
    rca_required: Literal["Yes", "No"]
    reason: str
    arguments: Optional[RCAArguments] = None  # Nested arguments with a clear structure
    missing_arguments: Optional[List[str]] = None  # List of missing arguments
    prompt_user: Optional[str] = None  # Prompt for missing information
    is_missing: Literal["Yes", "No"]



query_assessment = llm.with_structured_output(RCACombinedResponse)

query_assessment_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", combined_prompt),
        ("user", "User Query: {query}"),
    ]
)


def query_processor(user_query: str) -> str:
    print("---ANALYZING IF RCA NEEDED---")

    query_processor = query_assessment_prompt_template | query_assessment
    queries = query_processor.invoke(user_query)
    # print(queries)

    return queries


# print("query_processor is available in query_processor:", "query_processor" in globals())
