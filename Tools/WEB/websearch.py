from typing import Annotated

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

tavily_tool = TavilySearchResults(max_results=5)



# deep research
# plotting 
# rca
# sql
# PDF
# - pdf (general purpose)
# - business analyst (solution recommender)

# human in the loop 
# - human feedback for retraining and prompt engineering
# - complex question than needs multiple steps --> human approval

# do a deep analysis for where stc stands infront of its competitors mobily and zain between 2023 and 2024 by segment,
#  client and product using the stc database and based on reports (you might use ur deep search tool)

# node 1 : planning node for the tools 
# if we have more than two tools, human approval is needed 

# optimize the time for response 
