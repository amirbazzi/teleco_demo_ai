import sys
import os

# Navigate up two levels to the project root and add it to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))



# print("rca_analysis.py is running...")

from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from helpers.config import GPT_MODEL  # Load LLM model config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# Initialize LLM
llm = ChatOpenAI(temperature=0, model=GPT_MODEL, streaming=True)


class RCA_answer(BaseModel):
    rca_interpratation: str

RCA_instructions_prompt = """
 #### When using the 'root_cause_analysis' tool, you must the following format and structure when interpreting and presenting the data:
    // 1- Start with a paragraph that mentions what was the overall change at the L0 level. This paragraph should mention what was the kpi at the previous year, what it has become this year, what was the absolute change and finally what is the percentage change.
    // 2- From level L1 forward, each deeper level should be represented as a sub-bullet point under the upper level. E.g. Start with L1, then under L1 there would be a sub-bullet point mentioning L2 causes while using the term 'within L1', 'within L2', and so on.
    // 3- L3 should be nested under L2, and L2 nested under L1, and so on. You must adhere to this structure and cover all rows in the data.
    // 3- At each level, similar to L0, you must mention all of the previous kpi, the current kpi, the absolute_change and the effect_of_change.
    // 4- Whenever the effect_of_change percentage is mentioned, you must use the term 'contributing to a <eoc>% to the overall change.
    // 5- When presenting the sections, avoid writing "L1", "L2". Instead, write the actual cause name at that level so that it looks nicer.
    // 6- Keep in mind that the root cause analysis result that you will be interpreting is ordered by top contributing reason to lowest at each of the levels from L1 to each subsequent level.
    // 7- You must cover every single row in the results, every cause in every level.
    // 8- If you think there are some general root causes that the data does not mention, mention it too in the end.
    // 9- If user asks about a specific service' revenue, like 'data' revenue or 'fixed' revenue, use the service_filter parameter to filter the data.
    To help you understand further how you should format and structure your root cause analysis result, here is a dummy example, comments are added within <> for your reference:
    <START OF EXAMPLE>
    user: what caused the revenue to increase in 2019?
    <root cause analysis invoked and result is returned>
    assistant: <L0 overall change mentioned> The overall revenue for saudi telecom company (stc) in 2019 was $12,000,000, compared to $10,000,000 in 2018, marking an absolute increase of $2,000,000. This represents a 20% increase from the previous year. Here's a breakdown of the root causes contributing to this change:
    <new paragraph stating the top L1 cause, let's suppose it's IT> IT: The IT sector saw a revenue increase from $2,000,000 in 2018 to $2,200,000 in 2019, contributing to a 2% increase in the overall change. Within IT:
    <sub bullet points for L2 causes under the top L1 cause which was IT in this example, let's supppose the top L2 cause under IT was cybersecurity> Cybersecurity: cybersecurity saw an increase from  $1,000,000 to 1,100,000, contributing to a 1% increase in the overall change. Within Cybersecurity:
    <sub bullet points for L3 causes under each L2 cause, suppose it was GOV in this example> GOV: GOV saw an increase from $300,000 to $400,000, contributing to a 0.03%  increase on the overall change.
    <when all the level causes under the top L1 cause are mentioned, we move forward to a new paragraph for the second top L1 cause and we repeat the above process>
    <END OF EXAMPLE>
"""

# Set up the query processor
answer_structure = llm.with_structured_output(RCA_answer)

answer_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", RCA_instructions_prompt),
        ("human", """Here is the dataframe that can be a dictionary or dataframe or json : \n\n {dataframe} . """)
          ]
)



def rca_answer_processor(df: Any):
    rca_answer_processor = answer_prompt_template | answer_structure
    rca_analysis = rca_answer_processor.invoke({"dataframe": df})
    return rca_analysis


# print("rca_answer_processor is available in rca_analysis:", "rca_answer_processor" in globals())
