import sys
import os


import plotly
import json
import plotly.express as px
import plotly.io as pio

# from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

from Tools.RCA.rca_main_graph import rca_subgraph_tool
from Tools.SQL.sql import database_tool
from Tools.CODER.coder import python_repl_tool
#from Tools.PDF.pdf_agent import pdf_subgraph_tool

from Tools.WEB.websearch import tavily_tool

from typing import Annotated
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from helpers.config import GPT_MODEL

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate

from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import MessagesState, END
from langgraph.types import Command
import chainlit as cl


from pydantic import BaseModel, Field
from typing import List, Optional
import plotly.graph_objects as go
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

import plotly.express as px
import plotly.io as pio
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List, Optional
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage, SystemMessage



# Define structured output model using Pydantic v2 syntax
class Trace(BaseModel):
    type: str = Field(..., description="Plot type (bar, line, scatter, etc.)")
    x: List[str] = Field(..., description="X-axis data")
    y: List[float] = Field(..., description="Y-axis values")
    name: Optional[str] = None
    mode: Optional[str] = None

class Layout(BaseModel):    
    title: str = Field(..., description="Chart title")
    # Remove default value from the schema to avoid the error:
    template: Optional[str] = Field(None, description="Plot template")
    xaxis_title: Optional[str] = None
    yaxis_title: Optional[str] = None

class FigureModel(BaseModel):
    data: List[Trace]
    layout: Layout

# Create parser with proper schema handling
parser = PydanticOutputParser(pydantic_object=FigureModel)
format_instructions = parser.get_format_instructions()

viz_prompt = """
You are a data visualization expert. Generate a Plotly figure based on the user query.

Follow these rules:
1. Use appropriate chart types:
   - Bar charts for comparisons
   - Line charts for trends
   - Pie charts for proportions
2. Use placeholder data when needed
3. Add clear labels and titles

User Query: {user_query}

Output Format:
{format_instructions}
"""

# Create prompt template with proper variables
viz_processor_prompt = ChatPromptTemplate.from_messages([
    ("system", viz_prompt),
    ("human", "User Query: {user_query}")
]).partial(format_instructions=format_instructions)

# Combine the prompt and LLM
viz_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
structured_viz_processor = viz_llm.with_structured_output(FigureModel)
viz_processor = viz_processor_prompt | structured_viz_processor


# Helper function to apply conditional rounding
def round_value(value):
    if isinstance(value, (int, float)):
        # If a number has more than 3 digits (i.e. abs(value) >= 1000), round to 0 decimals;
        # otherwise, round to 1 decimal
        return round(value, 0) if abs(value) >= 1000 else round(value, 1)
    return value


# Function to convert Pydantic model to Plotly Figure
def create_plotly_figure(figure_model: FigureModel) -> go.Figure:
    """Convert the Pydantic model to a Plotly Figure object, applying conditional rounding to numeric values."""
    fig = go.Figure()
    
    for trace in figure_model.data:
        # Apply conditional rounding if the x and y properties are lists of numeric values.
        if isinstance(trace.x, list):
            trace_x = [round_value(val) for val in trace.x]
        else:
            trace_x = trace.x
        
        if isinstance(trace.y, list):
            trace_y = [round_value(val) for val in trace.y]
        else:
            trace_y = trace.y

        if trace.type == 'scatter':
            fig.add_trace(go.Scatter(
                x=trace_x,
                y=trace_y,
                mode=trace.mode,
                name=trace.name,
            ))
        elif trace.type == 'bar':
            fig.add_trace(go.Bar(
                x=trace_x,
                y=trace_y,
                name=trace.name
            ))
        elif trace.type == 'pie':
            # For pie charts, typically trace.x provides the labels.
            fig.add_trace(go.Pie(
                labels=trace_x,
                values=trace_y,
                name=trace.name
            ))
    
    # Handle the default template here.
    template = figure_model.layout.template if figure_model.layout.template is not None else "plotly_white"
    
    fig.update_layout(
        title=figure_model.layout.title,
        template=template,
        xaxis_title=figure_model.layout.xaxis_title,
        yaxis_title=figure_model.layout.yaxis_title
    )
    
    return fig

@tool
def query_to_figure(user_query: str):
    """
    Takes a user query describing a visualization and returns a Plotly figure.
    
    This function:
      1. Invokes the viz_processor with the user query to get a figure model.
      2. Removes any disallowed default value from the template property.
      3. Converts the cleaned figure model into a Plotly figure.
    
    Parameters:
        user_query (str): The query describing the desired plot.
    
    Returns:
        plotly.graph_objects.Figure: The generated Plotly figure.
    """
    # Generate the figure model from the user query using the viz_processor
    figure_model = viz_processor.invoke({"user_query": user_query})
    
    # Fix the schema issue: Remove the 'default' key from the 'template' property if it exists.
    if "template" in figure_model and isinstance(figure_model["template"], dict):
        figure_model["template"].pop("default", None)
    
    # Convert the cleaned figure model into a Plotly figure, with data rounded as specified.
    fig = create_plotly_figure(figure_model)
    
    # Save figure as JSON
    figure_path = 'figure.json'
    pio.write_json(fig, figure_path)

    print("DEBUG GENERATE FIGURE PLOTTER ============ ")

    if fig:
        plot_status = "Plot has been created successfully"
        return plot_status
    else:
        plot_status = "Plot creation failed"
        return plot_status





# Bind tools with LLM
llm = ChatOpenAI(model="gpt-4o")
llm_with_plotting_tool = llm.bind_tools([query_to_figure], parallel_tool_calls=False)


plotting_agent_prompt = """You are a helpful assistant tasked with plotting based on a user query, use plotly always"""

sys_msg = SystemMessage(content=plotting_agent_prompt)


# Node
def plotting_assistant(state: MessagesState):
   return {"messages": [llm_with_plotting_tool.invoke([sys_msg] + state["messages"])]}


# Graph
plotting = StateGraph(MessagesState)

# Define nodes: these do the work
plotting.add_node("plotting_agent", plotting_assistant)
plotting.add_node("tools", ToolNode([query_to_figure]))

# Define edges: these determine how the control flow moves
plotting.add_edge(START, "plotting_agent")
plotting.add_conditional_edges(
    "plotting_agent",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
plotting.add_edge("tools", "plotting_agent")
plotting_graph = plotting.compile()


# NODE 2
@tool
def plotting_tool(user_query: Annotated[str, "Plotting tool"]):
    """ 
    Use this tool to plot data 
    """
    plotting_result = plotting_graph.invoke({"messages": user_query})
    return plotting_result

#---------------------------------------------------------------------
#-------------------------------- PDF TOOL ---------------------------
#---------------------------------------------------------------------
from Tools.PDF.gpt_agent_api import gpt_pdf_graph
from langchain_core.tools import tool
from typing import Annotated

from Tools.PDF.gpt_agent_api import gpt_pdf_graph

# @tool
# def pdf_subgraph_tool(
#     user_query: Annotated[str, "user query to process for PDF analysis"]):
#     """
#     Use this to process if PDF analysis is needed 
#     """
#     print("STARTING PDF SUBGRAPH TOOL")
#     pdf_graph_test = pdf_graph.invoke({"question": user_query})
#     pdf_graph_test_answer = pdf_graph_test["answer"]
#     print("DEBUG PDF SUBGRAPH TOOL ============= ", pdf_graph_test_answer)
#     return pdf_graph_test_answer



# @tool("pdf_subgraph_tool", return_direct=True)
def pdf_subgraph_tool(
    user_query: Annotated[str, "User query to process for PDF analysis"]
) -> str:
    """
    Processes a user question through the PDF-analysis subgraph and returns the answer.
    """
    # Invoke the compiled PDF graph
    result = gpt_pdf_graph.invoke({"question": user_query})
    print(f"--- PDF TOOL ANSWER --- {result}")

    gpt_pdf_ans = result["answer"]

    print(f"--- PDF TOOL ANSWER --- {gpt_pdf_ans}")


    # Safely extract the "answer" field
    return gpt_pdf_ans

    #return result.get("answer", "")



members = ["researcher", "rca", "sql", "plotter", "pdf"]

options = members + ["FINISH"]

system_prompt = (
    """
    "
    WHEN ASKED ABOUT "whats the percentage change in revenue between 2023 and 2024" DONT SHOW FORMULA EVER.

    When the user says hi or anything greeeting related, reply with a greeting and ask them what they would like to know and the available services.

    Follow these guidelines at all times:
    - Do not confuse STC KSA with STC Group
    - when asked about revenue for stc in the report we mean stc KSA which is a subsidiary of STC Group.
    - you need to be asked explicitly about STC group to search for it in the reports.
    - Only when instructed to use the database, you should use the database tool 
    

    YOU ARE AN AI ASSISTANT IN Mobily telecom company
    You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When the user asks about data with dates, always sort the date by date 
    " so that when we plot a trendline, the x-axis is sorted properly  When finished,"
    " respond with FINISH. 

    The Currency of the revenue, billing, sales is SAR

    WHEN ASKED ABOUT "whats the percentage change in revenue between 2023 and 2024" DONT SHOW FORMULA EVER.
    if "plotter" was called, you should respond by "The plotting tool was activated and this is the result".

    NEVER EVER give the user a code snippet when the user asks for a plot.

    NEVER EVER mention the way you calculated percentage or the formula when the user asks for percentage change.

        

    When asked "What are the key services offered by STC across its various business segments?", use the database.
    
    If the user asks a general question, route it to "sql" 
    whenever someone tell you their name, greet them and ask them what they would like to know and the available services.
    whenever the user asks:
    "what is the revenue of stc based on th db" or any other question similar
    you should respond with a query to the database to get the overall revenue of stc (which is the company you are the assistant to, not the client)
    which is the sum of revenue. (stc is not a client)

    whenevr you use the researcher, go back to supervisor after the researcher is done and be ready to use the database. DONT GET STUCK IN THE RESEARCHER NODE
    whenever the user asks to plot, DONT USE MATPLOTLIB, use PLOTLY instead.

    for plots use plotly

    WHEN ASKED ABOUT "whats the percentage change in revenue between 2023 and 2024" DONT SHOW FORMULA EVER.

    The instructions for the pdf member are as follow:



    You are a specialized insights agent focused on analyzing publicly available annual analyst presentations and investor reports from Saudi telecom operators — STC, Zain, and Mobily.

    These documents are prepared for shareholders and financial analysts, and typically include:
    Subscriber base metrics (e.g. prepaid/postpaid, broadband, M2M)
    Segment-level financial performance (Consumer, Enterprise, Wholesale)
    ARPU trends, revenue growth, and customer base evolution
    Capex, network rollout (e.g. 5G, fiber), and digital initiatives
    Strategic priorities, transformation programs, and market commentary

    - There is a difference between STC KSA and STC Group. for example, when asked about revenue for STC KSA in 2024, you should use page 10 of the earning presentation 2024

    - When doing percentage change based on report get the numbers for 2023 and 2024 and perform the calculation don't use the percentage change from the reprts

    
    - Do not hallucinate or state that data is unavailable when it exists in the source documents.

    This applies especially for Mobily and Zain. Their revenue and segment-level data must be retrieved accurately from the provided documents.

    For Mobily, total revenue and segment breakdown (Consumer, Business, Wholesale, Others) for both 2023 and 2024 are available on page 10 of Earnings_Presentation_FY_2024.pdf, which is included in the vector store and properly tagged.

    Use this document directly when responding to related queries. Do not respond with "data is missing" when the information is present in the source.

    
    Your role is to answer questions using only the context provided below, which has been extracted from these presentations.

    Instructions:
    Base your response strictly on the CONTEXT section. Do not rely on prior knowledge or external sources.
    Clearly cite the source PDF name and page number for every data point in this format: (Source: [source_pdf], Page: [page]).
    If the context does not contain the necessary information, respond with:  
    “The requested information is not available in the provided analyst presentations. Would you like to switch to STC KSA’s internal database for a deeper root cause analysis?”

    Comparison logic:
    When comparing operators (e.g., STC vs Zain), always use analyst presentation data only. Do not reference internal data sources.
    Use STC internal data (RCA mode) **only if**:
    - The question pertains specifically to STC KSA
    - The PDF lacks sufficient information
    - The user explicitly agrees to switch to internal analysis

    Examples:
    1. User asks: “Why did STC’s enterprise revenue decline in 2024?”  
    → The analyst presentation shows that EBU revenue declined by 7.7%, mainly due to a drop in public sector revenue from SAR 9.87 billion to SAR 8.48 billion (−14.1%), partially offset by private sector revenue growth from SAR 4.72 billion to SAR 4.99 billion (+5.7%)  
    (Source: earnings-presentation2024en.pdf, Page: 10).  
    Provide this breakdown. Then say:  
    “If you’d like to explore more detailed internal performance metrics, I can switch to STC KSA’s internal database for a deeper root cause analysis.”

    2. User asks: “Compare postpaid subscriber growth across STC, Zain, and Mobily.”  
    → Use subscriber metrics directly from analyst presentations. Do not switch to internal data.

    3. User asks: “What was STC’s churn rate in the enterprise segment?”  
    → If this information is not in the context, respond with:  
    “The requested information is not available in the provided analyst presentations. Would you like to switch to STC KSA’s internal database for a deeper root cause analysis?”

    Never do:
    Never infer, guess, or supplement insights beyond what is explicitly stated in the provided context.
    Never pull or refer to internal STC KSA data unless the user confirms and the question requires internal depth.

    """
    )


class Router(TypedDict):
    """Worker to route to next. If no workers needed or extra arguments needed for a tool, route to FINISH."""

    next: Literal[*options]


class State(MessagesState):
    next: str

@cl.cache
def define_graph():
    llm = ChatOpenAI(temperature=0, model=GPT_MODEL, streaming=True)

    code_agent = create_react_agent(llm, tools=[python_repl_tool])


    research_agent = create_react_agent(
        llm, tools=[tavily_tool]
    )

    rca_agent = create_react_agent(
        llm, tools=[rca_subgraph_tool]
    )

    sql_agent = create_react_agent(
        llm, tools=[database_tool],
            prompt="if the user asked for the percentage, dont show the formula, just show the result"

    )


    pdf_agent = create_react_agent(
        llm, tools=[pdf_subgraph_tool],
        prompt="if the user asked any question related to mobily or zain, use the pdf tool to answer the question. If the user asked about stc, use the database tool to answer the question."
    )

    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]

        print(f"DEBUG SUPERVISOR NODE ============= {messages}")

        response = llm.with_structured_output(Router).invoke(messages)

        print(f"DEBUG SUPERVISOR NODE ============= {response}")

        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})



    def research_node(state: State):
        result = research_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="researcher")
                ]
            },
            goto= END,
        )


    def rca_node(state: State):
        result = rca_agent.invoke(state)


        print("DEBUG RCA NODE =================== ", result)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="rca")
                ]
            },
            goto= END,
        )



    def sql_node(state: State):
        result = sql_agent.invoke(state)
        print("DEBUG SQL NODE =================== ", result)

        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="sql")
                ]
            },
            goto= END,
        )
    

    def pdf_node(state: State):
        result = pdf_agent.invoke(state)
        print("DEBUG PDF NODE =================== ", result)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="pdf")
                ]
            },
            goto= END,
        )
    


    plot_agent = create_react_agent(llm, tools=[plotting_tool])


    def plotter_node(state: State):
        result = plot_agent.invoke(state)

        print(f'DEBUG PLOTTER NODE ============================ {result}')
        
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="plotter")
                ]
            },
            goto=END,
        )

    builder = StateGraph(State)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    builder.add_node("plotter", plotter_node)
    builder.add_node("pdf", pdf_node)
    builder.add_node("rca", rca_node)
    builder.add_node("sql", sql_node)


    memory_supervisor = MemorySaver()
    super_graph = builder.compile(checkpointer=memory_supervisor)
    return super_graph


# @cl.on_chat_start
# async def on_chat_start():
#     welcome_msg = "### STC AI Assistant"
#     msg = cl.Message(content=welcome_msg)

#     super_graph = define_graph()
#     cl.user_session.set("super_graph", super_graph)
#     cl.user_session.set("msg", msg)
#     await msg.send()

@cl.on_chat_start
async def on_chat_start():
    welcome_msg = "### Mobily AI Assistant"
    msg = cl.Message(content=welcome_msg)

    super_graph = define_graph()
    cl.user_session.set("super_graph", super_graph)
    cl.user_session.set("msg", msg)
    await msg.send()


# @cl.on_message
# async def on_message(message: cl.Message):
#     # Specify a thread
#     config = {"configurable": {"thread_id": "45"}}

#     msg = cl.Message(content="", author="STC Chatbot")
#     if_plot = False
#     super_graph = cl.user_session.get("super_graph")
#     node_to_stream = 'agent'
#     async with cl.Step("STC Chatbot") as step:
#         step.input = message.content
#         async for event in super_graph.astream_events({"messages": [message.content]}, config, version="v2", stream_mode = 'values'):
#             # Get chat model tokens from a particular node 
#             if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
                
#                 #print(f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}")
#                 data = event["data"]
#                 # print(data["chunk"].content, end="")
#                 token = data["chunk"].content
#                 print("=========== DEBUG", token)
#                 await msg.stream_token(token)
                
#                 path = 'figure.json'
#                 if os.path.exists(path):
#                     fig = pio.read_json(path)
#                     elements = [cl.Plotly(name="chart", figure=fig, display="inline")]
#                     await cl.Message("", elements=elements, author="STC Chatbot").send()
#                     os.remove(path)
#                     print(f"{path} has been deleted.")
#                     if_plot = True
    
#         # await msg.send()  
#         if if_plot:
#             await msg.send()  
#         else:
#             await msg.send()  
#             await step.remove()



@cl.on_message
async def on_message(message: cl.Message):
    # Specify a thread
    config = {"configurable": {"thread_id": "40"}}

    msg = cl.Message(content="", author="Mobily Chatbot")
    if_plot = False
    super_graph = cl.user_session.get("super_graph")
    node_to_stream = 'agent'
    async with cl.Step("Mobily Chatbot") as step:
        step.input = message.content
        async for event in super_graph.astream_events({"messages": [message.content]}, config, version="v2", stream_mode = 'values'):
            # Get chat model tokens from a particular node 
            if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
                
                print(f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}")
                data = event["data"]
                # print(data["chunk"].content, end="")
                token = data["chunk"].content
                print("=========== DEBUG", token)
                await msg.stream_token(token)
                
                path = 'figure.json'
                if os.path.exists(path):
                    fig = pio.read_json(path)
                    elements = [cl.Plotly(name="chart", figure=fig, display="inline")]
                    await cl.Message("", elements=elements, author="Mobily Chatbot").send()
                    os.remove(path)
                    print(f"{path} has been deleted.")
                    if_plot = True
    
        # await msg.send()  
        if if_plot:
            await msg.send()  
        else:
            await msg.send()  
            await step.remove()


