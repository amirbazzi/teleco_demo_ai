import sys
import os


import plotly
import json
import plotly.express as px
import plotly.io as pio

# Move up one level from 'src' to the root directory where 'Tools' exists
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now you can import from Tools.RCA
# from Tools.RCA import rca_main_graph
from Tools.RCA.rca_main_graph import rca_subgraph_tool
from Tools.SQL.sql import database_tool
from Tools.CODER.coder import python_repl_tool

from Tools.WEB.websearch import tavily_tool

from typing import Annotated
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from helpers.config import GPT_MODEL

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
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
from IPython.display import Image, display

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

# Function to convert Pydantic model to Plotly Figure
def create_plotly_figure(figure_model: FigureModel) -> go.Figure:
    """Convert the Pydantic model to a Plotly Figure object"""
    fig = go.Figure()
    
    for trace in figure_model.data:
        if trace.type == 'scatter':
            fig.add_trace(go.Scatter(
                x=trace.x,
                y=trace.y,
                mode=trace.mode,
                name=trace.name,
            ))
        elif trace.type == 'bar':
            fig.add_trace(go.Bar(
                x=trace.x,
                y=trace.y,
                name=trace.name
            ))
        elif trace.type == 'pie':
            fig.add_trace(go.Pie(
                labels=trace.x,
                values=trace.y,
                name=trace.name
            ))
    
    # Handle the default template here
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
    
    # Convert the cleaned figure model into a Plotly figure
    fig = create_plotly_figure(figure_model)


    figure_path = 'figure.json'


    pio.write_json(fig, figure_path)


    print("DEBUG GENERATE FIGURE PLOTTER ============ ")

    if fig:
        plot_status = "Plot has been created successfully"
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



members = ["researcher", "rca", "sql", "plotter"]

options = members + ["FINISH"]



system_prompt = (
    """
    "
    When the user says hi or anything greeeting related, reply with a greeting and ask them what they would like to know and the available services.
    

    YOU ARE AN AI ASSISTANT IN STC, stc, SAUDI TELECOM COMPANY. 
    You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When the user asks about data with dates, always sort the date by date 
    " so that when we plot a trendline, the x-axis is sorted properly  When finished,"
    " respond with FINISH. 

    if "plotter" was called, you should respond by "The plotting tool was activated and this is the result".

    NEVER EVER give the user a code snippet when the user asks for a plot.

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
        llm, tools=[database_tool]
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



    def research_node(state: State) -> Command[Literal["supervisor"]]:
        result = research_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="researcher")
                ]
            },
            goto= END,
        )


    def rca_node(state: State) -> Command[Literal["supervisor"]]:
        result = rca_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="rca")
                ]
            },
            goto= END,
        )



    def sql_node(state: State) -> Command[Literal["supervisor"]]:
        result = sql_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="sql")
                ]
            },
            goto= END,
        )
    


    plot_agent = create_react_agent(llm, tools=[plotting_tool])


    def plotter_node(state: State) -> Command[Literal["supervisor"]]:
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



    # def code_node(state: State) -> Command[Literal["supervisor"]]:
    #     result = code_agent.invoke(state)
    #     return Command(
    #         update={
    #             "messages": [
    #                 HumanMessage(content=result["messages"][-1].content, name="coder")
    #             ]
    #         },
    #         goto= END,
    #     )

    builder = StateGraph(State)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    # builder.add_node("coder", code_node)
    builder.add_node("plotter", plotter_node)

    builder.add_node("rca", rca_node)
    builder.add_node("sql", sql_node)


    memory_supervisor = MemorySaver()
    super_graph = builder.compile(checkpointer=memory_supervisor)
    return super_graph


@cl.on_chat_start
async def on_chat_start():
    welcome_msg = "### STC AI Assistant"
    msg = cl.Message(content=welcome_msg)

    super_graph = define_graph()
    cl.user_session.set("super_graph", super_graph)
    cl.user_session.set("msg", msg)
    await msg.send()


@cl.on_message
async def on_message(message: cl.Message):
    # Specify a thread
    config = {"configurable": {"thread_id": "42"}}

    msg = cl.Message(content="", author="STC Chatbot")
    if_plot = False
    super_graph = cl.user_session.get("super_graph")
    node_to_stream = 'agent'
    async with cl.Step("STC Chatbot") as step:
        step.input = message.content
        async for event in super_graph.astream_events({"messages": [message.content]}, config, version="v2", stream_mode = 'values'):
            # Get chat model tokens from a particular node 
            if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
                
                #print(f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}")
                data = event["data"]
                # print(data["chunk"].content, end="")
                token = data["chunk"].content
                print("=========== DEBUG", token)
                await msg.stream_token(token)
                
                path = 'figure.json'
                if os.path.exists(path):
                    fig = pio.read_json(path)
                    elements = [cl.Plotly(name="chart", figure=fig, display="inline")]
                    await cl.Message("", elements=elements, author="STC Chatbot").send()
                    os.remove(path)
                    print(f"{path} has been deleted.")
                    if_plot = True
    
        # await msg.send()  
        if if_plot:
            await msg.send()  
        else:
            await msg.send()  
            await step.remove()
    
    
    
    
    
    
    # async with cl.Step("STC Chatbot") as step:
    #     step.input = result
        
    #     async for chunk, metadata in graph_bot.astream(result, config, stream_mode='messages'):
            
    #         print("CHUNK DEBUG ====== ", chunk)
    #         if isinstance(chunk, AIMessage):
    #             token = chunk.content
    #             await msg.stream_token(token)
            
    #     await msg.send()    
    