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


from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import MessagesState, END
from langgraph.types import Command
import chainlit as cl

members = ["researcher", "coder", "rca", "sql"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    """
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH. If the user asks a general question, route it to "sql" 
    whenever someone tell you their name, greet them and ask them what they would like to know and the available services.
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
        response = llm.with_structured_output(Router).invoke(messages)
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



    def code_node(state: State) -> Command[Literal["supervisor"]]:
        result = code_agent.invoke(state)
        return Command(
            update={
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name="coder")
                ]
            },
            goto= END,
        )

    builder = StateGraph(State)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    builder.add_node("coder", code_node)
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
    