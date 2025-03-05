import sys
import os

# Move up one level from 'src' to the root directory where 'Tools' exists
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Now you can import from Tools.RCA
from Tools.RCA import rca_main_graph
from Tools.RCA.rca_main_graph import rca_subgraph_tool
from Tools.SQL.sql import database_tool
from Tools.CODER.coder import python_repl_tool

from Tools.WEB.websearch import tavily_tool

from typing import Annotated
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from helpers.config import GPT_MODEL




from typing import Literal
from typing_extensions import TypedDict

from langgraph.graph import MessagesState, END
from langgraph.types import Command


# Test import
#print(rca_main_graph)


llm = ChatOpenAI(temperature=0, model=GPT_MODEL, streaming=True)



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
    """
    )


class Router(TypedDict):
    """Worker to route to next. If no workers needed or extra arguments needed for a tool, route to FINISH."""

    next: Literal[*options]


class State(MessagesState):
    next: str


def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})



from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

#memory = MemorySaver()

research_agent = create_react_agent(
    llm, tools=[tavily_tool]
)

rca_agent = create_react_agent(
    llm, tools=[rca_subgraph_tool]
)

sql_agent = create_react_agent(
    llm, tools=[database_tool]
)

code_agent = create_react_agent(llm, tools=[python_repl_tool])


def research_node(state: State) -> Command[Literal["supervisor"]]:
    result = research_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="researcher")
            ]
        },
        goto="supervisor",
    )




def rca_node(state: State) -> Command[Literal["supervisor"]]:
    result = rca_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="rca")
            ]
        },
        goto="supervisor",
    )



def sql_node(state: State) -> Command[Literal["supervisor"]]:
    result = sql_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="sql")
            ]
        },
        goto="supervisor",
    )



def code_node(state: State) -> Command[Literal["supervisor"]]:
    result = code_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="coder")
            ]
        },
        goto="supervisor",
    )


    from langgraph.checkpoint.memory import MemorySaver


def stc_ai_graph(user_query:list, thread:str):

    builder = StateGraph(State)
    builder.add_edge(START, "supervisor")
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", research_node)
    builder.add_node("coder", code_node)
    builder.add_node("rca", rca_node)
    builder.add_node("sql", sql_node)



    memory_supervisor = MemorySaver()
    super_graph = builder.compile(checkpointer=memory_supervisor)

    config = {"configurable": {"thread_id": thread}}

    messages = super_graph.invoke({"messages": user_query},config)

    return messages

# # Specify an input
# messages = [HumanMessage(content="wha tare the sales by segmetn in april")]

# # Run
# messages = stc_ai_graph(messages,"0")
# for m in messages['messages']:
#     m.pretty_print()





#------------- ANOTHER FORMAT -----------------#



builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", research_node)
builder.add_node("coder", code_node)
builder.add_node("rca", rca_node)
builder.add_node("sql", sql_node)

from langgraph.checkpoint.memory import MemorySaver

memory_supervisor = MemorySaver()
super_graph = builder.compile(checkpointer=memory_supervisor)
# graph  = builder.compile()


# async def run_ai(input_message, node_to_stream):
#     async for event in super_graph.astream_events({"messages": [input_message]}, {"configurable": {"thread_id": "01"}}, version="v2"):
#         # Get chat model tokens from a particular node 
#         if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
#             #print(f"Node: {event['metadata'].get('langgraph_node','')}. Type: {event['event']}. Name: {event['name']}")
#             data = event["data"]
            
#             return print(data["chunk"].content, end="")


import asyncio

async def run_ai(input_message, node_to_stream):
    async for event in super_graph.astream_events({"messages": [input_message]}, {"configurable": {"thread_id": "01"}}, version="v2"):
        # Get chat model tokens from a particular node
        if event["event"] == "on_chat_model_stream" and event['metadata'].get('langgraph_node','') == node_to_stream:
            data = event["data"]
            print(data["chunk"].content, end="")

# Make sure to call run_ai within an event loop
async def main():
    input_message = "why did the salesdecline  by segment in 2024 , use  rca"
    node_to_stream = "agent"
    await run_ai(input_message, node_to_stream)

# Run the main function in the asyncio event loop
asyncio.run(main())

   