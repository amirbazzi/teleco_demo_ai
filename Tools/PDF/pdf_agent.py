import sys
import os

base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(base, "Tools"))

sys.path.append(base)


from Tools.PDF.query_agent_utils import *
from PDF.ingestion import initialize_services
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing import Dict, Any, Annotated, Literal



class RagInput(TypedDict):
    question: str
    answer: str 



def agentic_rag_tool(input: RagInput, vector_store: AstraDBVectorStore, generator_model: GenerativeModel):
    """LangGraph-compatible wrapper for the RAG pipeline."""
    question = input["question"]

    sub_questions = query_planner(question, generator_model)
    context = retrieve_context(sub_questions, vector_store, generator_model=generator_model)
    print(context)
    answer = generate_final_answer(question, context, generator_model)

    return {"answer": answer}


def build_graph(generator_model, vector_store):
    pdf_graph = StateGraph(RagInput)

    # Add the main tool node
    pdf_graph.add_node("agentic_rag", lambda input: agentic_rag_tool(input, vector_store, generator_model))

    # Define edges using START and END
    pdf_graph.add_edge(START, "agentic_rag")  # Entry point
    pdf_graph.add_edge("agentic_rag", END)    # Terminal output

    return pdf_graph.compile()



multimodal_model, generator_model, embedding_model, storage_client = initialize_services()
from langchain_openai import ChatOpenAI

# generator_model = ChatOpenAI(model="gpt-4o")

vector_store = AstraDBVectorStore(
    embedding=embedding_model,
    collection_name=ASTRA_DB_COLLECTION_NAME,
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
)

# Build LangGraph
pdf_graph = build_graph(generator_model, vector_store)

# # Sample input
# test_input = {"question": "What is the total revenue of Zain, and STC-KSA alone last year?"}

# # # Run the graph
# output = pdf_graph.invoke(test_input)

# # print("\n🧠 FINAL ANSWER FROM LANGGRAPH:")
# print(output["answer"])



# @tool
# def pdf_subgraph_tool(
#     user_query: Annotated[str, "user query to process for PDF analysis"]):
#     """
#     Use this to process if PDF analysis is needed 
#     """
#     pdf_graph_test = pdf_graph.invoke({"user_query":user_query})
#     return pdf_graph_test


# pdf_subgraph_tool("analyze the pdf and give me a one sentence summary")

if __name__ == "__main__":
    test_input = {"question": "What are the total revenues of stc KSA (not group), Mobily, and Zain in  2024 based on the reports?"}

    # # Run the graph
    output = pdf_graph.invoke(test_input)
    print(output["answer"])