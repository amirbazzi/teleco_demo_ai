import sys
import os

base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(base, "Tools"))

sys.path.append(base)


from Tools.PDF.pdf_analyzer_agent import *
#from PDF.ingestion import initialize_services
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing import Dict, Any, Annotated, Literal



class RagInput(TypedDict):
    question: str
    answer: str 



def agentic_rag_tool(input: RagInput, vector_store: AstraDBVectorStore):
    """LangGraph-compatible wrapper for the RAG pipeline."""
    question = input["question"]



    result_markdown = ask_agent(
        question= question,
        vector_store=vector_store,
        generator_model=llm,
        k=7,
        seen_parent_ids=set()
    )


    return {"answer": result_markdown}





def build_graph(vector_store):
    pdf_graph = StateGraph(RagInput)

    # Add the main tool node
    pdf_graph.add_node("agentic_rag", lambda input: agentic_rag_tool(input, vector_store))

    # Define edges using START and END
    pdf_graph.add_edge(START, "agentic_rag")  # Entry point
    pdf_graph.add_edge("agentic_rag", END)    # Terminal output

    return pdf_graph.compile()



# multimodal_model, generator_model, embedding_model, storage_client = initialize_services()
from langchain_openai import ChatOpenAI

# generator_model = ChatOpenAI(model="gpt-4o")
llm = ChatOpenAI(model="gpt-4o")

import getpass
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

ASTRA_DB_APPLICATION_TOKEN ='AstraCS:hBHWZLZxlRTehyQWSsbDZrOZ:9f15e3517330e6c4d43b5945f9bf79dbd5061fc4a18f202fde8cf204da687e52'
ASTRA_DB_API_ENDPOINT = 'https://66455754-aee6-4068-92ca-97ffedfb1c85-us-east1.apps.astra.datastax.com'
ASTRA_DB_NAMESPACE = 'keywords_parsing'

from astrapy import DataAPIClient

# Initialize the client
client = DataAPIClient(ASTRA_DB_APPLICATION_TOKEN)
db_astra = client.get_database_by_api_endpoint(
  "https://66455754-aee6-4068-92ca-97ffedfb1c85-us-east1.apps.astra.datastax.com"
)

print(f"Connected to Astra DB: {db_astra.list_collection_names()}")


from langchain_astradb import AstraDBVectorStore

vector_store = AstraDBVectorStore(
    embedding=embeddings,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    collection_name="astra_vector_langchain",
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)

# Build LangGraph
pdf_graph = build_graph(vector_store)



if __name__ == "__main__":
    test_input = {"question": "What are the total revenues of stc KSA (not group), Mobily, and Zain in  2024 based on the reports?"}

    # # Run the graph
    output = pdf_graph.invoke(test_input)
    print(output["answer"])