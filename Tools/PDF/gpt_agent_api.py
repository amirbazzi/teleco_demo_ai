# MAIN INGESTION

import os
import base64
import json
from typing import List, Optional, Literal
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import io
from dotenv import load_dotenv
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Tuple, Any
import pathlib

def setup_financial_assistant(
    file_paths: List[str],
    assistant_name: str = "Financial Analyst Assistant",
    instructions: str = (
        "You are an expert financial analyst. Use your knowledge base to answer "
        "questions about audited financial statements."
    ),
    model: str = "gpt-4o",
    vector_store_name: str = "Financial Statements"
) -> Tuple[OpenAI, Any, Any]:
    """
    1) Loads environment variables
    2) Instantiates OpenAI client
    3) Creates (or re-uses) a vector store called `vector_store_name`
    4) Creates an assistant (if none exists) and points it at the vector store
    5) Uploads the given PDF files into the vector store (skips duplicates)

    Returns (client, assistant, vector_store)
    """
    # ─── 1) ENV ──────────────────────────────────────────────────────────────────
    load_dotenv()
    assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY missing in env"

    # ─── 2) CLIENT ──────────────────────────────────────────────────────────────
    client = OpenAI()

    # ─── 3) VECTOR STORE ────────────────────────────────────────────────────────
    # Try to find an existing store with the desired name
    existing_store = next(
        (vs for vs in client.vector_stores.list().data if vs.name == vector_store_name),
        None,
    )

    # if existing_store:
    #     vector_store = existing_store
    #     print(f"Re-using existing vector store: {vector_store.id}")
    # else:
    vector_store = client.vector_stores.create(name=vector_store_name)
        # print(f"Created new vector store:    {vector_store.id}")

    # ─── 4) ASSISTANT ───────────────────────────────────────────────────────────
    assistant = client.beta.assistants.create(
        name=assistant_name,
        instructions=instructions,
        model=model,
        tools=[{"type": "file_search"}],
    )

    # Point the assistant at our vector store
    client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

    # ─── 5) FILE UPLOAD ─────────────────────────────────────────────────────────
    # To avoid re-uploading files that are already present, collect existing file IDs
    existing_file_ids = {f.id for f in client.vector_stores.files.list(vector_store.id).data}

    # Filter paths whose SHA (or file name) isn’t already in the store
    files_to_upload = [
        open(p, "rb") for p in file_paths
        if pathlib.Path(p).stem not in existing_file_ids  # simple heuristic
    ]

    if files_to_upload:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=files_to_upload,
        )
        print(f"Upload status: {batch.status} — {batch.file_counts}")
    else:
        print("All provided files are already in the vector store; nothing uploaded.")

    return client, assistant, vector_store

def ask_financial_question(
    client: OpenAI,
    assistant: Any,
    vector_store: Any,
    question: str
) -> Tuple[str, List[str]]:
    """
    1) Sends a question to the assistant via a new thread
    2) Waits for the run to complete
    3) Extracts the answer text and any file citations
    4) Returns (answer_text, citations)
    """
    # 1) Create thread
    thread = client.beta.threads.create(
        messages=[{"role": "user", "content": question}],
        tool_resources={
            "file_search": {"vector_store_ids": [vector_store.id]}
        }
    )

    # 2) Poll until done
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id
    )

    # 3) Gather messages & annotations
    messages = list(
        client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id)
    )
    content = messages[0].content[0].text
    citations = []
    for idx, ann in enumerate(content.annotations):
        # replace the annotated text with an index placeholder
        content.value = content.value.replace(ann.text, f"[{idx}]")
        if file_cit := getattr(ann, "file_citation", None):
            cited = client.files.retrieve(file_cit.file_id)
            citations.append(f"[{idx}] {cited.filename}")

    # 4) Return
    return content.value, citations




file_paths = [r"C:\Users\amirb\Desktop\pdf_parser\pdfs\mobily\mobily_earnings_fy2024.pdf", 
              r"C:\Users\amirb\Desktop\pdf_parser\pdfs\stc\stc_earnings_fy2024.pdf",
             r"C:\Users\amirb\Desktop\pdf_parser\pdfs\zain\zain_earnings_fy2024.pdf",
             r"C:\Users\amirb\Downloads\Earnings_Presentation_FY+2023.pdf",
             r"C:\Users\amirb\Downloads\Earnings_Presentation_FY-2022+(1).pdf",
             r"C:\Users\amirb\Downloads\Earnings_Presentation_Q1+2025_.pdf"]


client, assistant, vs = setup_financial_assistant(file_paths)

# ans, cits = ask_financial_question(
#         client, assistant, vs,
#         "What are the total revenues of Mobily, Zain and for STC KSA (not STC Group) for 2024?"
#     )
# print(ans)
# print("\n".join(cits))








import sys
import os

base = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(base, "Tools"))

sys.path.append(base)


from PDF.ingestion import initialize_services
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from typing import Dict, Any, Annotated, Literal



class RagInput(TypedDict):
    question: str
    answer: str 



def agentic_rag_tool(
    input: RagInput,
    client: OpenAI,
    assistant: Any,
    vector_store: Any
) -> Dict[str, str]:
    """
    LangGraph-compatible wrapper — now simply asks the OpenAI assistant
    using our helper and returns its answer.
    """
    question = input["question"]
    # send the question off to the pre-configured assistant + vector store
    answer_text, citations = ask_financial_question(
        client, assistant, vector_store, question
    )
    # you could log or surface citations here if you want
    return {"answer": answer_text}

from langgraph.graph import StateGraph, START, END

def build_graph():
    pdf_graph = StateGraph(RagInput)

    # Add the main tool node, capturing client/assistant/vector_store from outer scope
    pdf_graph.add_node(
        "agentic_rag",
        lambda inp: agentic_rag_tool(inp, client, assistant, vs)
    )

    # Wire up start → agentic_rag → end
    pdf_graph.add_edge(START, "agentic_rag")
    pdf_graph.add_edge("agentic_rag", END)

    return pdf_graph.compile()

# Now just call without args:
gpt_pdf_graph = build_graph()


# if __name__ == "__main__":
#     test_input = {
#         "question": "What are the total revenues of STC KSA (not group), Mobily, and Zain in 2024 based on the reports?"
#     }
#     output = gpt_pdf_graph.invoke(test_input)
#     print(output["answer"])


