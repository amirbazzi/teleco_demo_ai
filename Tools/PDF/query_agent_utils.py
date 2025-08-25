from dotenv import load_dotenv
load_dotenv()
import os
import tempfile
import base64
from io import BytesIO
from typing import List, Dict, Any

# from PIL import Image
# from pdf2image import convert_from_path
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_astradb import AstraDBVectorStore
import json
# from PDF.ingestion import initialize_services

# from Tools.PDF.ranker import *

# --- CONFIGURATION ---
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
VERTEX_REGION = os.getenv("VERTEX_REGION")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_COLLECTION_NAME = os.getenv("ASTRA_DB_COLLECTION_NAME")
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# ---------------------

# ranker = Ranker(openai_api_key=os.getenv("OPENAI_API_KEY"))


def query_planner(user_question: str, generator_model: GenerativeModel) -> List[str]:
    """Decomposes a complex question into simpler sub-questions using a robust JSON output format."""
    print(f"\nPlanning query: '{user_question}'")
    prompt = f"""
    You are an expert query planner. Your task is to decompose a user's question into a series of simpler, self-contained sub-questions. This is crucial for retrieving information accurately from a database.

    CRITICAL RULE: If the question asks for the same fact for multiple distinct entities (e.g., "What are the revenues for STC, Zain, and Mobily?"), you MUST create a separate sub-question for each entity.

    Follow these rules:
    - **Multi-Entity Questions:** Decompose into one question per entity.
      Example: "What are the revenues for STC, Zain, and Mobily?" -> `["What is the revenue for STC?", "What is the revenue for Zain?", "What is the revenue for Mobily?"]`
    - **Comparative Questions:** Break down into one question per entity/metric pair.
      Example: "Compare the revenue of STC with the EBITDA of Zain." -> `["What is the revenue of STC?", "What is the EBITDA of Zain?"]`
    - **Simple Questions:** If the question is already simple, return it as a single-item list.
      Example: "What is STC's revenue?" -> `["What is STC's revenue?"]`

    Output Format: You MUST respond with a JSON object containing a single key "sub_queries" which holds a list of the generated question strings.
    Example JSON: {{"sub_queries": ["question 1", "question 2"]}}

    User Question: "{user_question}"
    """
    response = generator_model.generate_content(prompt) # for gemini models
    # response = generator_model.invoke(prompt)
    try:
        # Clean the response text to ensure it's valid JSON
        json_str = response.text.strip().replace("`json", "").replace("`", "")
        parsed_json = json.loads(json_str)
        sub_questions = parsed_json["sub_queries"]
        if isinstance(sub_questions, list) and sub_questions:
            print(f"  > Decomposed into {len(sub_questions)} sub-question(s): {sub_questions}")
            return sub_questions
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"  > Could not parse JSON, falling back to simple query. Error: {e}")
        return [user_question]
    return [user_question]


def expand_sub_question(sub_question: str, generator_model: GenerativeModel) -> List[str]:
    """Expands a single sub-question into multiple, semantically similar variants."""
    print(f"  > Expanding sub-question: '{sub_question}'")
    prompt = f"""
    You are a query expansion expert. Your task is to rewrite the given user question in 3 different ways to improve search recall in a vector database.
    The new questions should be semantically similar but use different phrasing and keywords. Focus on synonyms and rephrasing key concepts.
    You will also decompose the question if it contains multiple entities.

    - Original Question: "What was the 2024 revenue for STC?"
    - Example Output: {{"expanded_queries": ["What were the total sales for STC in 2024?", "Show the 2024 turnover for STC.", "What was the top-line financial performance of STC in 2024?"]}}

    - Original Question: "Can you please breakdown the total revenues for stc ksa, mobily, and zain into Consumer, Business, Wholesale and Others based on the reports"
    - Example Output: {{"expanded_queries": ["Can you provide the 2024 revenue breakdown for STC KSA by Consumer, Business, Wholesale, and Others?", "What were the 2024 revenues for Mobily in each segment: Consumer, Business, Wholesale, and Others?", "Please detail the 2024 revenue segments for Zain across Consumer, Business, Wholesale, and Others."]}}

    You MUST respond with a JSON object containing a single key "expanded_queries" which holds a list of the 3 generated question strings.

    Original Question: "{sub_question}"
    """
    response = generator_model.generate_content(prompt)
    # response = generator_model.invoke(prompt)

    try:
        json_str = response.text.strip().replace("`json", "").replace("`", "")
        parsed_json = json.loads(json_str)
        expanded = parsed_json["expanded_queries"]
        # Also include the original question for completeness
        all_queries = [sub_question] + expanded
        print(f"    - Expanded to: {all_queries}")
        return all_queries
    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"    - Could not expand query, using original. Error: {e}")
        return [sub_question]


def retrieve_context(sub_questions: List[str], vector_store: AstraDBVectorStore, generator_model: GenerativeModel) -> str:
    """Retrieves relevant context for a list of sub-questions, now with query expansion."""
    print("\nRetrieving context for sub-questions...")
    all_context = ""
    unique_doc_ids = set()

    for question in sub_questions:
        # Expand each sub-question
        expanded_queries = expand_sub_question(question, generator_model)

        # Search for all expanded queries
        retrieved_docs = []
        for eq in expanded_queries:
            retrieved_docs.extend(vector_store.similarity_search(eq, k=7)) # k=2 per variant

        # Filter for unique documents to avoid duplicates in context
        unique_docs_for_question = []
        for doc in retrieved_docs:
            # rank = ranker.extract_rank(user_query=question, docs=doc.page_content)
            # if rank["rank"] == "Yes":
            doc_id = f"{doc.metadata.get('source_pdf')}-{doc.metadata.get('page')}"
            if doc_id not in unique_doc_ids:
                unique_doc_ids.add(doc_id)
                unique_docs_for_question.append(doc)

        context_for_question = "\n\n---\n\n".join(
            [f"Context from {doc.metadata['source_pdf']} (Page {doc.metadata['page']} for company {doc.metadata['company_name']}):\n{doc.page_content}" for doc in unique_docs_for_question]
        )
        all_context += f"\n\n--- Context for sub-question: '{question}' ---\n{context_for_question}"

    print("✅ Context retrieval complete.")
    return all_context


def generate_final_answer(original_question: str, context: str, generator_model: GenerativeModel) -> str:
    """Uses the retrieved context and the original question to generate a final answer."""
    print("\nGenerating final answer...")
    prompt = f"""
    You are a helpful and accurate Q&A agent. Provide a comprehensive answer to the user's original question based *only* on the provided context.
    - Synthesize the information from the context.
    - For every piece of information or data point you use, you MUST cite the source PDF and page number, like `(Source: [source_pdf], Page: [page])`.
    - If the context does not contain enough information to answer any part of the question, state that clearly.
    - Structure your response clearly. Use bullet points for lists or comparisons.
    - Always extract numbers not only percentages, but also absolute values, and provide them in the answer.
    - when it comes to question about Zain exclude sales co and Tamam

    Do not hallucinate or state that data is unavailable when it exists in the source documents.

    This applies especially for Mobily and Zain. Their revenue and segment-level data must be retrieved accurately from the provided documents.

    For Mobily, total revenue and segment breakdown (Consumer, Business, Wholesale, Others) for both 2023 and 2024 are available on page 10 of Earnings_Presentation_FY_2024.pdf, which is included in the vector store and properly tagged.

    Use this document directly when responding to related queries. Do not respond with "data is missing" when the information is present in the source.

    ---
    CONTEXT:
    {context}
    ---
    ORIGINAL QUESTION:
    {original_question}
    ---
    Final Answer:
    """
    #response = generator_model.generate_content(prompt, generation_config={"temperature": 0})
    response = generator_model.generate_content(prompt, generation_config={"temperature": 0,"top_p": 1.0})

    print("✅ Final answer generated.")
    return response.text


def ask_agent(question: str, vector_store: AstraDBVectorStore, generator_model: GenerativeModel):
    """The main function that orchestrates the entire agentic RAG process."""
    print("\n\n--- Starting Agentic Query ---")
    if not vector_store:
        print("Error: Vector store is not available.")
        return

    sub_questions = query_planner(question, generator_model)
    context = retrieve_context(sub_questions, vector_store, generator_model)
    final_answer = generate_final_answer(question, context, generator_model)

    print("\n\n===================================")
    print("          FINAL ANSWER")
    print("===================================\n")
    print(final_answer)
    print("\n--- Agentic Query Complete ---")



