import math
from typing import Dict, Any, List, Tuple, Optional, Set
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from pydantic import BaseModel
import getpass
import os
# from helpers.config import GOOGLE_API_KEY  # Load LLM model config
import json
from langchain_openai import ChatOpenAI
# from helpers.config import GPT_MODEL
# if not os.environ.get("GOOGLE_API_KEY"):
#     if not GOOGLE_API_KEY:
#         raise RuntimeError("Google API key not found in environment or helpers.config.")
#     os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

from langchain.chat_models import init_chat_model

#llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")
# Initialize LLM
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

# --- Data Models ---
class QuestionAnalysisItem(BaseModel):
    question: str
    company: Optional[str] = None
    year: Optional[int] = None
    info_request: Optional[str] = None

class QueryAnalysis(BaseModel):
    needs_decomposition: bool
    items: List[QuestionAnalysisItem]

class FallbackDecomposition(BaseModel):
    items: List[QuestionAnalysisItem]

class ParentRanking(BaseModel):
    parent_ids: List[str]


# --- LLM wrappers (assume existing) ---
analyzer = llm.with_structured_output(QueryAnalysis)
decomposer = llm.with_structured_output(FallbackDecomposition)
ranker = llm.with_structured_output(ParentRanking)


# --- Alias / normalization helpers ---
ALIAS_MAP = {
    "stc ksa": "stc",
    "stc group": "stc"
}


# ——— 1️⃣ Decompose & Analyze ———
def analyze_query(query: str) -> QueryAnalysis:
    print("=== [DEBUG] analyze_query: Start ===")
    print(f"[DEBUG] Input query: {query}")
    prompt = HumanMessage(content=(
            "=== Section 0: Example ===\n"
            "Input: \"Breakdown the revenue by business, wholesale, consumer and others of mobily, stc ksa, and zain companies in 2024\"\n"
            "Expected Output:\n"
            "{\n"
            "  \"needs_decomposition\": true,\n"
            "  \"items\": [\n"
            "    {\"question\": \"Breakdown the revenue by business, wholesale, consumer and others of mobily in 2024\", "
                "\"company\": \"mobily\", \"year\": 2024, "
                "\"info_request\": \"Breakdown the revenue of mobily by business, wholesale, consumer and others\"},\n"
            "    {\"question\": \"Breakdown the revenue by business, wholesale, consumer and others of stc ksa in 2024\", "
                "\"company\": \"stc\", \"year\": 2024, "
                "\"info_request\": \"Breakdown the revenue of stc ksa by business, wholesale, consumer and others\"},\n"
            "    {\"question\": \"Breakdown the revenue by business, wholesale, consumer and others of zain in 2024\", "
                "\"company\": \"zain\", \"year\": 2024, "
                "\"info_request\": \"Breakdown the revenue of zain by business, wholesale, consumer and others\"}\n"
            "  ]\n"
            "}\n\n"
            "=== Section 1: Decomposition Rules ===\n"
            "1. Normalize any mention of 'stc ksa' or 'stc group' to company='stc'.\n"
            "2. If the user lists multiple companies, set needs_decomposition=true and split into one sub‑question per company.\n"
            "3. Preserve the original metric/request text (here, 'Breakdown the revenue by business, wholesale, consumer and others') as info_request.\n"
            "4. The info_request should contain the main keywords of the question such as the noun, verb, adjective and any word that is influential.\n"
            "5. Always extract the year if present.\n\n"
            "=== Section 2: Output Schema ===\n"
            "Return a JSON object with:\n"
            "- needs_decomposition: boolean\n"
            "- items: list of { question: str, company: str, year: int, info_request: str }\n\n"
            f"Now decompose: '{query}'"
    ))
    result = analyzer.invoke([prompt])
    print(f"[DEBUG] analyze_query output: {result.json()}")
    print("=== [DEBUG] analyze_query: End ===")
    return result


def canonical_company(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    key = raw.lower().strip()
    mapped = ALIAS_MAP.get(key, key)
    print(f"[NORMALIZATION] Raw company '{raw}' → canonical '{mapped}'")
    return mapped

def build_filter(company: Optional[str], extra: Optional[dict] = None) -> dict:
    filt: dict = {}
    if company:
        filt["company_name"] = canonical_company(company)
    if extra:
        filt.update(extra)
    print(f"[FILTER BUILD] Built filter: {filt}")
    return filt


def extract_keywords(text: str, llm) -> List[str]:
    prompt = HumanMessage(content=(
        "Extract keywords (as a JSON array of strings) that capture the main concepts from the following text:\n\n"
        f"{text}\n\n"
        "Respond with only the JSON array."
    ))
    resp = llm.invoke([prompt]).content.strip()
    try:
        keywords = json.loads(resp)
        if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
            return keywords
    except json.JSONDecodeError:
        pass
    # fallback: simple split
    return [w for w in set(text.lower().split()) if len(w) > 4][:5]




# --- Keyword extraction from user query ---
def query_keywords(query: str, llm, max_kw: int = 8) -> List[str]:
    print("=== [KEYWORD EXTRACTION] Start ===")
    print(f"[KEYWORD EXTRACTION] Input query: {query}")
    try:
        kws = extract_keywords(query, llm)[:max_kw]
        print(f"[KEYWORD EXTRACTION] Raw extracted keywords: {kws}")
    except Exception as e:
        print(f"[KEYWORD EXTRACTION] LLM keyword extraction failed: {e}")
        kws = []
    normalized = [k.lower() for k in kws if isinstance(k, str)]
    print(f"[KEYWORD EXTRACTION] Normalized keywords: {normalized}")
    print("=== [KEYWORD EXTRACTION] End ===\n")
    return normalized


def hybrid_with_score(vector_store, query: str, flt: dict, k: int):
    print("=== [HYBRID SEARCH] Start ===")
    print(f"[HYBRID SEARCH] Query: {query}")
    print(f"[HYBRID SEARCH] Filter: {flt}")
    print(f"[HYBRID SEARCH] K: {k}")
    print(f"[HYBRID SEARCH] Available attrs: {[attr for attr in dir(vector_store) if not attr.startswith('_')]}")
    
    # Prefer hybrid_search_with_score if it's a callable method
    hsws = getattr(vector_store, "hybrid_search_with_score", None)
    if callable(hsws):
        print("[HYBRID SEARCH] Calling hybrid_search_with_score")
        hits = hsws(query, k=k, filter=flt, alpha=0.65)
        print(f"[HYBRID SEARCH] Received {len(hits)} hits from hybrid_search_with_score")
        print("=== [HYBRID SEARCH] End ===\n")
        return hits

    # Next try hybrid_search if callable
    hs = getattr(vector_store, "hybrid_search", None)
    if callable(hs):
        print("[HYBRID SEARCH] Calling hybrid_search")
        try:
            hits = hs(query, k=k, filter=flt, alpha=0.65)
        except TypeError:
            # Some wrappers have different signature; try without alpha
            hits = hs(query, k=k, filter=flt)
        # Normalize to (doc, score) if needed
        if isinstance(hits, list) and hits and isinstance(hits[0], Document):
            result = [(doc, 0.0) for doc in hits]
        else:
            result = hits
        print(f"[HYBRID SEARCH] Received {len(result)} hits from hybrid_search")
        print("=== [HYBRID SEARCH] End ===\n")
        return result

    # Fallback to vector similarity only
    sim = getattr(vector_store, "similarity_search_with_score", None)
    if callable(sim):
        print("[HYBRID SEARCH] Calling similarity_search_with_score (fallback)")
        vec_hits = sim(query, k=k, filter=flt)
        print(f"[HYBRID SEARCH] Received {len(vec_hits)} vector-only hits")
        print("=== [HYBRID SEARCH] End ===\n")
        return vec_hits

    raise RuntimeError("No suitable search method found on vector_store.")

def retrieve_snippets(
    item: QuestionAnalysisItem,
    vector_store,
    llm,
    k: int = 7
) -> Tuple[List[Tuple[Document, float]], List[Document]]:
    print("\n=== [RETRIEVE SNIPPETS] Start ===")
    print(f"[RETRIEVE SNIPPETS] Sub-question: {item.question}")
    print(f"[RETRIEVE SNIPPETS] Company raw: {item.company}")

    # 1. Extract keywords from the sub-question
    kw_query = query_keywords(item.question, llm)
    print(f"[RETRIEVE SNIPPETS] Query keywords: {kw_query}")

    # 2. Build base company filter
    company_filter = build_filter(item.company)  # e.g., {"company_name": "mobily"}
    print(f"[RETRIEVE SNIPPETS] Company filter: {company_filter}")

    # 3. Build keyword-constrained filter using $in on the keywords array
    if kw_query:
        print(f"[RETRIEVE SNIPPETS] Adding keyword overlap constraint with $in: {kw_query}")
        keyword_filter = {"keywords": {"$in": kw_query}}
        combined_filter = {"$and": [company_filter, keyword_filter]}
    else:
        combined_filter = company_filter
    print(f"[RETRIEVE SNIPPETS] Combined filter before entry_type: {combined_filter}")

    # 4. Restrict to content/question entry types
    primary_filter = {
        "$and": [
            combined_filter,
            {"entry_type": {"$in": ["content", "question"]}}
        ]
    }
    print(f"[RETRIEVE SNIPPETS] Primary filter for candidate selection: {primary_filter}")

    # 5. Hybrid/vector search to get top candidate snippets
    hits = hybrid_with_score(vector_store, item.question, primary_filter, k=k)

    # Normalize if the wrapper returned raw Documents
    if hits and isinstance(hits[0], Document):
        print("[RETRIEVE SNIPPETS] Normalizing doc-only hits to (doc, score) tuples with dummy scores")
        hits = [(doc, 0.0) for doc in hits]

    print(f"[RETRIEVE SNIPPETS] Retrieved {len(hits)} primary hits with keyword filter applied.")

    # 6. Fallback: if no hits, relax to company + entry_type only
    if not hits:
        print("[RETRIEVE SNIPPETS] No hits found with keyword constraint. Falling back to company-only filter.")
        relaxed_filter = {
            "$and": [
                build_filter(item.company),
                {"entry_type": {"$in": ["content", "question"]}}
            ]
        }
        print(f"[RETRIEVE SNIPPETS] Relaxed primary filter: {relaxed_filter}")
        hits = hybrid_with_score(vector_store, item.question, relaxed_filter, k=k)
        if hits and isinstance(hits[0], Document):
            hits = [(doc, 0.0) for doc in hits]
        print(f"[RETRIEVE SNIPPETS] Retrieved {len(hits)} hits after fallback.")

    # 7. Collect unique parent_ids from those hits
    parent_ids = []
    for doc, _ in hits:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in parent_ids:
            parent_ids.append(pid)
    print(f"[RETRIEVE SNIPPETS] Extracted parent_ids from primary hits: {parent_ids}")

    # 8. Fetch sibling documents (main + content + question) for each parent_id
    siblings: List[Document] = []
    for pid in parent_ids:
        print(f"[RETRIEVE SNIPPETS] Fetching siblings for parent_id: {pid}")
        sibling_filter = build_filter(item.company, {"parent_id": pid})
        print(f"[RETRIEVE SNIPPETS] Sibling filter: {sibling_filter}")
        sibs = vector_store.similarity_search("", k=50, filter=sibling_filter)
        print(f"[RETRIEVE SNIPPETS] Retrieved {len(sibs)} sibling docs for {pid}")
        siblings.extend(sibs)

    print(f"[RETRIEVE SNIPPETS] Final primary hits used: {len(hits)}; total siblings collected: {len(siblings)}")
    print("=== [RETRIEVE SNIPPETS] End ===\n")
    return hits, siblings


# --- Parent ranking with heuristic + optional LLM refinement ---
def rank_parent_ids(
    item: QuestionAnalysisItem,
    hits: List[Tuple[Document, float]]
) -> List[str]:
    print("=== [RANK PARENTS] Start ===")
    print(f"[RANK PARENTS] Sub-question: {item.question}")
    # Aggregate scores per parent_id
    score_map: Dict[str, List[float]] = defaultdict(list)
    for doc, score in hits:
        pid = doc.metadata.get("parent_id")
        if pid is not None:
            score_map[pid].append(score)

    # Compute average score (lower is assumed better if similarity returns distance-like; adjust if different)
    parent_avg = []
    for pid, scores in score_map.items():
        avg = sum(scores) / len(scores) if scores else float("inf")
        parent_avg.append((pid, avg))
    parent_avg.sort(key=lambda x: x[1])
    heuristic_order = [pid for pid, _ in parent_avg]
    print(f"[RANK PARENTS] Heuristic ranking (by avg score): {heuristic_order}")

    parent_ids = heuristic_order

    # LLM refinement if too many candidates
    if len(parent_ids) > 3:
        print("[RANK PARENTS] More than 3 parent candidates, invoking LLM to refine order.")
        prompt_lines = [
            f"Sub-question: {item.question}",
            "Parents and their representative snippets:"
        ]
        for pid, _ in parent_avg:
            prompt_lines.append(f"{pid}:")
            for doc, _ in hits:
                if doc.metadata.get("parent_id") == pid:
                    snippet = doc.page_content.strip().replace("\n", " ")
                    prompt_lines.append(f"  - {snippet}")
        prompt_lines.append("Return a JSON array of the most relevant parent_ids in order, dropping irrelevant ones.")
        prompt_str = "\n".join(prompt_lines)
        print(f"[RANK PARENTS] Prompt to ranker LLM:\n{prompt_str}")
        resp = ranker.invoke([HumanMessage(prompt_str)])
        print(f"[RANK PARENTS] LLM output: {resp.json()}")
        parent_ids = resp.parent_ids

    final_parents = parent_ids[:5]
    print(f"[RANK PARENTS] Final parent_ids used (capped to 5): {final_parents}")
    print("=== [RANK PARENTS] End ===\n")
    return final_parents

def answer_subquestion(
    item: QuestionAnalysisItem,
    hits: List[Tuple[Document, float]],
    siblings: List[Document],
    vector_store,
    llm
) -> str:
    print("=== [ANSWER SUBQUESTION] Start ===")
    print(f"[ANSWER SUBQUESTION] Sub-question: {item.question}")

    # 1. Rank parent IDs
    parent_ids = rank_parent_ids(item, hits)
    print(f"[ANSWER SUBQUESTION] Using parent_ids: {parent_ids}")

    # 2. Collect context per parent_id: main, all content, plus optionally best-matching stored question
    context_parts: List[str] = []
    seen_texts: Set[str] = set()

    # Build a simple similarity function for matching stored question snippets to the query (optional)
    def simple_overlap(a: str, b: str) -> int:
        return len(set(a.lower().split()) & set(b.lower().split()))

    for pid in parent_ids:
        print(f"\n--- [CONTEXT ASSEMBLY] Parent ID: {pid} ---")

        # a) Fetch main summary for this parent_id
        main_filter = {
            "$and": [
                build_filter(item.company),
                {"entry_type": "main"},
                {"parent_id": pid}
            ]
        }
        mains = vector_store.similarity_search("", k=1, filter=main_filter)
        if mains:
            main = mains[0]
            if main.page_content not in seen_texts:
                context_parts.append(f"[MAIN - {pid}]\n{main.page_content}")
                seen_texts.add(main.page_content)
            print(f"[CONTEXT ASSEMBLY] Included MAIN: {main.page_content[:200].replace(chr(10), ' ')}...")
        else:
            print(f"[CONTEXT ASSEMBLY] No MAIN found for {pid}")

        # b) Include all content bullets from siblings matching this parent_id
        for doc in siblings:
            if doc.metadata.get("parent_id") != pid:
                continue
            if doc.metadata.get("entry_type") == "content":
                piece = doc.page_content.strip()
                if piece and piece not in seen_texts:
                    context_parts.append(f"[CONTENT - {pid}]\n{piece}")
                    seen_texts.add(piece)
                    print(f"[CONTEXT ASSEMBLY] Included CONTENT bullet: {piece[:150]}...")

        # c) Optionally include the most similar stored question snippet (if any)
        question_snips = [d for d in siblings if d.metadata.get("parent_id") == pid and d.metadata.get("entry_type") == "question"]
        if question_snips:
            # pick best matching stored question to the user sub-question
            best_q = max(question_snips, key=lambda d: simple_overlap(d.page_content, item.question))
            overlap_score = simple_overlap(best_q.page_content, item.question)
            if best_q.page_content not in seen_texts:
                context_parts.append(f"[STORED QUESTION - {pid} (overlap={overlap_score})]\n{best_q.page_content}")
                seen_texts.add(best_q.page_content)
                print(f"[CONTEXT ASSEMBLY] Included STORED QUESTION (score={overlap_score}): {best_q.page_content[:150]}...")
        else:
            print(f"[CONTEXT ASSEMBLY] No stored question snippets for {pid}")

    # 3. Final assembled context
    context = "\n\n".join(context_parts)
    print("\n=== [FINAL CONTEXT SENT TO LLM] ===")
    for i, part in enumerate(context_parts, 1):
        print(f"\n--- Context piece {i} ---\n{part}\n")
    print("=== [END CONTEXT DUMP] ===\n")

    # 4. Invoke LLM
    system_msg = SystemMessage("Answer strictly from the provided context. Use concise bullet points. Highlight numeric facts clearly.")
    human_msg = HumanMessage(content=f"Context:\n{context}\n\nQuestion: {item.question}")
    ans = llm.invoke([system_msg, human_msg]).content

    print(f"[ANSWER SUBQUESTION] Generated answer:\n{ans}\n")
    print("=== [ANSWER SUBQUESTION] End ===\n")
    return ans



# --- Orchestrator ---
def answer_with_context(
    query: str,
    vector_store,
    llm,
    k: int = 7,
    seen_parent_ids: Optional[Set[str]] = None
) -> str:
    print("=== [ORCHESTRATOR] Start ===")
    print(f"[ORCHESTRATOR] Received user query: {query}")
    seen_parent_ids = seen_parent_ids or set()
    analysis: QueryAnalysis = analyze_query(query)
    print(f"[ORCHESTRATOR] Decomposition result: needs_decomposition={analysis.needs_decomposition}, items={[i.dict() for i in analysis.items]}")
    sections: List[str] = []

    for item in analysis.items:
        print(f"\n--- [ORCHESTRATOR] Processing sub-question: {item.question} ---")
        hits, siblings = retrieve_snippets(item, vector_store, llm, k=k)

        # Filter out already-seen parent_ids to avoid repetition
        filtered_hits = []
        for doc, score in hits:
            pid = doc.metadata.get("parent_id")
            if pid and pid in seen_parent_ids:
                print(f"[ORCHESTRATOR] Skipping already-seen parent_id: {pid}")
                continue
            filtered_hits.append((doc, score))
        hits = filtered_hits

        # Update memory of seen parent_ids
        for doc, _ in hits:
            pid = doc.metadata.get("parent_id")
            if pid:
                seen_parent_ids.add(pid)
                print(f"[ORCHESTRATOR] Marking parent_id seen: {pid}")

        answer = answer_subquestion(item, hits, siblings, vector_store, llm)
        sections.append(f"### {item.question}\n\n{answer}")

    final_md = "\n\n".join(sections)
    print("=== [ORCHESTRATOR] End ===\n")
    return final_md


# query = "what is the revenue of stc ksa not stc group in 2024"
# result_markdown = answer_with_context(
#     query=query,
#     vector_store=vector_store,
#     llm=llm,
#     k=7,  # number of top snippets to consider
#     seen_parent_ids=set()  # pass empty set on first call; reuse between related queries to avoid repetition
# )

# print("=== Final Answer ===")
# print(result_markdown)





from typing import Optional, Set

def ask_agent(
    question: str,
    vector_store: AstraDBVectorStore,
    generator_model,  # your llm (e.g., ChatOpenAI or gemini wrapper)
    k: int = 7,
    seen_parent_ids: Optional[Set[str]] = None,
) -> str:
    """The main function that orchestrates the agentic RAG process."""
    print("\n\n--- Starting Agentic Query ---")
    if not vector_store:
        raise RuntimeError("Vector store is not available.")

    seen_parent_ids = seen_parent_ids or set()

    # 1. Plan / decompose
    analysis: QueryAnalysis = analyze_query(question)  # uses generator_model internally via global `llm`
    print(f"[ask_agent] Decomposition: needs_decomposition={analysis.needs_decomposition}, items={[i.dict() for i in analysis.items]}")

    sections: List[str] = []

    # 2. Retrieve per sub-question and answer
    for item in analysis.items:
        print(f"\n--- [ask_agent] Processing sub-question: {item.question} ---")
        hits, siblings = retrieve_snippets(item, vector_store, generator_model, k=k)

        # Filter out already-seen parents to avoid repetition
        filtered_hits = []
        for doc, score in hits:
            pid = doc.metadata.get("parent_id")
            if pid and pid in seen_parent_ids:
                print(f"[ask_agent] Skipping already-seen parent_id: {pid}")
                continue
            filtered_hits.append((doc, score))
        hits = filtered_hits

        # Mark seen
        for doc, _ in hits:
            pid = doc.metadata.get("parent_id")
            if pid:
                seen_parent_ids.add(pid)
                print(f"[ask_agent] Marking parent_id seen: {pid}")

        answer = answer_subquestion(item, hits, siblings, vector_store, generator_model)
        sections.append(f"### {item.question}\n\n{answer}")

    final_md = "\n\n".join(sections)

    print("\n\n===================================")
    print("          FINAL ANSWER")
    print("===================================\n")
    print(final_md)
    print("\n--- Agentic Query Complete ---")
    return final_md




# result_markdown = ask_agent(
#     question="what is the revenue of stc ksa not stc group in 2024",
#     vector_store=vector_store,
#     generator_model=llm,
#     k=7,
#     seen_parent_ids=set()
# )
