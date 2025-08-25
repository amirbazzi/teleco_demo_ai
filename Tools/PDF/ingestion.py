import sys
import os

# Navigate up two levels to the project root and add it to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from dotenv import load_dotenv
load_dotenv()
import os
import tempfile
import base64
from io import BytesIO
from typing import List, Dict, Any

# from PIL import Image
# from pdf2image import 
from pdf2image import convert_from_path

from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_astradb import AstraDBVectorStore

# --- CONFIGURATION ---
PROJECT_ID = os.getenv("PROJECT_ID")
BUCKET_NAME = os.getenv("BUCKET_NAME")
VERTEX_REGION = os.getenv("VERTEX_REGION")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_COLLECTION_NAME = os.getenv("ASTRA_DB_COLLECTION_NAME")
# ---------------------

def initialize_services():
    """Initializes Vertex AI, embedding model, and GCS client."""
    vertexai.init(project=PROJECT_ID, location=VERTEX_REGION)
    multimodal_model = GenerativeModel("gemini-1.5-flash")
    generator_model = GenerativeModel("gemini-1.5-pro")
    embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")
    storage_client = storage.Client(project=PROJECT_ID)
    return multimodal_model, generator_model, embedding_model, storage_client


def get_company_name_from_path(gcs_uri: str) -> str:
    """Extracts the company name from the GCS folder structure."""
    try:
        path_parts = gcs_uri.replace(f"gs://{BUCKET_NAME}/", "").split('/')
        if len(path_parts) > 1:
            return path_parts[0].replace('_', ' ').strip()
    except Exception:
        pass
    return "Unknown Company"


def parse_pdf_advanced(
    gcs_uri: str,
    company_name: str,
    multimodal_model: GenerativeModel,
    storage_client: storage.Client
) -> List[Dict[str, Any]]:
    """Downloads a PDF from GCS, converts pages to images, and uses a vision LLM to parse each page."""
    print(f"\nProcessing PDF for '{company_name}': {gcs_uri}")
    pdf_file_name = os.path.basename(gcs_uri)
    chunks: List[Dict[str, Any]] = []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_pdf:
        bucket = storage_client.bucket(gcs_uri.split('/')[2])
        blob = bucket.blob('/'.join(gcs_uri.split('/')[3:]))
        blob.download_to_filename(temp_pdf.name)

        images = convert_from_path(temp_pdf.name)
        for page_num, image in enumerate(images, start=1):
            try:
                print(f"  > Parsing page {page_num} with vision model...")
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                data = base64.b64encode(buffer.getvalue()).decode()

                prompt = (
                    "You are an expert financial document analyst. Analyze the provided image of a PDF page. "
                    "Transcribe all text, then summarize any charts or tables into a cohesive text chunk."
                )
                request_parts = [
                    Part.from_data(data=base64.b64decode(data), mime_type="image/png"),
                    Part.from_text(prompt)
                ]
                response = multimodal_model.generate_content(request_parts)
                page_content = response.text

                chunks.append({
                    "page_content": page_content,
                    "metadata": {
                        "company_name": company_name,
                        "source_pdf": pdf_file_name,
                        "page": page_num
                    }
                })
            except Exception as e:
                print(f"  ! Error processing page {page_num}: {e}")
    return chunks


def build_vector_store(
    multimodal_model: GenerativeModel,
    embedding_model: VertexAIEmbeddings,
    storage_client: storage.Client
) -> AstraDBVectorStore:
    """Runs offline ingestion: scans GCS for PDFs, parses and embeds all pages into Astra DB."""
    print("\n--- Starting Offline Ingestion Pipeline ---")
    all_chunks: List[Dict[str, Any]] = []
    blobs = storage_client.list_blobs(BUCKET_NAME)

    for blob in blobs:
        if blob.name.lower().endswith('.pdf'):
            uri = f"gs://{BUCKET_NAME}/{blob.name}"
            company = get_company_name_from_path(uri)
            all_chunks.extend(parse_pdf_advanced(uri, company, multimodal_model, storage_client))

    if not all_chunks:
        print("\nNo PDFs found or processed.")
        return None

    print(f"\nEmbedding {len(all_chunks)} chunks into Astra DB...")
    texts = [chunk['page_content'] for chunk in all_chunks]
    metas = [chunk['metadata'] for chunk in all_chunks]

    vector_store = AstraDBVectorStore(
        embedding=embedding_model,
        collection_name=ASTRA_DB_COLLECTION_NAME,
        token=ASTRA_DB_APPLICATION_TOKEN,
        api_endpoint=ASTRA_DB_API_ENDPOINT
    )
    vector_store.add_texts(texts=texts, metadatas=metas)

    print(f"✅ Vector Store updated in Astra DB collection: '{ASTRA_DB_COLLECTION_NAME}'")
    return vector_store
