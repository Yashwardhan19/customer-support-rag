"""
INGEST SCRIPT (Qdrant version)
-------------------------------
This script does the following:
1. Reads .txt and .pdf files from the data/ folder
2. Splits the text into small chunks
3. Generates an embedding for each chunk (using sentence-transformers, free & local)
4. Stores everything in a Qdrant collection (local embedded mode - no server/docker needed)

Run: python ingest.py
"""

import os
import glob
import uuid
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
# from embeddings import get_embedding
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# OCR-related imports (used only when a PDF has no extractable text layer,
# i.e. it's a scanned document / image-based PDF)
import pytesseract
from pdf2image import convert_from_path

DATA_DIR = "data"
CHUNK_SIZE = 300       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between chunks (to reduce loss of context at chunk boundaries)
QDRANT_PATH = "qdrant_data"          # local disk folder - embedded mode, no server needed
COLLECTION_NAME = "support_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # free, fast, small (384 dim)

# Minimum characters we expect from a normal text-based PDF page.
# If pypdf extracts less than this, we treat the page as scanned/image-based
# and fall back to OCR instead.
MIN_TEXT_LENGTH_THRESHOLD = 20

# OCR language(s) for Tesseract. "eng" = English. For Hindi text use "hin",
# or "eng+hin" for documents that mix both languages.
OCR_LANGUAGES = "eng"


def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def ocr_pdf(path):
    """
    Extract text from a scanned/image-based PDF using OCR.
    Steps:
      1. Convert each PDF page into an image (using pdf2image / Poppler)
      2. Run Tesseract OCR on each image to recognize the text
      3. Combine text from all pages
    """
    print(f"      -> Running OCR on '{os.path.basename(path)}' (this may take a moment)...")
    images = convert_from_path(path, dpi=300)  # higher DPI = better OCR accuracy, but slower
    text = ""
    for i, image in enumerate(images):
        page_text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES)
        text += page_text + "\n"
    return text


def read_pdf(path):
    """
    Reads a PDF. Tries normal text extraction first (fast, works for
    regular digital PDFs). If that yields little or no text, assumes the
    PDF is scanned/image-based and falls back to OCR automatically.
    """
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"

    if len(text.strip()) < MIN_TEXT_LENGTH_THRESHOLD:
        print(f"      -> No text layer found in '{os.path.basename(path)}', treating as scanned PDF.")
        text = ocr_pdf(path)

    return text


def load_documents():
    """Load all .txt and .pdf files from the data/ folder."""
    docs = []
    for path in glob.glob(os.path.join(DATA_DIR, "*")):
        if path.lower().endswith(".txt"):
            docs.append({"source": os.path.basename(path), "text": read_txt(path)})
        elif path.lower().endswith(".pdf"):
            docs.append({"source": os.path.basename(path), "text": read_pdf(path)})
    return docs


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Simple sliding-window chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def main():
    print("Step 1: Loading documents from 'data/' folder...")
    docs = load_documents()
    if not docs:
        print("Warning: no .txt or .pdf files found in 'data/' folder.")
        print("   Add a sample .txt file and run again.")
        return
    print(f"   {len(docs)} document(s) loaded.")

    print("Step 2: Chunking text...")
    all_chunks = []  # list of dicts: {source, text}
    for doc in docs:
        for chunk in chunk_text(doc["text"]):
            all_chunks.append({"source": doc["source"], "text": chunk})
    print(f"   {len(all_chunks)} chunks created.")

    print(f"Step 3: Loading embedding model '{EMBED_MODEL_NAME}' (first time may take a moment)...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Step 4: Generating embeddings...")

    texts = [c["text"] for c in all_chunks]
    
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]

    # embeddings = []

    # print("Generating Gemini embeddings...")

    # for chunk in tqdm(all_chunks):
    #     vector = get_embedding(chunk["text"])
    #     embeddings.append(vector)

    # dim = len(embeddings[0])

    print("Step 5: Connecting to Qdrant (local embedded mode)...")
    client = QdrantClient(path=QDRANT_PATH)

    # Delete the old collection if it already exists, so we always start fresh
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    print("Step 6: Uploading points to Qdrant...")
    points = []
    for i, (chunk, vector) in enumerate(zip(all_chunks, embeddings)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={"source": chunk["source"], "text": chunk["text"]},
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    client.close()

    print(f"Done! Qdrant collection '{COLLECTION_NAME}' saved at ./{QDRANT_PATH}")
    print(f"   Total chunks indexed: {len(all_chunks)}")


if __name__ == "__main__":
    main()