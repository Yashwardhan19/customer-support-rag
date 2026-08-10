import os
from dotenv import load_dotenv

load_dotenv() 

# Path
DATA_DIR = "data"
QDRANT_PATH = "qdrant_data"          

# Chunking 
CHUNK_SIZE = 300      
CHUNK_OVERLAP = 100   

# Qdrant 
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "document_chunks"
QDRANT_UPSERT_BATCH_SIZE = 100

# Embeddings 

GEMINI_EMBEDDING_MODEL_NAME = "gemini-embedding-001"
EMBEDDING_OUTPUT_DIM = 3072      # gemini-embedding-001 default; can reduce (e.g. 768) for smaller storage/faster search with a small quality trade-off
EMBEDDING_MAX_WORKERS = 8        # parallel API requests — tune based on your rate limits


# PDF text/scan detection 
MIN_TEXT_LENGTH_THRESHOLD = 20

# Vision model (Gemini) 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash-lite")
PAGE_RENDER_DPI = 200  

# --- Vision prompts used by src/loaders.py ---

# Used when a page has a real text layer + tables already extracted separately.
# Vision's ONLY job is to catch charts/graphs/diagrams the text layer misses.
VISION_PROMPT_CHART_ONLY = (
    "This image is a rendered PDF page. The page's plain text and any tables "
    "have already been extracted separately — do NOT transcribe or repeat them. "
    "Only describe non-text visual elements: charts, graphs, diagrams, or "
    "illustrations. Respond in PLAIN TEXT ONLY — no markdown syntax "
    "(no #, *, |, ```, or --- characters).\n"
    "For each chart/graph, state its title, axis labels, and approximate data "
    "values as plain sentences. For diagrams, describe the flow/structure shown "
    "as a plain sentence.\n"
    "If there are no charts, graphs, or diagrams on the page, respond with "
    "exactly: NO_VISUAL_CONTENT"
)

# Used when a page has NO usable text layer (scanned) or is a standalone image —
# vision is the only source of truth, so it must capture everything.
VISION_PROMPT_FULL_PAGE = (
    "This image is a scanned PDF page with no separate text extraction available. "
    "Transcribe ALL content on the page in PLAIN TEXT ONLY — no markdown syntax "
    "(no #, *, |, ```, or --- characters).\n"
    "1. All body text, in reading order, as plain sentences.\n"
    "2. Any tables — write each row as a plain sentence, e.g. "
    "'ID 101, Name Alice, Department Engineering, Salary 75000.'\n"
    "3. Any charts, graphs, or diagrams — describe title, axes, and approximate "
    "data values, or the structure/flow shown, as plain sentences.\n"
    "Be complete and do not omit any section of the page."
)

# Supported file extensions 
TEXT_EXTENSIONS = (".txt",)
PDF_EXTENSIONS = (".pdf",)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

#text generation

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "openai/gpt-oss-120b"
GROQ_TEMPERATURE = 0.3          # lower = more deterministic/grounded answers, good default for RAG
GROQ_REASONING_EFFORT = "medium"  # "low" | "medium" | "high" — gpt-oss-120b specific param
RETRIEVAL_TOP_K = 5             # how many chunks to retrieve per query

MIN_RELEVANCE_SCORE = 0.60



HISTORY_TURNS = 3  # number of prior user+assistant exchanges to include when
                    # rewriting a follow-up question into a standalone one


DATABASE_URL = os.getenv("DATABASE_URL")  # e.g. postgresql://user:pass@host/db?sslmode=require