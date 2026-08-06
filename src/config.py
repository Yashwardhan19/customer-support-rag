import os

# Path
DATA_DIR = "data"
QDRANT_PATH = "qdrant_data"          

# Chunking 
CHUNK_SIZE = 300      
CHUNK_OVERLAP = 100   

# Qdrant 
COLLECTION_NAME = "support_docs"

# Embeddings 
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  

# PDF text/scan detection 
MIN_TEXT_LENGTH_THRESHOLD = 20

# Vision model (Gemini) 
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
PAGE_RENDER_DPI = 200  

VISION_PROMPT = (
    "You are extracting content from a document image for a search index. "
    "Transcribe all readable body text verbatim. "
    "If there are tables, output each one as pipe-separated rows (col1 | col2 | col3). "
    "If there are charts or graphs, describe them precisely: chart type, title, "
    "axis labels/units, key data points or values, and the overall trend shown. "
    "Do not add commentary, opinions, or formatting like markdown headers - "
    "output plain text only."
)

# Supported file extensions 
TEXT_EXTENSIONS = (".txt",)
PDF_EXTENSIONS = (".pdf",)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")