FROM python:3.11-slim

# curl is only needed for the HEALTHCHECK below. build-essential is NOT needed —
# pymupdf/psycopg2-binary/etc. all ship prebuilt manylinux wheels for linux/amd64,
# so pip doesn't need a compiler here. Leaving it out cuts build time drastically.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the app
COPY . .

# Qdrant/Postgres/Gemini/Groq are all remote (Cloud) in this project, so no
# local data volume is strictly required — but keep a spot for any temp/cache
# files the app writes, in case QDRANT_PATH-style local paths are ever used.
RUN mkdir -p /app/data /app/output

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# --server.address=0.0.0.0 is required so Streamlit is reachable outside the container;
# --server.headless=true stops it from trying to open a local browser / prompting for email
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]