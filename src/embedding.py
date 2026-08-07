"""
embeddings.py

Generates embeddings for chunked documents using Google's gemini-embedding-001
API model. Reuses the same Gemini client/auth pattern already set up in
loaders.py (get_gemini_client), so no new API key or .env changes needed.

Requires:
    pip install -U google-genai   (already installed, used by loaders.py)

Usage:
    from src.loaders import load_documents
    from src.chunking import chunk_documents
    from src.embeddings import embed_documents

    docs = load_documents()
    docs = chunk_documents(docs)
    docs = embed_documents(docs)   # each chunk now has an "embedding" field

Note: gemini-embedding-001 accepts ONE input text per request (no native
batch endpoint like BGE-M3's encode()), so embed_texts() calls the API
once per chunk, using a thread pool to parallelize the I/O-bound calls
instead of running them one at a time sequentially.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src import config
from src.loaders import get_gemini_client  # reuse the same client/singleton
from google.genai import types


def embed_one(text: str, task_type: str) -> list[float]:
    """
    Embeds a single text string. task_type matters for retrieval quality:
      - "RETRIEVAL_DOCUMENT" -> use for chunks going INTO the vector store
      - "RETRIEVAL_QUERY"    -> use for a user's search query AT RETRIEVAL TIME
    Retries once on transient failure before giving up and returning None,
    so one flaky API call doesn't kill an entire batch run.
    """
    client = get_gemini_client()
    for attempt in range(2):
        try:
            response = client.models.embed_content(
                model=config.GEMINI_EMBEDDING_MODEL_NAME,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=config.EMBEDDING_OUTPUT_DIM,
                ),
            )
            return response.embeddings[0].values
        except Exception as e:
            if attempt == 0:
                time.sleep(1)
                continue
            print(f"      -> Warning: embedding failed for a chunk: {e}")
            return None


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT",
                 max_workers: int = None) -> list[list[float]]:
    """
    Embeds a list of texts in parallel (gemini-embedding-001 has no native
    batch call, so this fans out individual requests across a thread pool).
    Returns vectors in the SAME ORDER as the input list.
    """
    if not texts:
        return []

    max_workers = max_workers or config.EMBEDDING_MAX_WORKERS
    results = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(embed_one, text, task_type): i
            for i, text in enumerate(texts)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            results[i] = future.result()

    return results


def embed_chunks(chunks: list[dict], task_type: str = "RETRIEVAL_DOCUMENT") -> list[dict]:
    """
    Takes a list of chunk dicts (each with an "embedding_text" field, as
    produced by loaders.py / chunking.py) and returns the same chunks with
    an added "embedding" field (list[float], or None if it failed/was empty).
    Chunks with empty embedding_text are skipped rather than wasting an
    API call on a meaningless input.
    """
    texts = [c["embedding_text"] for c in chunks]
    non_empty_indices = [i for i, t in enumerate(texts) if t and t.strip()]
    non_empty_texts = [texts[i] for i in non_empty_indices]

    vectors = embed_texts(non_empty_texts, task_type=task_type)

    embedded_chunks = [dict(c, embedding=None) for c in chunks]
    for idx, vec in zip(non_empty_indices, vectors):
        embedded_chunks[idx]["embedding"] = vec

    return embedded_chunks


def embed_documents(docs: list[dict]) -> list[dict]:
    """
    Applies embed_chunks() across every document's chunk list.
    Input: output of chunk_documents() (or load_documents(), works either way).
    Output: same structure, each chunk now has an "embedding" field.
    Always uses task_type="RETRIEVAL_DOCUMENT" — these are chunks going
    INTO the index, not a user query.
    """
    embedded_docs = []
    for doc in docs:
        embedded_docs.append({
            "source": doc["source"],
            "chunks": embed_chunks(doc["chunks"], task_type="RETRIEVAL_DOCUMENT"),
        })
    return embedded_docs


def embed_query(query: str) -> list[float]:
    """
    Embeds a single user search query at retrieval time. Uses
    task_type="RETRIEVAL_QUERY" — deliberately different from
    embed_documents(), since Gemini's embedding space is optimized
    differently for queries vs. the documents they're matched against.
    """
    return embed_one(query, task_type="RETRIEVAL_QUERY")


if __name__ == "__main__":
    import os
    import json
    from src.loaders import load_documents
    from src.chunking import chunk_documents

    output_path = os.path.join("output", "embedded_documents.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    docs = load_documents()
    docs = chunk_documents(docs)
    docs = embed_documents(docs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    total_chunks = sum(len(d["chunks"]) for d in docs)
    embedded_count = sum(1 for d in docs for c in d["chunks"] if c["embedding"] is not None)
    print(f"Loaded {len(docs)} document(s), {total_chunks} chunks, {embedded_count} embedded.")
    print(f"Saved embedded output to '{output_path}'")