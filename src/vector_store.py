"""
vector_store.py

Stores embedded chunks (output of embed_documents()) into a Qdrant Cloud
collection, and provides a search() helper for retrieval at query time.

Requires:
    pip install -U qdrant-client

Setup:
    1. Create a free cluster at https://cloud.qdrant.io
    2. Copy its URL and API key into your .env:
         QDRANT_URL=https://xxxxxxxx.qdrant.io
         QDRANT_API_KEY=xxxxxxxxxxxxxxxx
    3. Make sure src/config.py reads them (see config additions below).

Usage:
    from src.loaders import load_documents
    from src.chunking import chunk_documents
    from src.embeddings import embed_documents, embed_query
    from src.vector_store import upsert_documents, search

    docs = load_documents()
    docs = chunk_documents(docs)
    docs = embed_documents(docs)
    upsert_documents(docs)

    # later, at query time:
    query_vector = embed_query("what is Alice's salary?")
    results = search(query_vector, top_k=5)
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from src import config

_qdrant_client = None


def get_qdrant_client() -> QdrantClient:
    """Lazy singleton for the Qdrant Cloud client, same pattern as
    get_gemini_client() in loaders.py."""
    global _qdrant_client
    if _qdrant_client is None:
        if not config.QDRANT_URL or not config.QDRANT_API_KEY:
            raise RuntimeError("QDRANT_URL / QDRANT_API_KEY are not set.")
        _qdrant_client = QdrantClient(
            url=config.QDRANT_URL,
            api_key=config.QDRANT_API_KEY,
        )
    return _qdrant_client


def ensure_payload_index(client: QdrantClient, field_name: str, field_schema: str = "keyword") -> None:
    """
    Qdrant Cloud requires an explicit payload index before you can filter
    on a field (unlike the vector itself, payload fields aren't indexed
    automatically). Safe to call every run — checks the collection's
    existing indexed fields first and skips if already present, so this
    never errors on repeated calls.
    """
    info = client.get_collection(config.QDRANT_COLLECTION_NAME)
    existing_indexes = info.payload_schema or {}
    if field_name in existing_indexes:
        return

    client.create_payload_index(
        collection_name=config.QDRANT_COLLECTION_NAME,
        field_name=field_name,
        field_schema=field_schema,
    )
    print(f"      -> Created payload index on '{field_name}' ({field_schema})")


def ensure_collection(vector_size: int = None) -> None:
    """
    Creates the collection if it doesn't already exist, and ensures the
    payload indexes needed for filtering (source, type) are present.
    Safe to call every run.
    vector_size must match the dimensionality your embedding model actually
    outputs (config.EMBEDDING_OUTPUT_DIM for gemini-embedding-001).
    """
    client = get_qdrant_client()
    vector_size = vector_size or config.EMBEDDING_OUTPUT_DIM

    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"      -> Created Qdrant collection '{config.QDRANT_COLLECTION_NAME}' (dim={vector_size})")

    # Needed for filter_conditions={"source": ...} / {"type": ...} in search()
    ensure_payload_index(client, "source")
    ensure_payload_index(client, "type")


def chunk_to_point(source: str, chunk: dict) -> PointStruct | None:
    """
    Converts one embedded chunk dict into a Qdrant PointStruct.
    The vector is the chunk's embedding; everything else (embedding_text,
    display_text, page, type, table_name, heading, source, etc.) becomes
    payload metadata, so you can filter/display results without a second
    lookup. Skips chunks with no embedding (e.g. empty text that was
    intentionally not embedded).
    """
    if not chunk.get("embedding"):
        return None

    payload = {k: v for k, v in chunk.items() if k != "embedding"}
    payload["source"] = source

    return PointStruct(
        id=str(uuid.uuid4()),
        vector=chunk["embedding"],
        payload=payload,
    )


def upsert_documents(docs: list[dict], batch_size: int = None) -> int:
    """
    Upserts all embedded chunks across all documents into Qdrant.
    Input: output of embed_documents() — [{"source": ..., "chunks": [...]}].
    Returns the number of points actually upserted (chunks without an
    embedding are skipped and don't count).
    """
    ensure_collection()
    client = get_qdrant_client()
    batch_size = batch_size or config.QDRANT_UPSERT_BATCH_SIZE

    points = []
    for doc in docs:
        for chunk in doc["chunks"]:
            point = chunk_to_point(doc["source"], chunk)
            if point is not None:
                points.append(point)

    total = 0
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=config.QDRANT_COLLECTION_NAME, points=batch)
        total += len(batch)
        print(f"      -> Upserted {total}/{len(points)} points to Qdrant...")

    return total


def search(query_vector: list[float], top_k: int = 5, filter_conditions: dict = None):
    """
    Searches the collection for the top_k most similar chunks to a query
    vector (use embed_query() from embeddings.py to produce it — NOT
    embed_texts/embed_one with RETRIEVAL_DOCUMENT task_type, since queries
    and documents are embedded differently for best retrieval quality).

    filter_conditions: optional dict for simple payload filtering, e.g.
        {"type": "table_row"} or {"source": "myfile.pdf"}

    Returns a list of {"score": float, "payload": dict}.
    """
    client = get_qdrant_client()
    ensure_collection()  # guarantees payload indexes exist even if search() is
                          # called before any upsert in this process (e.g. app restart)

    query_filter = None
    if filter_conditions:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]
        )

    results = client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
    ).points

    return [{"score": r.score, "payload": r.payload} for r in results]


if __name__ == "__main__":
    from src.loaders import load_documents
    from src.chunking import chunk_documents
    from src.embedding import embed_documents

    docs = load_documents()
    docs = chunk_documents(docs)
    docs = embed_documents(docs)

    count = upsert_documents(docs)
    print(f"Done. Upserted {count} chunks into Qdrant collection '{config.QDRANT_COLLECTION_NAME}'.")