"""
VIEW EMBEDDINGS SCRIPT
------------------------
Shows the chunks stored in the Qdrant collection along with their
embeddings (vectors).

Each embedding is an array of 384 numbers (since the all-MiniLM-L6-v2
model produces 384-dimension vectors) - printing the full vector isn't
very readable, so by default this script shows:
  - The first 8 numbers (preview)
  - The vector's length (384)
  - The vector's norm/magnitude (as a sanity check)

If you want the FULL vector (all 384 numbers), set FULL_VECTOR = True below.

Run: python view_embeddings.py
"""

from qdrant_client import QdrantClient

QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "support_docs"

FULL_VECTOR = False   # set to True to print all 384 numbers
PREVIEW_COUNT = 8     # how many numbers to show in the preview
MAX_CHUNKS_TO_SHOW = 5  # how many chunks to display (set to None to show all)


def main():
    client = QdrantClient(path=QDRANT_PATH)

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' not found. Run ingest.py first.")
        client.close()
        return

    info = client.get_collection(COLLECTION_NAME)
    print(f"Total chunks stored: {info.points_count}")
    print(f"Embedding dimension: {info.config.params.vectors.size}")
    print(f"Distance metric: {info.config.params.vectors.distance}")
    print("=" * 70)

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=MAX_CHUNKS_TO_SHOW if MAX_CHUNKS_TO_SHOW else 10000,
        with_payload=True,
        with_vectors=True,   # we need the vectors here too
    )

    for i, point in enumerate(points, start=1):
        source = point.payload.get("source", "unknown")
        text = point.payload.get("text", "")
        vector = point.vector  # list of 384 floats

        # Vector magnitude (length) - normalized embeddings should have a norm close to 1.0
        norm = sum(v ** 2 for v in vector) ** 0.5

        print(f"\n[Chunk #{i}] source: {source}")
        print(f"Text preview: {text[:100]}...")
        print(f"Vector length: {len(vector)} dimensions")
        print(f"Vector norm: {norm:.4f}")

        if FULL_VECTOR:
            print(f"Full vector:\n{vector}")
        else:
            preview = [round(v, 4) for v in vector[:PREVIEW_COUNT]]
            print(f"First {PREVIEW_COUNT} values: {preview} ...")

        print("-" * 70)

    client.close()


if __name__ == "__main__":
    main()