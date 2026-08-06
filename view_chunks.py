"""
VIEW CHUNKS SCRIPT
-------------------
Prints all chunks stored in the Qdrant collection, so you can verify
that ingestion worked correctly.

Run: python view_chunks.py
"""

from qdrant_client import QdrantClient

QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "support_docs"


def main():
    client = QdrantClient(path=QDRANT_PATH)

    if not client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' not found. Run ingest.py first.")
        client.close()
        return

    info = client.get_collection(COLLECTION_NAME)
    total_points = info.points_count
    print(f"Total chunks stored: {total_points}\n")
    print("=" * 70)

    # scroll() fetches all points in batches (we don't need the vectors, just the payload)
    offset = None
    count = 0
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=50,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            count += 1
            source = point.payload.get("source", "unknown")
            text = point.payload.get("text", "")
            print(f"\n[Chunk #{count}] (source: {source})")
            print("-" * 70)
            print(text)
            print("-" * 70)

        if offset is None:
            break

    print(f"\n{'=' * 70}")
    print(f"Total chunks displayed: {count}")
    client.close()


if __name__ == "__main__":
    main()