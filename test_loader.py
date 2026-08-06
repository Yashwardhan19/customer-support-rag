from src.loaders import load_documents

docs = load_documents()

print(f"Total documents loaded: {len(docs)}")
print("=" * 80)

for doc in docs:
    print(f"Source: {doc['source']}")
    print("-" * 80)
    for i, chunk in enumerate(doc["chunks"], start=1):
        print(f"[Chunk {i}] page={chunk['page']} type={chunk['type']}")
        print(f"  embedding_text: {chunk['embedding_text'][:300]}")
        print(f"  display_text  : {chunk['display_text'][:300]}")
        print()
    print("=" * 80)