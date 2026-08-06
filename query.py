"""
QUERY SCRIPT
------------
This script answers a customer's question based on company documents,
using the RAG approach:

1. Embeds the question (using the same model used in ingest.py)
2. Retrieves the top-k most relevant chunks from Qdrant
3. Sends those chunks + the question to the Groq LLM
4. Generates an answer - based ONLY on the retrieved context
5. Displays token counts (RAG vs full-context cost comparison)

"""

import os
from sentence_transformers import SentenceTransformer
# from embeddings import get_embedding
from qdrant_client import QdrantClient
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "support_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"   # free, fast Groq model
TOP_K = 3                             # how many relevant chunks to retrieve

SYSTEM_PROMPT = """You are a customer support assistant for an e-commerce company.
Answer the customer's question using ONLY the CONTEXT provided below.

Rules:
- Use only the information present in CONTEXT. Do not add anything from outside knowledge.
- If the CONTEXT does not contain the answer, clearly say: "I'm sorry, I don't have
  that information. Please contact our support team for further help."
- Always reply in the SAME language and script the customer used in their question.
  - If the customer wrote in English, reply in English.
  - If the customer wrote in Hindi (Devanagari), reply in Hindi (Devanagari).
  - If the customer wrote in Hinglish (Hindi words in Roman/English script), reply in Hinglish the same way.
  - Never switch the customer's language on them.
- Keep answers clear, short, and helpful.
- Never hallucinate or invent information.
- Ignore the context  language , only follow customer  language.
"""


def rough_token_count(text):
    """
    Approximate token estimate (1 token ~ 4 characters, a commonly used
    rough rule for English/Hinglish text). For an exact count you could
    use tiktoken, but this is good enough for demo/comparison purposes.
    """
    return max(1, len(text) // 4)


def retrieve_chunks(client, embed_model, query, top_k=TOP_K):
    """Embed the query and fetch the top-k most relevant chunks from Qdrant."""
    query_vector = embed_model.encode(query).tolist()
    # query_vector = get_embedding(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return results


def build_prompt(query, retrieved_chunks):
    """Combine the retrieved chunks into a single context block."""
    context = "\n\n".join(
        f"[Source: {r.payload['source']}]\n{r.payload['text']}" for r in retrieved_chunks
    )
    user_prompt = f"CONTEXT:\n{context}\n\nCUSTOMER QUESTION: {query}"
    return context, user_prompt


def get_full_docs_token_estimate(client):
    """
    For comparison: estimate how many tokens it WOULD take if we skipped
    RAG entirely and sent the full documentation to the LLM every time.
    """
    all_points, _ = client.scroll(
        collection_name=COLLECTION_NAME, limit=10000, with_payload=True, with_vectors=False
    )
    full_text = "\n\n".join(p.payload["text"] for p in all_points)
    return rough_token_count(full_text)


def ask_groq(groq_client, user_prompt):
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
    )
    answer = response.choices[0].message.content
    # The Groq API also returns actual token usage - prefer this over our estimate (it's accurate)
    usage = response.usage
    return answer, usage


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is not set.")

        return

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Connecting to Qdrant...")
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    if not qdrant_client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' not found. Run ingest.py first.")
        return

    groq_client = Groq(api_key=api_key)

    full_docs_tokens = get_full_docs_token_estimate(qdrant_client)

    print("\nReady! Type your question below (type 'exit' to quit).\n")

    while True:
        query = input("Customer: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        # Step 1: Retrieve relevant chunks
        retrieved = retrieve_chunks(qdrant_client,embed_model, query)

        if not retrieved:
            print("Bot: No relevant information found.\n")
            continue

        # Step 2: Build prompt with only retrieved context
        context, user_prompt = build_prompt(query, retrieved)

        # Step 3: Ask Groq LLM
        answer, usage = ask_groq(groq_client, user_prompt)

        print(f"\nBot: {answer}\n")

        # Step 4: Show which chunks were used (transparency)
        print("--- Retrieved sources ---")
        for r in retrieved:
            print(f"  - {r.payload['source']} (score: {r.score:.3f})")

        # Step 5: Cost comparison (RAG vs full-context)
        rag_input_tokens = usage.prompt_tokens
        savings_pct = round((1 - rag_input_tokens / full_docs_tokens) * 100, 1) if full_docs_tokens else 0

        print("\n--- Token / cost comparison ---")
        print(f"  Without RAG (sending full documentation): ~{full_docs_tokens} input tokens")
        print(f"  With RAG (only top-{TOP_K} chunks):          {rag_input_tokens} input tokens")
        print(f"  Token savings: ~{savings_pct}%")
        print(f"  Output tokens: {usage.completion_tokens} | Total: {usage.total_tokens}")
        print("=" * 60 + "\n")

    qdrant_client.close()


if __name__ == "__main__":
    main()