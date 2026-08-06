import os
import streamlit as st
from sentence_transformers import SentenceTransformer
# from embeddings import get_embedding
from qdrant_client import QdrantClient
from groq import Groq
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

QDRANT_PATH = "qdrant_data"
COLLECTION_NAME = "support_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 3

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
"""

st.set_page_config(page_title="Customer Support Assistant", page_icon="💬", layout="centered")


# ---------- Cached resources (loaded once, not on every rerun) ----------
@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def load_qdrant_client():
    return QdrantClient(path=QDRANT_PATH)


@st.cache_resource
def load_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


@st.cache_data
def rough_token_count(text):
    return max(1, len(text) // 4)


@st.cache_data
def get_full_docs_token_estimate(_client):
    all_points, _ = _client.scroll(
        collection_name=COLLECTION_NAME, limit=10000, with_payload=True, with_vectors=False
    )
    full_text = "\n\n".join(p.payload["text"] for p in all_points)
    return rough_token_count(full_text)


def retrieve_chunks(client,embed_model, query, top_k=TOP_K):
    query_vector = embed_model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points
    return results


def build_prompt(query, retrieved_chunks):
    context = "\n\n".join(
        f"[Source: {r.payload['source']}]\n{r.payload['text']}" for r in retrieved_chunks
    )
    user_prompt = f"CONTEXT:\n{context}\n\nCUSTOMER QUESTION: {query}"
    return user_prompt


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
    return response.choices[0].message.content, response.usage


# ---------- UI ----------
st.title("💬 Customer Support Assistant")
st.caption("RAG-powered • Qdrant + Groq (Llama 3.1) + sentence-transformers")

# Sidebar: setup status + info
with st.sidebar:
    st.header("Setup Status")

    api_key_present = bool(os.environ.get("GROQ_API_KEY"))
    st.write("✅ GROQ_API_KEY set" if api_key_present else "❌ GROQ_API_KEY missing")

    try:
        qdrant_client = load_qdrant_client()
        collection_exists = qdrant_client.collection_exists(COLLECTION_NAME)
        st.write("✅ Qdrant collection found" if collection_exists else "❌ Collection not found — run ingest.py")
    except Exception as e:
        collection_exists = False
        st.write(f"❌ Qdrant error: {e}")

    st.divider()
    st.header("How it works")
    st.markdown(
        "1. Your question is embedded\n"
        "2. Top-3 relevant chunks retrieved from Qdrant\n"
        "3. Only those chunks + your question go to Groq LLM\n"
        "4. Answer is grounded in company documents only"
    )

if not api_key_present:
    st.error("GROQ_API_KEY environment variable is not set. Set it in your terminal and rerun Streamlit.")
    st.stop()

if not collection_exists:
    st.error(f"Collection '{COLLECTION_NAME}' not found. Run `python ingest.py` first.")
    st.stop()

embed_model = load_embed_model()
groq_client = load_groq_client()
full_docs_tokens = get_full_docs_token_estimate(qdrant_client)

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            with st.expander("📎 Sources & cost details"):
                for src in meta["sources"]:
                    st.write(f"- **{src['source']}** (relevance score: {src['score']:.3f})")
                st.divider()
                c1, c2, c3 = st.columns(3)
                c1.metric("Without RAG (tokens)", meta["full_tokens"])
                c2.metric("With RAG (tokens)", meta["rag_tokens"])
                c3.metric("Savings", f"{meta['savings_pct']}%")

# Chat input
user_query = st.chat_input("Ask a question... (e.g. what is the return policy?)")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            retrieved = retrieve_chunks(qdrant_client,embed_model  ,user_query)

            if not retrieved:
                answer = "Sorry, no relevant information was found."
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                user_prompt = build_prompt(user_query, retrieved)
                answer, usage = ask_groq(groq_client, user_prompt)
                st.write(answer)

                rag_tokens = usage.prompt_tokens
                savings_pct = round((1 - rag_tokens / full_docs_tokens) * 100, 1) if full_docs_tokens else 0

                meta = {
                    "sources": [{"source": r.payload["source"], "score": r.score} for r in retrieved],
                    "full_tokens": full_docs_tokens,
                    "rag_tokens": rag_tokens,
                    "savings_pct": savings_pct,
                }

                with st.expander("📎 Sources & cost details"):
                    for src in meta["sources"]:
                        st.write(f"- **{src['source']}** (relevance score: {src['score']:.3f})")
                    st.divider()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Without RAG (tokens)", meta["full_tokens"])
                    c2.metric("With RAG (tokens)", meta["rag_tokens"])
                    c3.metric("Savings", f"{meta['savings_pct']}%")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "meta": meta}
                )