"""
app.py

Streamlit UI for the RAG pipeline, with ChatGPT-style persistent chat
sessions (single-user, no login — see chat_store.py):
    1. User uploads a PDF or image, indexed via the pipeline (loaders ->
       chunking -> embeddings -> vector_store).
    2. Each chat session is tied to one document and stored in SQLite, so
       chat history survives app restarts / new browser tabs.
    3. Sidebar shows recent chats (most recently active first), a
       "New chat" button, and lets you switch between past conversations.

Requires (on top of everything already installed for the pipeline):
    pip install -U streamlit

Run:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from src.loaders import read_pdf, read_image_file
from src.chunking import chunk_documents
from src.embedding import embed_documents
from src.vector_store import upsert_documents
from src.retrieval import answer_question
from src import config
from chat_store import (
    init_db, create_chat, list_chats, get_messages,
    add_message, rename_chat_if_untitled, delete_chat, update_chat_source,
)

init_db()

st.set_page_config(page_title="Chat with your PDF", page_icon="📄", layout="wide")
st.title("Helper.ai")

# File types accepted by the uploader / process_upload dispatch below.
IMAGE_EXTS = ("png", "jpg", "jpeg")

# --- Session state setup ---
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()   # filenames indexed into Qdrant this session
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []             # loaded from DB for the active chat


def load_chat(chat_id: str) -> None:
    """Switches the active chat and loads its messages from SQLite."""
    st.session_state.active_chat_id = chat_id
    st.session_state.messages = get_messages(chat_id)


def start_new_chat() -> None:
    """Creates a fresh, empty chat with no document attached yet."""
    chat_id = create_chat(source_document=None)
    load_chat(chat_id)


def process_upload(uploaded_file) -> bool:
    """
    Save the upload to disk, run it through the full pipeline (routing to
    the PDF or image loader depending on file type), upsert into Qdrant,
    and attach it as the active chat's document. Returns True on success,
    False on failure (with an error shown to the user).
    """
    filename = uploaded_file.name
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, filename)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Extracting '{filename}' (text, tables, charts)..."):
                if ext == "pdf":
                    pages = read_pdf(tmp_path)
                    flat_chunks = [
                        {"page": p["page"], **c} for p in pages for c in p["chunks"]
                    ]
                elif ext in IMAGE_EXTS:
                    # read_image_file() already returns a flat chunk list
                    # (single vision chunk, page=None) — no flattening needed.
                    flat_chunks = read_image_file(tmp_path)
                else:
                    st.error(f"Unsupported file type: .{ext or '?'}")
                    return False

                docs = [{"source": filename, "chunks": flat_chunks}]

            with st.spinner("Chunking..."):
                docs = chunk_documents(docs)

            with st.spinner("Generating embeddings..."):
                docs = embed_documents(docs)

            with st.spinner("Indexing into Qdrant..."):
                count = upsert_documents(docs)

    except Exception as e:
        st.error(f"Failed to process '{filename}': {e}")
        with st.expander("Error details"):
            st.exception(e)
        return False

    st.session_state.processed_files.add(filename)

    # Attach this document to the CURRENT chat (create one if none is active yet)
    if st.session_state.active_chat_id is None:
        chat_id = create_chat(source_document=filename, title="New chat")
        load_chat(chat_id)
    else:
        update_chat_source(st.session_state.active_chat_id, filename)

    st.success(f"Indexed '{filename}' — {count} chunks stored.")
    return True


def get_active_source() -> str | None:
    """Looks up which document the active chat is tied to."""
    for c in list_chats():
        if c["chat_id"] == st.session_state.active_chat_id:
            return c["source_document"]
    return None


# --- Ensure there's always an active chat ---
if st.session_state.active_chat_id is None:
    existing = list_chats()
    if existing:
        load_chat(existing[0]["chat_id"])
    else:
        start_new_chat()


# --- Sidebar: new chat, recent chats, upload ---
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.divider()
    st.subheader("Recent chats")
    for c in list_chats():
        label = c["title"]
        is_active = c["chat_id"] == st.session_state.active_chat_id
        cols = st.columns([5, 1])
        with cols[0]:
            if st.button(
                ("**" + label + "**") if is_active else label,
                key=f"chat_{c['chat_id']}",
                use_container_width=True,
            ):
                load_chat(c["chat_id"])
                st.rerun()
        with cols[1]:
            if st.button("🗑", key=f"del_{c['chat_id']}"):
                delete_chat(c["chat_id"])
                if is_active:
                    st.session_state.active_chat_id = None
                st.rerun()

    st.divider()
    st.header("Upload a document")
    active_source = get_active_source()
    if active_source:
        st.caption(f"This chat's document: **{active_source}**")

    uploaded_file = st.file_uploader(
        "Choose a PDF or image",
        type=["pdf", "png", "jpg", "jpeg"],
    )
    if uploaded_file is not None:
        already_done = uploaded_file.name in st.session_state.processed_files
        label = "Re-process anyway" if already_done else "Process & Index File"
        if st.button(label, type="primary"):
            process_upload(uploaded_file)
            st.rerun()

    st.divider()
    st.session_state.debug_mode = st.checkbox("Show relevance scores (debug)", value=False)


# --- Main: chat interface ---
active_source = get_active_source()

if not active_source:
    st.info("Upload a PDF or image in the sidebar to start chatting in this session.")
else:
    st.caption(f"Chatting with: **{active_source}**")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.caption(f"- {s['source']} · page {s['page']} · {s['type']} · score {s['score']:.3f}")

    question = st.chat_input("Ask a question about this document...")

    if question:
        chat_id = st.session_state.active_chat_id

        st.session_state.messages.append({"role": "user", "content": question, "sources": []})
        add_message(chat_id, "user", question)
        rename_chat_if_untitled(chat_id, question)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching and generating answer..."):
                    history_for_llm = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    result = answer_question(
                        question,
                        chat_history=history_for_llm,
                        filter_conditions={"source": active_source},
                    )
            except Exception as e:
                st.error(f"Something went wrong while answering: {e}")
                with st.expander("Error details"):
                    st.exception(e)
                result = None

            if result is not None:
                st.markdown(result["answer"])
                if result.get("in_scope") and result["sources"]:
                    with st.expander("Sources"):
                        for s in result["sources"]:
                            st.caption(f"- {s['source']} · page {s['page']} · {s['type']} · score {s['score']:.3f}")

                if st.session_state.get("debug_mode") and result.get("top_score") is not None:
                    st.caption(f"🔧 debug: top score = {result['top_score']:.3f} (threshold = {config.MIN_RELEVANCE_SCORE}) · in_scope = {result.get('in_scope')}")
                if st.session_state.get("debug_mode") and result.get("standalone_question") and result["standalone_question"] != question:
                    st.caption(f"🔧 debug: rewritten as → \"{result['standalone_question']}\"")

        if result is not None:
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })
            add_message(chat_id, "assistant", result["answer"], sources=result["sources"])
            st.rerun()  # refresh sidebar so this chat moves to top of "recent"