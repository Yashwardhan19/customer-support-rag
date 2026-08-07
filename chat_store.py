"""
chat_store.py

Persistent chat history using a CLOUD Postgres database (e.g. Neon,
Supabase, Railway — any standard Postgres works). Same single-user,
no-login design as before, but now the data lives in the cloud instead of
a local SQLite file — so you can browse it from your provider's web
dashboard (e.g. Neon's Tables editor) instead of hunting for a local file,
and it survives even if your machine/deployment changes.

Requires:
    pip install -U psycopg2-binary

Setup:
    1. Create a free Postgres project at https://neon.tech (or Supabase,
       Railway, etc.)
    2. Copy its connection string into .env:
         DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
    3. Make sure src/config.py reads it:
         DATABASE_URL = os.getenv("DATABASE_URL")

Usage — IDENTICAL to the SQLite version, no changes needed elsewhere:
    from src.chat_store import (
        init_db, create_chat, list_chats, get_messages,
        add_message, rename_chat_if_untitled, delete_chat,
    )

    init_db()
    chat_id = create_chat(source_document="myfile.pdf")
    add_message(chat_id, "user", "What is Alice's salary?")
    add_message(chat_id, "assistant", "Alice's salary is 75000.", sources=[...])
    messages = get_messages(chat_id)
    chats = list_chats()   # most recent first
"""

import json
import uuid
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

from src import config

_connection = None


def get_connection():
    """
    Lazy singleton connection to the cloud Postgres database, same pattern
    as get_gemini_client() / get_groq_client() elsewhere in this project.
    Reconnects automatically if the connection has gone stale (cloud DBs
    can drop idle connections after a period of inactivity).
    """
    global _connection
    if _connection is None or _connection.closed:
        if not config.DATABASE_URL:
            raise RuntimeError("DATABASE_URL is not set.")
        _connection = psycopg2.connect(config.DATABASE_URL)
        _connection.autocommit = True
    return _connection


def init_db() -> None:
    """Creates the chats/messages tables if they don't already exist. Safe to call every run."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chats (
                chat_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                source_document TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES chats (chat_id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT,
                created_at TIMESTAMPTZ NOT NULL
            )
        """)


def create_chat(source_document: str = None, title: str = "New chat") -> str:
    """Creates a new chat session and returns its chat_id."""
    chat_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chats (chat_id, title, source_document, created_at, updated_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (chat_id, title, source_document, now, now),
        )
    return chat_id


def list_chats() -> list[dict]:
    """Returns all chats, most recently updated first."""
    conn = get_connection()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT chat_id, title, source_document, created_at, updated_at "
            "FROM chats ORDER BY updated_at DESC"
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def get_messages(chat_id: str) -> list[dict]:
    """Returns all messages for a chat, in order, as {"role", "content", "sources"} dicts."""
    conn = get_connection()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT role, content, sources_json FROM messages "
            "WHERE chat_id = %s ORDER BY id ASC",
            (chat_id,),
        )
        rows = cur.fetchall()
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "sources": json.loads(r["sources_json"]) if r["sources_json"] else [],
        }
        for r in rows
    ]


def add_message(chat_id: str, role: str, content: str, sources: list[dict] = None) -> None:
    """Appends one message to a chat and bumps the chat's updated_at (so it
    sorts to the top of the recent list, like ChatGPT)."""
    now = datetime.now(timezone.utc)
    sources_json = json.dumps(sources) if sources else None
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (chat_id, role, content, sources_json, created_at) "
            "VALUES (%s, %s, %s, %s, %s)",
            (chat_id, role, content, sources_json, now),
        )
        cur.execute("UPDATE chats SET updated_at = %s WHERE chat_id = %s", (now, chat_id))


def rename_chat_if_untitled(chat_id: str, new_title: str, max_len: int = 50) -> None:
    """
    Auto-titles a chat from its first user question (like ChatGPT does),
    but only if it's still the default "New chat" title — never overwrites
    a title the user set some other way later.
    """
    title = new_title.strip().replace("\n", " ")
    if len(title) > max_len:
        title = title[:max_len].rstrip() + "..."

    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("SELECT title FROM chats WHERE chat_id = %s", (chat_id,))
        row = cur.fetchone()
        if row and row[0] == "New chat":
            cur.execute("UPDATE chats SET title = %s WHERE chat_id = %s", (title, chat_id))


def update_chat_source(chat_id: str, source_document: str) -> None:
    """Attaches/updates which document a chat is tied to (used by app.py's process_pdf)."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE chats SET source_document = %s WHERE chat_id = %s",
            (source_document, chat_id),
        )


def delete_chat(chat_id: str) -> None:
    """Deletes a chat and all its messages."""
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM messages WHERE chat_id = %s", (chat_id,))
        cur.execute("DELETE FROM chats WHERE chat_id = %s", (chat_id,))