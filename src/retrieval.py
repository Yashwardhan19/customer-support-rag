"""
retrieval.py

The query-time half of the RAG pipeline:
    user question
       -> embed_query()                  [Gemini, task_type=RETRIEVAL_QUERY]
       -> search() in Qdrant             [cosine similarity]
       -> assemble context from display_text of top-k chunks
       -> Groq (openai/gpt-oss-120b) generates the final answer

Requires:
    pip install -U groq

Setup:
    Add to .env:
        GROQ_API_KEY=your-groq-api-key-here

Usage:
    from src.retrieval import answer_question

    result = answer_question("What is Alice's salary?")
    print(result["answer"])
    print(result["sources"])
"""

import re

from groq import Groq

from src import config
from src.embedding import embed_query
from src.vector_store import search

_groq_client = None

# Small talk / greetings that aren't real questions about the document —
# handled directly, without running retrieval at all. Keeps the RAG
# pipeline focused on actual document questions and avoids the LLM
# awkwardly trying to "answer" a greeting from document context.
_SMALLTALK_PATTERNS = [
    r"^\s*(hi|hello|hey|yo|hiya)[\s!.]*$",
    r"^\s*(good\s?(morning|afternoon|evening))[\s!.]*$",
    r"^\s*(how are you|how's it going|what's up|sup)[\s?!.]*$",
    r"^\s*(thanks|thank you|thx|ty)[\s!.]*$",
    r"^\s*(bye|goodbye|see ya|see you|cya)[\s!.]*$",
    r"^\s*(ok|okay|cool|nice|great|got it)[\s!.]*$",
]
_SMALLTALK_RE = re.compile("|".join(_SMALLTALK_PATTERNS), re.IGNORECASE)

_SMALLTALK_REPLIES = {
    "greeting": "Hi there! Ask me anything about the document you uploaded and I'll do my best to help.",
    "thanks": "You're welcome! Let me know if you have more questions about the document.",
    "bye": "Goodbye! Come back anytime you have more questions about the document.",
    "ack": "Got it — let me know what you'd like to know about the document.",
}


def classify_smalltalk(question: str) -> str | None:
    """Returns a smalltalk category if the question is a greeting/chit-chat,
    not an actual document question. Returns None otherwise."""
    stripped = question.strip().lower()
    if not _SMALLTALK_RE.match(stripped):
        return None
    if any(w in stripped for w in ["thank", "thx", "ty"]):
        return "thanks"
    if any(w in stripped for w in ["bye", "goodbye", "cya", "see ya", "see you"]):
        return "bye"
    if any(w in stripped for w in ["ok", "okay", "cool", "nice", "great", "got it"]):
        return "ack"
    return "greeting"


def get_groq_client() -> Groq:
    """Lazy singleton for the Groq client, same pattern as get_gemini_client()."""
    global _groq_client
    if _groq_client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set.")
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def rewrite_standalone_question(question: str, chat_history: list[dict]) -> str:
    """
    Rewrites a follow-up question into a standalone one using prior chat
    turns, so retrieval (which has no memory of its own) can search for
    the right thing. E.g. "who is the richest man" -> "where does he
    live" becomes "where does the richest man live".
    Returns the original question unchanged if there's no history yet,
    or if the rewrite fails for any reason (fail-open, not fail-closed).
    """
    if not chat_history:
        return question

    recent = chat_history[-config.HISTORY_TURNS * 2:]
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    prompt = (
        "Given the conversation history and a follow-up question, rewrite "
        "the follow-up question into a standalone question that includes "
        "all necessary context from the history (e.g. resolve pronouns like "
        "'he', 'it', 'that' into what they actually refer to). "
        "If the follow-up question is already standalone, return it unchanged. "
        "Output ONLY the rewritten question, nothing else — no explanation, "
        "no quotes.\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"Follow-up question: {question}\n\n"
        "Standalone question:"
    )

    try:
        client = get_groq_client()
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        rewritten = completion.choices[0].message.content.strip()
        return rewritten if rewritten else question
    except Exception:
        return question  # fail-open: worst case, behaves like no history was used


def build_context(chunks: list[dict]) -> str:
    """
    Assembles retrieved chunks into a single context string for the LLM
    prompt. Uses display_text (NOT embedding_text) — display_text is the
    richer, LLM-facing version (e.g. a full table even if only one row
    matched the search), while embedding_text is the leaner version that
    was only meant for the vector search itself.
    Each chunk is labeled with its source and page so the LLM can cite them.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        payload = chunk["payload"]
        source = payload.get("source", "unknown")
        page = payload.get("page")
        page_label = f", page {page}" if page is not None else ""
        parts.append(f"[Source {i}: {source}{page_label}]\n{payload.get('display_text', '')}")
    return "\n\n---\n\n".join(parts)


def strip_stray_citation_markers(text: str) -> str:
    """
    Defense-in-depth cleanup: gpt-oss models occasionally reproduce
    OpenAI's internal file-citation format (e.g. 【1†L7-L9】) out of training
    habit even when explicitly told not to. The system prompt asks it not
    to, but LLMs don't always perfectly follow negative instructions, so
    this strips the pattern if it slips through anyway.
    """
    return re.sub(r"【[^】]*】", "", text).strip()


def generate_answer(question: str, context: str) -> str:
    """Calls Groq's gpt-oss-120b to generate an answer grounded in the retrieved context."""
    client = get_groq_client()

    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the "
        "provided context. Do not use outside knowledge, even if you know the "
        "answer. If the context doesn't contain enough information to answer "
        "the question, clearly say the document doesn't cover that — do not "
        "guess or fill gaps with general knowledge.\n\n"
        "When you reference a source, use the format [Source N] (e.g. "
        "'[Source 1]') matching the source numbers given in the context below. "
        "Do NOT use any other citation format — specifically, never use "
        "bracketed reference markers like 【1†L1-L2】 or similar file-citation "
        "syntax; that format is not valid here and should never appear in "
        "your answer. Plain [Source N] only, or no citation at all if it's "
        "not needed."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    completion = client.chat.completions.create(
        model=config.GROQ_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.GROQ_TEMPERATURE,
        reasoning_effort=config.GROQ_REASONING_EFFORT,  # "low" | "medium" | "high"
    )
    return strip_stray_citation_markers(completion.choices[0].message.content)


def answer_question(question: str, chat_history: list[dict] = None, top_k: int = None,
                     filter_conditions: dict = None) -> dict:
    """
    Full RAG query: embed the question, search Qdrant, generate a grounded
    answer with Groq. Returns {"answer": str, "sources": list[dict], "chunks_used": int}.

    chat_history: list of {"role": "user"/"assistant", "content": str} from
    prior turns in the SAME conversation. Used to rewrite follow-up
    questions (e.g. "where does he live") into standalone ones (e.g.
    "where does the richest man live") BEFORE retrieval — otherwise the
    vector search has no way to know who "he" refers to. Pass None or []
    for a fresh conversation with no prior context.

    Out-of-scope handling (two layers):
      1. Hard guard: if the BEST retrieved score is below
         config.MIN_RELEVANCE_SCORE, the question is treated as unrelated
         to the document and answered directly WITHOUT calling the LLM —
         avoids the model stretching a weak/irrelevant match into a
         made-up answer.
      2. Soft guard: even when relevant chunks ARE found, the system
         prompt still instructs Groq to say so if the context doesn't
         actually contain the answer (a chunk can be topically similar
         without containing the specific fact asked about).

    Small talk (greetings, thanks, "ok", etc.) is detected BEFORE retrieval
    and answered directly — it's not a document question, so running it
    through search/relevance scoring produces confusing results.
    """
    smalltalk = classify_smalltalk(question)
    if smalltalk:
        return {
            "answer": _SMALLTALK_REPLIES[smalltalk],
            "sources": [],
            "chunks_used": 0,
            "in_scope": None,   # not applicable — this wasn't a document question at all
            "top_score": None,
            "standalone_question": question,
        }

    top_k = top_k or config.RETRIEVAL_TOP_K
    standalone_question = rewrite_standalone_question(question, chat_history or [])

    query_vector = embed_query(standalone_question)
    results = search(query_vector, top_k=top_k, filter_conditions=filter_conditions)

    if not results:
        return {
            "answer": "I couldn't find anything in this document related to that question.",
            "sources": [],
            "chunks_used": 0,
            "in_scope": False,
            "top_score": None,
            "standalone_question": standalone_question,
        }

    best_score = results[0]["score"]
    if best_score < config.MIN_RELEVANCE_SCORE:
        return {
            "answer": (
                "That doesn't seem to be covered in this document. "
                "I can only answer questions based on the content you've uploaded — "
                "try rephrasing, or ask something else about the document."
            ),
            "sources": [],          # nothing shown to the user — query was out of scope
            "chunks_used": 0,
            "in_scope": False,
            "top_score": best_score,  # kept for debug/calibration only, not shown as a "source"
            "standalone_question": standalone_question,
        }

    context = build_context(results)
    answer = generate_answer(standalone_question, context)

    sources = [
        {
            "source": r["payload"].get("source"),
            "page": r["payload"].get("page"),
            "type": r["payload"].get("type"),
            "score": r["score"],
        }
        for r in results
    ]

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(results),
        "in_scope": True,
        "top_score": best_score,
        "standalone_question": standalone_question,
    }


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is this document about?"
    result = answer_question(question)

    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['source']} (page {s['page']}, type={s['type']}, score={s['score']:.3f})")