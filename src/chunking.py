"""
chunking.py

Takes the flat chunk list produced by src.loaders (load_documents / read_pdf)
and applies proper chunking on top of it:

  - type == "text"   -> structure-aware + recursive chunking (headings,
                         paragraphs, then recursive character splitting with
                         overlap, as a fallback for anything still too large)
  - everything else  -> passed through unchanged. table_row, chart_description,
                         and vision chunks are already the right granularity
                         (one row / one visual description / one full-page
                         transcription) and re-splitting them would break
                         their structure.

Usage:
    from src.loaders import load_documents
    from src.chunking import chunk_documents

    docs = load_documents()
    docs = chunk_documents(docs)   # docs[i]["chunks"] now has text split up
"""

import re
from src import config


HEADING_PATTERN = re.compile(
    r"^(#{1,6}\s+.+|[A-Z][A-Za-z0-9 ,&/-]{2,80})$"
)


def is_heading_line(line: str) -> bool:
    """
    Heuristic heading detector for plain-text page output (no real markdown
    guaranteed). A line counts as a heading if it's short, has no trailing
    sentence punctuation, and is either markdown-style ("# ...") or
    title/sentence-case without a period at the end.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.endswith((".", ",", ";", ":")):
        return False
    # Short, capitalized, no terminal punctuation -> likely a heading/title
    words = stripped.split()
    if 1 <= len(words) <= 12 and stripped[0].isupper():
        return True
    return False


def split_by_structure(text: str) -> list[dict]:
    """
    First pass: break text into sections using headings as boundaries.
    Returns a list of {"heading": str | None, "body": str} sections, in order.
    Text before the first detected heading gets heading=None.
    """
    lines = text.split("\n")
    sections = []
    current_heading = None
    current_body_lines = []

    def flush():
        body = "\n".join(current_body_lines).strip()
        if body or current_heading:
            sections.append({"heading": current_heading, "body": body})

    for line in lines:
        if is_heading_line(line):
            flush()
            current_heading = line.strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)
    flush()

    return sections if sections else [{"heading": None, "body": text.strip()}]


def recursive_split(text: str, max_chars: int, overlap: int, separators=None) -> list[str]:
    """
    Recursive character-based splitter, LangChain-style. Tries the biggest
    separator first (paragraph breaks), and only falls back to smaller
    separators for pieces that are still too large after splitting on the
    current one. Keeps `overlap` characters of trailing context between
    consecutive chunks so retrieval doesn't lose meaning at a hard cut.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= max_chars:
        return [text] if text.strip() else []

    sep = separators[0]
    remaining_separators = separators[1:]

    if sep == "":
        # Last resort: hard character split
        pieces = [text[i:i + max_chars] for i in range(0, len(text), max_chars - overlap)]
        return [p for p in pieces if p.strip()]

    parts = text.split(sep)
    chunks = []
    current = ""

    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > max_chars:
                # This single part is still too big -> recurse with a smaller separator
                if remaining_separators:
                    chunks.extend(recursive_split(part, max_chars, overlap, remaining_separators))
                    current = ""
                else:
                    current = part
            else:
                current = part

    if current.strip():
        chunks.append(current)

    # Apply overlap by prepending trailing chars of the previous chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap:]
            overlapped.append((prev_tail + " " + chunks[i]).strip())
        chunks = overlapped

    return chunks


def chunk_text_entry(chunk: dict, max_chars: int = None, overlap: int = None) -> list[dict]:
    """
    Takes ONE "text"-type chunk (a whole page's clean text) and returns a
    list of smaller chunks: heading-aware sections first, then recursively
    split further if a section is still bigger than max_chars.
    Every output chunk keeps the original chunk's "page" and gets a fresh
    "type": "text_chunk" plus a "heading" field (may be None) and a
    "chunk_index" for ordering/debugging.
    """
    max_chars = max_chars or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP

    sections = split_by_structure(chunk["embedding_text"])
    results = []
    idx = 0

    for section in sections:
        heading = section["heading"]
        body = section["body"]
        if not body:
            continue

        pieces = recursive_split(body, max_chars=max_chars, overlap=overlap)
        for piece in pieces:
            # Prefix the heading into both fields so retrieval keeps section context
            text_with_heading = f"{heading}\n{piece}" if heading else piece
            results.append({
                "page": chunk.get("page"),
                "type": "text_chunk",
                "heading": heading,
                "chunk_index": idx,
                "embedding_text": text_with_heading,
                "display_text": text_with_heading,
            })
            idx += 1

    return results if results else [chunk]  # fallback: never drop content


def chunk_documents(docs: list[dict], max_chars: int = None, overlap: int = None) -> list[dict]:
    """
    Applies chunking across all loaded documents (output of load_documents()).
    "text" chunks get structure-aware + recursive splitting; every other
    chunk type (table_row, chart_description, vision) passes through
    unchanged, since those are already the right granularity.
    """
    chunked_docs = []
    for doc in docs:
        new_chunks = []
        for chunk in doc["chunks"]:
            if chunk["type"] == "text":
                new_chunks.extend(chunk_text_entry(chunk, max_chars=max_chars, overlap=overlap))
            else:
                new_chunks.append(chunk)
        chunked_docs.append({"source": doc["source"], "chunks": new_chunks})
    return chunked_docs


if __name__ == "__main__":
    import os
    import json
    from src.loaders import load_documents

    output_path = os.path.join("output", "chunked_documents.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    docs = load_documents()
    chunked_docs = chunk_documents(docs)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)

    total_chunks = sum(len(d["chunks"]) for d in chunked_docs)
    print(f"Loaded {len(chunked_docs)} document(s), {total_chunks} total chunks after chunking.")
    print(f"Saved chunked output to '{output_path}'")