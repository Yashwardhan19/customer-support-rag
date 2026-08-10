import os
import io
import glob
import json

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image
from src import config
from google import genai
from google.genai import types

_gemini_client = None


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def vision_extract(image: Image.Image, prompt: str) -> str:
    """
    Sends an image to Gemini with the given prompt.
    Caller chooses the prompt:
      - config.VISION_PROMPT_CHART_ONLY  -> page already has text/tables extracted;
        vision should ONLY describe charts/graphs/diagrams.
      - config.VISION_PROMPT_FULL_PAGE   -> no separate text layer exists (scanned
        page or standalone image); vision must transcribe everything.
    """
    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=[prompt, image],
        )
        return (response.text or "").strip()
    except Exception as e:
        print(f"      -> Warning: Gemini vision extraction failed: {e}")
        return ""


# Plain Text

def read_txt(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return [{"page": None, "type": "text", "embedding_text": content, "display_text": content}]


# Standalone images

def read_image_file(path: str) -> list[dict]:
    """Always routed to the vision model - we can't know in advance
    whether it's a scanned page, a chart, a graph, or a table screenshot.
    No separate text/table extraction exists for a standalone image, so it
    gets the full-page prompt (transcribe everything)."""
    print(f"      -> Sending image '{os.path.basename(path)}' to Gemini vision model...")
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    vision_text = vision_extract(image, prompt=config.VISION_PROMPT_FULL_PAGE)
    return [{"page": None, "type": "vision", "embedding_text": vision_text, "display_text": vision_text}]


# Pdf

def render_pdf_page_as_image(page: "fitz.Page", dpi: int = config.PAGE_RENDER_DPI) -> Image.Image:
    """Rasterize a PyMuPDF page to a PIL image for the vision model."""
    pix = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def table_to_markdown(table, table_name="Extracted Table") -> str:
    """Render a pdfplumber-extracted table as a markdown table.
    Used as display_text — shown to the LLM once retrieved."""
    if not table or len(table) < 2:
        return ""

    headers = [str(h).strip() if h else "" for h in table[0]]
    lines = [
        f"**{table_name}**",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in table[1:]:
        cells = [str(c).strip() if c else "" for c in row]
        cells = (cells + [""] * len(headers))[: len(headers)]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def table_row_to_sentence(headers, row, table_name="") -> str:
    """Convert a single table row into a natural-language sentence.
    Used as embedding_text — what the embedding model actually sees."""
    parts = [f"{h} is {(v or '').strip()}" for h, v in zip(headers, row) if v]
    prefix = f"{table_name}: " if table_name else ""
    return prefix + ", ".join(parts) + "."


def table_to_chunks(table, page_num, table_index=1) -> list[dict]:
    """
    Converts one pdfplumber-extracted table into a list of structured
    chunks — one per row — each with its own embedding_text (natural
    language, for retrieval) and display_text (markdown, for LLM context).
    All rows share the same display_text (the full table) so the LLM
    sees the whole table for context even if only one row matched.
    """
    if not table or len(table) < 2:
        return []

    table_name = f"Table {table_index} on page {page_num}"
    headers = [str(h).strip() if h else "" for h in table[0]]
    full_markdown = table_to_markdown(table, table_name=table_name)

    chunks = []
    for row in table[1:]:
        embedding_text = table_row_to_sentence(headers, row, table_name=table_name)
        chunks.append({
            "type": "table_row",
            "table_name": table_name,
            "embedding_text": embedding_text,
            "display_text": full_markdown,   # full table, so LLM gets complete context
        })
    return chunks


def extract_page_text_and_tables(pl_page: "pdfplumber.page.Page"):
    """
    Returns (clean_text, tables) for a single pdfplumber page.
    Table regions are cropped OUT of the page before text extraction,
    so table content is never duplicated between the raw text and the
    structured table output. Bboxes are shrunk slightly inward so text
    that merely borders the table (not inside it) isn't clipped.
    """
    # Use "text" strategy instead of default "lines" to prevent infinite hangs 
    # on heavily styled PDFs with vector backgrounds (like grids/charts).
    found_tables = pl_page.find_tables(
        table_settings={"vertical_strategy": "text", "horizontal_strategy": "text"}
    )

    if not found_tables:
        return pl_page.extract_text() or "", []

    margin = 1  # points to shrink inward — avoids clipping boundary text
    non_table_area = pl_page
    for t in found_tables:
        x0, top, x1, bottom = t.bbox
        shrunk_bbox = (x0 + margin, top + margin, x1 - margin, bottom - margin)
        non_table_area = non_table_area.outside_bbox(shrunk_bbox)

    clean_text = non_table_area.extract_text() or ""
    tables = [t.extract() for t in found_tables]
    return clean_text, tables


def read_pdf(path: str) -> list[dict]:
    """
    Reads a PDF page by page and returns a list of PAGE entries, in page
    order. Each page entry is itself ordered: whole-page text first, then
    any tables found on that page, then (if applicable) the chart/vision
    description for that page:

      {
        "page": int,
        "chunks": [
            {"type": "text", "embedding_text": ..., "display_text": ...},
            {"type": "table_row", "table_name": ..., ...},   # 0+ of these
            {"type": "chart_description", ...},              # 0 or 1
        ]
      }

    Routing per page:
      - Page has little/no text layer (scanned) -> send rendered page to
        Gemini vision using VISION_PROMPT_FULL_PAGE (transcribe everything:
        text, tables, AND charts/diagrams, since vision is the only source
        of truth for this page). Single "vision" chunk, no table/chart split.
      - Page has a normal text layer -> text minus table regions, tables
        extracted structurally via pdfplumber.
      - That same page ALSO has embedded images -> additional Gemini vision
        pass using VISION_PROMPT_CHART_ONLY (describe ONLY charts/graphs/
        diagrams, since text and tables are already captured above). If
        Gemini finds no visual content, no chart_description chunk is added.

    Use flatten_pages() if you need the old flat chunk-list shape (e.g.
    for feeding an embedding/indexing step that doesn't care about page
    grouping).
    """
    doc = fitz.open(path)
    plumber_pdf = pdfplumber.open(path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        page_chunks = []
        text = page.get_text()
        has_embedded_images = len(page.get_images()) > 0
        pl_page = plumber_pdf.pages[page_num - 1]

        if len(text.strip()) < config.MIN_TEXT_LENGTH_THRESHOLD:
            print(f"      -> Page {page_num} of '{os.path.basename(path)}' looks scanned, using Gemini vision...")
            image = render_pdf_page_as_image(page)
            vision_text = vision_extract(image, prompt=config.VISION_PROMPT_FULL_PAGE)
            page_chunks.append({
                "type": "vision",
                "embedding_text": vision_text,
                "display_text": vision_text,
            })
            pages.append({"page": page_num, "chunks": page_chunks})
            continue

        clean_text, tables = extract_page_text_and_tables(pl_page)

        # 1) whole-page text comes first
        if clean_text.strip():
            page_chunks.append({
                "type": "text",
                "embedding_text": clean_text,
                "display_text": clean_text,
            })

        # 2) then any tables on this page, in order found
        for i, table in enumerate(tables, start=1):
            page_chunks.extend(table_to_chunks(table, page_num, table_index=i))

        # 3) then the chart/vision pass, if this page has embedded images
        if has_embedded_images:
            print(f"      -> Page {page_num} of '{os.path.basename(path)}' has embedded images, adding Gemini vision pass...")
            image = render_pdf_page_as_image(page)
            vision_text = vision_extract(image, prompt=config.VISION_PROMPT_CHART_ONLY)
            if vision_text.strip() and vision_text.strip() != "NO_VISUAL_CONTENT":
                page_chunks.append({
                    "type": "chart_description",
                    "embedding_text": vision_text,
                    "display_text": vision_text,
                })

        pages.append({"page": page_num, "chunks": page_chunks})

    doc.close()
    plumber_pdf.close()

    return pages


def flatten_pages(pages: list[dict]) -> list[dict]:
    """
    Converts the page-grouped structure from read_pdf() back into the
    flat chunk list (with "page" re-attached to each chunk) that the
    rest of the pipeline (embedding/indexing) expects. Order is
    preserved: page 1's chunks, then page 2's, etc.
    """
    flat = []
    for page_entry in pages:
        for chunk in page_entry["chunks"]:
            flat.append({"page": page_entry["page"], **chunk})
    return flat


def load_documents(data_dir: str = config.DATA_DIR) -> list[dict]:
    """
    Load all supported files (.txt, .pdf, .png, .jpg, .jpeg) from a folder.
    Returns a list of {"source": filename, "chunks": [...]} dicts, where
    each chunk is {"page", "type", "embedding_text", "display_text"}.
    PDFs are read page-grouped internally, then flattened here so the
    return shape stays consistent across file types.
    """
    docs = []
    for path in glob.glob(os.path.join(data_dir, "*")):
        lower = path.lower()
        try:
            if lower.endswith(config.TEXT_EXTENSIONS):
                docs.append({"source": os.path.basename(path), "chunks": read_txt(path)})
            elif lower.endswith(config.PDF_EXTENSIONS):
                pages = read_pdf(path)
                docs.append({"source": os.path.basename(path), "chunks": flatten_pages(pages)})
            elif lower.endswith(config.IMAGE_EXTENSIONS):
                docs.append({"source": os.path.basename(path), "chunks": read_image_file(path)})
            else:
                continue
        except Exception as e:
            print(f"   Warning: failed to load '{os.path.basename(path)}': {e}")
    return docs


def save_pdf_as_json(path: str, output_path: str) -> None:
    """Extract a single PDF and write its page-grouped structure to JSON."""
    pages = read_pdf(path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"source": os.path.basename(path), "pages": pages}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    output_path = os.path.join("output", "extracted_documents.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    docs = load_documents()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"Loaded {len(docs)} document(s).")
    print(f"Saved extracted output to '{output_path}'")