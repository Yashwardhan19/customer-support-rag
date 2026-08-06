import os
import io
import glob

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
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


def vision_extract(image: Image.Image) -> str:
    client = get_gemini_client()
    try:
        response = client.models.generate_content(
            model=config.GEMINI_MODEL_NAME,
            contents=[config.VISION_PROMPT, image],
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
    whether it's a scanned page, a chart, a graph, or a table screenshot."""
    print(f"      -> Sending image '{os.path.basename(path)}' to Gemini vision model...")
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    vision_text = vision_extract(image)
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
            "page": page_num,
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
    found_tables = pl_page.find_tables()

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
    Reads a PDF page by page and returns a list of structured chunks:
      {"page": int, "type": str, "embedding_text": str, "display_text": str}

    Routing per page:
      - Page has a normal text layer -> text minus table regions
      - Page has that text layer AND embedded images -> also send the
        rendered page to Gemini, since a chart/graph on an otherwise
        text-normal page won't show up in the text layer at all
      - Page has little/no text layer (scanned) -> send rendered page to
        Gemini vision model, which handles scanned text, scanned tables,
        and charts/graphs all in one pass
    Digital tables are extracted per-page via pdfplumber, cropped out of
    the text pass, and converted to one chunk per row (see table_to_chunks).
    """
    doc = fitz.open(path)
    plumber_pdf = pdfplumber.open(path)
    chunks = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        has_embedded_images = len(page.get_images()) > 0
        pl_page = plumber_pdf.pages[page_num - 1]

        if len(text.strip()) < config.MIN_TEXT_LENGTH_THRESHOLD:
            print(f"      -> Page {page_num} of '{os.path.basename(path)}' looks scanned, using Gemini vision...")
            image = render_pdf_page_as_image(page)
            vision_text = vision_extract(image)
            chunks.append({
                "page": page_num,
                "type": "vision",
                "embedding_text": vision_text,
                "display_text": vision_text,
            })
            continue

        clean_text, tables = extract_page_text_and_tables(pl_page)

        if clean_text.strip():
            if has_embedded_images:
                print(f"      -> Page {page_num} of '{os.path.basename(path)}' has embedded images, adding Gemini vision pass...")
                image = render_pdf_page_as_image(page)
                vision_text = vision_extract(image)
                chunks.append({
                    "page": page_num,
                    "type": "text",
                    "embedding_text": clean_text,
                    "display_text": clean_text,
                })
                chunks.append({
                    "page": page_num,
                    "type": "chart_description",
                    "embedding_text": vision_text,
                    "display_text": vision_text,
                })
            else:
                chunks.append({
                    "page": page_num,
                    "type": "text",
                    "embedding_text": clean_text,
                    "display_text": clean_text,
                })

        for i, table in enumerate(tables, start=1):
            chunks.extend(table_to_chunks(table, page_num, table_index=i))

    doc.close()
    plumber_pdf.close()

    return chunks


def load_documents(data_dir: str = config.DATA_DIR) -> list[dict]:
    """
    Load all supported files (.txt, .pdf, .png, .jpg, .jpeg) from a folder.
    Returns a list of {"source": filename, "chunks": [...]} dicts, where
    each chunk is {"page", "type", "embedding_text", "display_text"}.
    """
    docs = []
    for path in glob.glob(os.path.join(data_dir, "*")):
        lower = path.lower()
        try:
            if lower.endswith(config.TEXT_EXTENSIONS):
                docs.append({"source": os.path.basename(path), "chunks": read_txt(path)})
            elif lower.endswith(config.PDF_EXTENSIONS):
                docs.append({"source": os.path.basename(path), "chunks": read_pdf(path)})
            elif lower.endswith(config.IMAGE_EXTENSIONS):
                docs.append({"source": os.path.basename(path), "chunks": read_image_file(path)})
            else:
                continue
        except Exception as e:
            print(f"   Warning: failed to load '{os.path.basename(path)}': {e}")
    return docs