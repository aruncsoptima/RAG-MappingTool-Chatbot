import os
import re
import logging
import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".pdf", ".txt"}


def _safe_path(file_path: str) -> str:
    """Resolve and validate file path to prevent path traversal."""
    resolved = os.path.realpath(file_path)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"File not found: {resolved}")
    ext = os.path.splitext(resolved)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    return resolved


def load_csv_or_excel(file_path: str) -> list[str]:
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    chunks = []
    for _, row in df.iterrows():
        parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        chunks.append(". ".join(parts) + ".")
    return chunks


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
        i += chunk_size - overlap
    return chunks


def load_pdf(file_path: str) -> list[str]:
    chunks = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            table_bbox_texts = set()

            for table in tables:
                if not table or len(table) < 2:
                    continue
                headers = [str(h).strip() if h else "" for h in table[0]]
                table_rows = []
                for row in table:
                    row_text = " | ".join(str(c).strip() if c else "" for c in row)
                    table_bbox_texts.add(row_text)
                    if row == table[0]:
                        continue
                    if not any(row):
                        continue
                    parts = [
                        f"{headers[i]}: {str(val).strip()}"
                        for i, val in enumerate(row)
                        if val and str(val).strip()
                    ]
                    if parts:
                        table_rows.append(". ".join(parts))
                if table_rows:
                    chunks.append("\n".join(table_rows))

            text = page.extract_text()
            if text:
                paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
                if not paragraphs:
                    paragraphs = [text.strip()]

                for para in paragraphs:
                    if para in table_bbox_texts:
                        continue
                    para_chunks = _chunk_text(para)
                    for c in para_chunks:
                        if len(c) > 2 and c not in table_bbox_texts:
                            chunks.append(c)
    return chunks


def load_txt(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 20]


def load_documents(file_paths: list[str]) -> list[dict]:
    docs = []
    for path in file_paths:
        try:
            safe = _safe_path(path)
        except (FileNotFoundError, ValueError) as e:
            logger.warning("Skipping file %s: %s", path, e)
            continue

        name = safe.lower()
        domain = _infer_domain(os.path.basename(name))

        try:
            if name.endswith(".csv") or name.endswith(".xlsx"):
                chunks = load_csv_or_excel(safe)
            elif name.endswith(".pdf"):
                chunks = load_pdf(safe)
            elif name.endswith(".txt"):
                chunks = load_txt(safe)
            else:
                continue
        except Exception as e:
            logger.error("Failed to load %s: %s", safe, e)
            continue

        docs.extend({"text": chunk, "source": safe, "domain": domain} for chunk in chunks)
        logger.info("Loaded %d chunks from %s", len(chunks), safe)
    return docs


def _infer_domain(filename: str) -> str:
    for domain in ["ae", "lb", "dm", "cm", "vs", "ex", "mh"]:
        if domain in filename:
            return domain.upper()
    return "GENERAL"
