import os
import re
import pickle
import hashlib
import logging
import numpy as np
import faiss
from dataclasses import dataclass, field
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from ollama import Client
from backend.data_loader import load_documents
from backend.prompts import CHAT_PROMPT

logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BACKEND_DIR, "faiss_index.pkl")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"
TOP_K = 5
BATCH_SIZE = 32
MAX_INPUT_LENGTH = 2000


@dataclass
class PipelineState:
    embedder: object = None
    index: object = None
    docs: list = field(default_factory=list)
    ollama: Client = field(default_factory=Client)
    retrieval_cache: dict = field(default_factory=dict)
    response_cache: dict = field(default_factory=dict)


_state = PipelineState()


def _cache_key(*args) -> str:
    return hashlib.sha256("|".join(str(a) for a in args).encode()).hexdigest()


def _get_embedder():
    if _state.embedder is None:
        _state.embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _state.embedder


def build_index(file_paths: list[str]):
    _state.docs = load_documents(file_paths)
    if not _state.docs:
        raise ValueError("No documents loaded. Check your file paths.")

    embedder = _get_embedder()
    texts = [d["text"] for d in _state.docs]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    _state.index = faiss.IndexFlatL2(dim)
    _state.index.add(embeddings.astype(np.float32))

    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump({"index": _state.index, "docs": _state.docs}, f)

    logger.info("Index built with %d chunks.", len(_state.docs))


def index_uploaded_file(file_path: str, progress_callback=None) -> int:
    if progress_callback:
        progress_callback(10, "Reading & extracting content...")

    new_docs = load_documents([file_path])
    if not new_docs:
        raise ValueError("No content extracted from the uploaded file.")

    if progress_callback:
        progress_callback(30, f"Extracted {len(new_docs)} chunks. Loading embedder...")

    if _state.index is None and os.path.exists(FAISS_INDEX_PATH):
        load_index()

    embedder = _get_embedder()
    texts = [d["text"] for d in new_docs]
    all_embeddings = []
    total = len(texts)
    total_batches = max(1, (total + BATCH_SIZE - 1) // BATCH_SIZE)

    for i in range(0, total, BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        all_embeddings.append(embedder.encode(batch, convert_to_numpy=True).astype(np.float32))

        if progress_callback:
            pct = 30 + int(((i // BATCH_SIZE + 1) / total_batches) * 55)
            progress_callback(pct, f"Embedding chunks... ({min(i + BATCH_SIZE, total)}/{total})")

    embeddings = np.vstack(all_embeddings)

    if progress_callback:
        progress_callback(88, "Storing in FAISS index...")

    if _state.index is None:
        _state.index = faiss.IndexFlatL2(embeddings.shape[1])
        _state.docs = []

    _state.index.add(embeddings)
    _state.docs.extend(new_docs)
    _state.retrieval_cache.clear()

    if progress_callback:
        progress_callback(95, "Saving index to disk...")

    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump({"index": _state.index, "docs": _state.docs}, f)

    logger.info("Indexed %d new chunks from %s.", len(new_docs), file_path)
    return len(new_docs)


def load_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Upload a document to build the index.")
    with open(FAISS_INDEX_PATH, "rb") as f:
        data = pickle.load(f)
    _state.index = data["index"]
    _state.docs = data["docs"]
    logger.info("FAISS index loaded with %d documents.", len(_state.docs))


@lru_cache(maxsize=512)
def _embed_query(query: str) -> np.ndarray:
    return _get_embedder().encode([query], convert_to_numpy=True).astype(np.float32)


def _is_counting_query(query: str) -> bool:
    keywords = ["how many", "count", "total", "number of", "list all", "all holidays", "how much"]
    return any(k in query.lower() for k in keywords)


def _retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    if _state.index is None:
        load_index()

    if _is_counting_query(query):
        return _state.docs

    key = _cache_key(query, top_k)
    if key in _state.retrieval_cache:
        return _state.retrieval_cache[key]

    distances, indices = _state.index.search(_embed_query(query), top_k * 3)
    results = [_state.docs[idx] for idx in indices[0] if idx != -1][:top_k]
    _state.retrieval_cache[key] = results
    return results


def _validate_input(text: str) -> str:
    """Sanitize and validate user input before passing to LLM."""
    text = text.strip()
    if not text:
        raise ValueError("Input cannot be empty.")
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(f"Input exceeds maximum allowed length of {MAX_INPUT_LENGTH} characters.")
    return text


def _call_llm(prompt: str) -> str:
    key = _cache_key(prompt)
    if key in _state.response_cache:
        return _state.response_cache[key]

    response = _state.ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    result = response["message"]["content"]
    _state.response_cache[key] = result
    return result


def _inject_count(query: str, docs: list[dict]) -> str:
    if not _is_counting_query(query):
        return ""
    month_match = re.search(
        r"(january|february|march|april|may|june|july|august|september|october|november|december)",
        query.lower()
    )
    if month_match:
        month = month_match.group(1)
        matching = [d for d in docs if month.lower() in d["text"].lower()]
        return f"\n[SYSTEM COUNT] There are exactly {len(matching)} items matching '{month}' in the knowledge base.\n"
    return f"\n[SYSTEM COUNT] Total items in knowledge base: {len(docs)}.\n"


def chat(user_input: str, chat_history: list) -> dict:
    user_input = _validate_input(user_input)
    retrieved = _retrieve(user_input)
    count_hint = _inject_count(user_input, retrieved)
    context = count_hint + "\n".join([f"[{d['source']}] {d['text']}" for d in retrieved])
    sources = list({d["source"] for d in retrieved})

    history_text = "\n".join(
        [f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-4:]]
    )
    prompt = CHAT_PROMPT.format(context=context, chat_history=history_text, question=user_input)
    answer = _call_llm(prompt)
    logger.info("Chat response generated for query: %.60s...", user_input)
    return {"answer": answer, "sources": sources}
