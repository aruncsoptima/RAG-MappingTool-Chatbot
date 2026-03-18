import sys
import os
import json
import logging
import tempfile
import streamlit as st
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.rag_pipeline import index_uploaded_file, chat, _get_embedder, load_index, FAISS_INDEX_PATH

logger = logging.getLogger(__name__)

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend", "indexed_files.json")


def load_registry() -> list:
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r") as f:
            return json.load(f)
    return []


def _save_registry(records: list):
    with open(REGISTRY_PATH, "w") as f:
        json.dump(records, f, indent=2)


def add_to_registry(filename: str, chunks: int):
    records = [r for r in load_registry() if r["name"] != filename]
    records.append({
        "name": filename,
        "chunks": chunks,
        "indexed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    })
    _save_registry(records)


@st.cache_resource(show_spinner=False)
def preload():
    _get_embedder()
    if os.path.exists(FAISS_INDEX_PATH):
        load_index()


def index_file(uploaded_file, progress_callback=None) -> dict:
    try:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        count = index_uploaded_file(tmp_path, progress_callback=progress_callback)
        os.unlink(tmp_path)
        add_to_registry(uploaded_file.name, count)
        return {"success": True, "chunks": count}
    except Exception as e:
        return {"success": False, "error": str(e)}


def run_chat(user_input: str, chat_history: list) -> dict:
    try:
        return chat(user_input, chat_history)
    except FileNotFoundError:
        return {
            "answer": "⚠️ Knowledge base is empty. Please upload documents in the Knowledge Base tab first.",
            "sources": [],
        }
    except Exception as e:
        return {"answer": f"❌ Error: {str(e)}", "sources": []}
