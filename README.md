# CDASH → SDTM Assistant

A RAG-powered chatbot and mapping tool for clinical data standards (CDASH to SDTM). Built with Streamlit, FAISS, Sentence Transformers, and Ollama (local LLM).

---

## Features

- **Chat Assistant** — Ask natural language questions about CDASH/SDTM variables and domains
- **Knowledge Base Management** — Upload PDF, CSV, or Excel files to build and extend the vector index
- **Semantic Search** — FAISS-powered retrieval with sentence-transformer embeddings
- **Local LLM** — Runs fully offline using Ollama (no API keys required)
- **Context-aware responses** — Maintains chat history and shows source references

---

## Project Structure

```
RAG_MappingTool_Chatbot/
├── backend/
│   ├── data_loader.py       # Document ingestion (PDF, CSV, Excel, TXT)
│   ├── rag_pipeline.py      # Embedding, FAISS indexing, retrieval, LLM chat
│   ├── prompts.py           # LLM prompt templates
│   ├── requirements.txt     # Backend dependencies
│   └── faiss_index.pkl      # Persisted FAISS index (auto-generated)
├── frontend/
│   ├── app.py               # Streamlit UI
│   ├── utils.py             # UI helper functions
│   └── requirements.txt     # Frontend dependencies
└── README.md
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Ollama | Latest |
| llama3 model | via Ollama |

---

## Setup & Installation

### 1. Install Ollama and pull the LLM

Download Ollama from https://ollama.com and then run:

```bash
ollama pull llama3
```

Verify it works:

```bash
ollama run llama3 "Hello"
```

### 2. Clone the repository

```bash
git clone <your-repo-url>
cd RAG_MappingTool_Chatbot
```

### 3. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

---

## Running the App

Make sure Ollama is running in the background, then:

```bash
cd frontend
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## How to Use

### Step 1 — Build the Knowledge Base

1. Open the app in your browser
2. On the left panel under **Knowledge Base**, click **Browse files**
3. Upload one or more files:
   - `.pdf` — CDASH/SDTM specification documents
   - `.csv` / `.xlsx` — Mapping tables (columns like `cdash_variable`, `sdtm_variable`, `domain`, `description`)
   - `.txt` — Plain text reference documents
4. Wait for the progress bar to complete — chunks are embedded and stored in FAISS

### Step 2 — Chat with the Assistant

1. Type your question in the chat box on the right panel
2. Example queries:
   - `"What domain does AEDECOD belong to?"`
   - `"How does CDASH AESTDAT map to SDTM?"`
   - `"Explain the difference between AETERM and AEDECOD"`
   - `"List variables in the LB domain"`
3. The assistant responds using only your uploaded documents
4. Click **📚 Sources** under any response to see which documents were used

### Step 3 — Extend the Knowledge Base

Upload additional files at any time — new chunks are appended to the existing index without rebuilding from scratch.

---

## Recommended CSV Format for Mapping Data

```csv
cdash_variable,sdtm_variable,sdtm_domain,description
AEDECOD,AEDECOD,AE,Dictionary-derived adverse event term
AESTDAT,AESTDTC,AE,Adverse event start date in ISO 8601 format
VSORRESU,VSORRESU,VS,Original units for vital signs measurement
LBTEST,LBTEST,LB,Lab test name
CMTRT,CMTRT,CM,Concomitant medication name
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `"Knowledge base is empty"` | Upload at least one document in the Knowledge Base panel |
| `"I don't have enough information"` | The uploaded documents don't cover that topic — add relevant files |
| Ollama connection error | Make sure Ollama is running: `ollama serve` |
| Slow first response | The embedding model downloads on first run (~90MB) — subsequent runs are fast |
| PDF tables not extracted | Ensure the PDF is not scanned/image-based; use text-based PDFs |

---

## Tech Stack

- [Streamlit](https://streamlit.io) — UI
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Sentence Transformers](https://www.sbert.net) — `all-MiniLM-L6-v2` embeddings
- [Ollama](https://ollama.com) — Local LLM inference (llama3)
- [pdfplumber](https://github.com/jsvine/pdfplumber) — PDF text and table extraction
- [pandas](https://pandas.pydata.org) — CSV/Excel processing

---

## License

MIT
