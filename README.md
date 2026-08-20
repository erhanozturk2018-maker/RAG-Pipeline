# RAG MVP

A small-scale, self-hosted Retrieval-Augmented Generation (RAG) pipeline.

## Stack

- **Parsing:** pypdf, python-docx
- **Embedding:** sentence-transformers (local, GPU-accelerated)
- **Vector DB:** ChromaDB (embedded, self-hosted)
- **Generation:** Gemini API (free tier)

## Setup

```bash
uv sync
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
```

## Status

Work in progress — MVP stage.
