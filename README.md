# RAG MVP

A small-scale, self-hosted Retrieval-Augmented Generation (RAG) pipeline. Built from scratch, without a framework (no LangChain/LlamaIndex), so every stage of the pipeline is transparent and easy to reason about.

## How it works

```
INGESTION
  PDF / DOCX / TXT
        |
        v
  Parser            -> extracts raw text (page-aware for PDFs)
        |
        v
  Chunker           -> splits text into token-sized, overlapping chunks
        |              (sized using the embedding model's own tokenizer,
        |               not character count -- see "Design notes" below)
        v
  Embedder          -> turns each chunk into a vector (local, GPU-accelerated)
        |
        v
  Vector store       -> stores chunk text + vector + metadata (ChromaDB)


RETRIEVAL & GENERATION
  User query
        |
        v
  Embedder           -> turns the query into a vector
        |
        v
  Vector store       -> finds the top-K most similar chunks (cosine similarity)
        |
        v
  Context builder    -> concatenates retrieved chunks into a single context
        |
        v
  LLM (Gemini API)   -> answers the question using ONLY the given context
```

The LLM never searches for anything itself -- by the time it's called, retrieval has already found the relevant chunks. Its only job is to read the given context and answer.

## Tech stack

| Component | Choice | Why |
|---|---|---|
| Parsing | `pypdf`, `python-docx` | Lightweight, minimal dependencies |
| Chunking | Hand-written, token-aware | Full control, no black-box framework |
| Embedding | `intfloat/multilingual-e5-large` (local, via `sentence-transformers`) | Strong multilingual (Turkish + English) performance, runs on GPU, free |
| Vector store | ChromaDB (embedded, self-hosted) | Zero server setup, persists to disk |
| Generation | Gemini API (`gemini-3.5-flash-lite`, free tier) | No local GPU large enough for a good LLM; free tier is generous |
| Package management | `uv` | Fast, and supports pinning a CUDA-specific PyTorch index directly in `pyproject.toml` |

## Project structure

```
.
├── main.py                    # CLI entry point (ingest / ask / serve)
├── pyproject.toml
├── Dockerfile                 # Container image (uv + CUDA torch from pyproject)
├── compose.yaml               # Standalone stack; also included by wazuh-dashboard
├── .dockerignore
│
├── data/raw/                  # Put source documents here (.pdf, .docx, .txt)
├── storage/chroma_db/         # ChromaDB's persistent data (auto-created)
├── logs/                      # Daily CSV audit logs, YYYY-MM-DD.csv (auto-created)
│
├── rag_pipeline_core/
│   ├── config.py              # All tunable settings in one place
│   ├── api.py                 # FastAPI service (/status, /ingest, /ask, /documents)
│   ├── pipeline.py            # Orchestrates the full ingest / query / delete flow
│   ├── logging_utils.py       # Daily-rotating CSV audit log
│   ├── ingestion/
│   │   ├── parser.py          # File -> raw text
│   │   └── chunker.py         # Raw text -> token-sized chunks + metadata
│   ├── embedding/
│   │   ├── embedder.py        # Text -> vector (e5 query/passage convention)
│   │   └── vectorstore.py     # ChromaDB read/write
│   ├── retrieval/
│   │   └── retriever.py       # Query -> top-K chunks -> context string
│   └── generation/
│       └── generator.py       # Query + context -> LLM answer
│
└── tests/
```

`main.py` is deliberately the only Python file at the project root -- everything
else lives inside the `rag_pipeline_core` package, so the whole codebase is
importable under one namespace (`from rag_pipeline_core.pipeline import ...`)
whether it runs on the host or in a container.

## Setup

### 1. Install dependencies

```bash
uv sync
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

`pyproject.toml` pins a CUDA-enabled PyTorch build (`cu121` by default). If your GPU driver reports a different CUDA version (`nvidia-smi`), update the `cu121` references in `pyproject.toml` accordingly.

### 2. Set up your API key

```bash
cp env.example .env
```

Get a free Gemini API key at [Google AI Studio](https://aistudio.google.com/apikey) (no credit card required) and paste it into `.env`:

```
GEMINI_API_KEY=your_key_here
```

### 3. Verify GPU is detected (optional but recommended)

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Usage

```bash
# Put your documents in data/raw/, then:
python main.py ingest

# Ask a question:
python main.py ask "your question here"

# See what's actually indexed (not the same as what's in data/raw/ --
# see "Deleting a document" below):
python main.py list

# Remove a document from the vector store. Use the document_id shown by
# `list` -- the filename WITHOUT its extension.
python main.py delete test_dokuman

# Start the HTTP API instead of using the CLI directly:
python main.py serve --host 0.0.0.0 --port 8000
```

Every command accepts an optional `--user <name>` flag, recorded in the audit log (see "Audit logging" below). Defaults to `"local"` when omitted -- there's no auth system yet, this is just attribution for whoever/whatever triggered the operation.

### Deleting a document

Removing a file from `data/raw/` does **not** remove it from the vector store -- nothing scans that directory for deletions. If you want a document to stop showing up as a source in answers, you have to explicitly delete it:

```bash
python main.py delete <document_id>
```

Re-ingesting an already-indexed file also clears its *old* chunks first, then writes the new ones -- so shortening a document and re-running `ingest` doesn't leave stale, larger-indexed chunks behind from the previous version.

## HTTP API

Started with `python main.py serve` (or via Docker -- see below). Interactive docs at `/docs` once running.

| Method | Path | Purpose |
|---|---|---|
| GET | `/status` | Whether a Gemini key is configured and whether anything is indexed |
| POST | `/ingest` | Ingest every supported file in `data/raw/`. Optional JSON body: `{"user": "..."}` |
| POST | `/ask` | `{"query": "...", "user": "..."}` (user optional) -> answer + source chunks |
| GET | `/documents` | `{document_id: chunk_count}` for everything currently indexed |
| DELETE | `/documents/{document_id}` | Remove a document's chunks. Idempotent -- deleting something not indexed returns 200 with `chunks_removed: 0`, not a 404 |

## Docker

```bash
cp env.example .env   # fill in GEMINI_API_KEY first
docker compose build
docker compose up -d
```

The container binds to `0.0.0.0:8000` (published as `${RAG_PORT:-8000}:8000` on the host) and runs the exact same `main.py serve` entry point as a local run. `data/`, `storage/`, and `logs/` are host bind-mounts, not baked into the image -- the vector DB, raw documents, and audit logs all persist on the host and survive a rebuild. `.env` is never copied into the image; the API key is injected at container start via `env_file`.

GPU passthrough is enabled by default (`deploy.resources.reservations.devices`, `driver: nvidia`) so the embedding model runs on the host GPU inside the container too -- this requires `nvidia-container-toolkit` on the host. Without it, remove that block to run CPU-only.

Set `TZ` in `.env` (e.g. `TZ=Europe/Istanbul`) so audit-log timestamps written inside the container match the host's timezone -- it defaults to UTC otherwise.

## Audit logging

Every `ingest`, `ask`, `delete`, and service `startup` writes one row to `logs/YYYY-MM-DD.csv` -- a new file starts automatically each calendar day. Columns:

```
timestamp, event_type, user, model, embedding_model, chunk_count,
file, query, description, status, error, duration_ms
```

`timestamp` is local time with an explicit UTC offset. A logging failure (disk full, permissions) never crashes the actual operation -- it's caught and printed as a warning instead, since an audit trail isn't worth failing a real ingest or query over.

## Design notes

- **Token-based chunking, not character-based.** Chunk boundaries are computed using the embedding model's own tokenizer, because character count is an unreliable proxy for token count -- especially in agglutinative languages like Turkish, where a single word can split into several subword tokens. Character-based chunking risks silently exceeding the model's max token limit (truncation with no warning).
- **e5 query/passage prefix convention.** The `multilingual-e5-large` model was trained asymmetrically: it expects a `"passage: "` prefix on indexed text and a `"query: "` prefix on questions. The embedder module adds these automatically -- callers always pass raw, unprefixed text.
- **Chunk metadata carries `prev_chunk_id` / `next_chunk_id`.** Not used yet, but sets up "context expansion" (pulling in a chunk's neighbors when it's retrieved) for Phase 2.
- **Deletion is explicit, not automatic.** The vector store never scans `data/raw/` to notice a file went missing and prune it -- an "anything missing gets purged" sweep would wipe the whole collection the moment the directory looked empty for the wrong reason (an unmounted volume, a container starting before its bind-mount is ready). `delete` (CLI) / `DELETE /documents/{id}` (API) are the only way chunks leave the store, aside from ingestion overwriting a document's own previous chunks.

## Roadmap

**Phase 1 (current)** -- MVP: parse -> chunk -> embed -> store -> retrieve -> generate. No reranking, no query expansion, no context expansion.

**Phase 2 (planned)**:
- Reranker (cross-encoder) to improve retrieval precision
- Context expansion using the `prev_chunk_id` / `next_chunk_id` links already stored
- Multi-query expansion for better recall

**Phase 3 (if scale grows)**:
- Migrate from ChromaDB to Qdrant if filtering/scaling needs grow
- Per-document filtering in retrieval

## Status

Work in progress -- Phase 1 (MVP) is functional and tested end-to-end.
