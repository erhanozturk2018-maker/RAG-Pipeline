"""
FastAPI service exposing the RAG pipeline over HTTP.

Run with:
    uv run uvicorn api:app --host 127.0.0.1 --port 8000

This lets other applications (like a separate dashboard) call the RAG
pipeline without needing Python, torch, or any of its other heavy
dependencies installed locally -- they just make HTTP requests.

Assumes GEMINI_API_KEY is already present in .env before this starts.
If your integrating application collects the key from its own UI, write
it to this project's .env file before launching this service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from src.pipeline import ingest_directory, answer_query
from src.generation.generator import is_configured

app = FastAPI(title="RAG MVP API", version="0.1.0")

# Allows browser-based frontends (e.g. a dashboard's JS running on a
# different port) to call this API directly. If you only ever call this
# from server-side Python (e.g. the `requests` library), CORS doesn't
# apply and this middleware is harmless to leave in.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your dashboard's actual origin in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


class StatusResponse(BaseModel):
    configured: bool
    chunks_available: bool


class IngestResponse(BaseModel):
    message: str


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Report whether the service is ready to answer questions.

    `configured`: whether a Gemini API key is set.
    `chunks_available`: whether anything has been ingested yet.
    """
    from src.embedding.vectorstore import _collection

    return StatusResponse(
        configured=is_configured(),
        chunks_available=_collection.count() > 0,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Ingest every supported file currently in data/raw/."""
    try:
        ingest_directory(config.RAW_DATA_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    return IngestResponse(message="Ingestion complete.")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question using the documents already ingested."""
    if not is_configured():
        raise HTTPException(
            status_code=400,
            detail="No Gemini API key configured. Add GEMINI_API_KEY to .env and restart the service.",
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = answer_query(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    return AskResponse(answer=result["answer"], sources=result["matches"])