"""
FastAPI service exposing the RAG pipeline over HTTP.

Run with:
    uv run uvicorn rag_pipeline_core.api:app --host 127.0.0.1 --port 8000

or, equivalently and preferably:
    uv run python main.py serve --host 127.0.0.1 --port 8000

This lets other applications (like a separate dashboard) call the RAG
pipeline without needing Python, torch, or any of its other heavy
dependencies installed locally -- they just make HTTP requests.

Assumes GEMINI_API_KEY is already present in .env before this starts.
If your integrating application collects the key from its own UI, write
it to this project's .env file before launching this service.
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline_core import config
from rag_pipeline_core.pipeline import ingest_directory, answer_query
from rag_pipeline_core.generation.generator import is_configured
from rag_pipeline_core.logging_utils import log_event


def _safe_log(**kwargs) -> None:
    """Write an audit row without ever letting a logging failure reach the client.

    log_event() already swallows its own exceptions; this second layer
    covers a failure while building the arguments, so a broken audit log
    can never turn a working request into a 500.
    """
    try:
        log_event(**kwargs)
    except Exception as e:  # noqa: BLE001
        print(f"[api] WARNING: audit logging failed: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Record one `startup` row each time the service boots.

    Uses the lifespan API rather than the older @app.on_event("startup"),
    which FastAPI now deprecates. Everything before `yield` runs at
    startup; there is nothing to tear down on shutdown.
    """
    _safe_log(
        event_type="startup",
        embedding_model=config.EMBEDDING_MODEL_NAME,
        model=config.GEMINI_MODEL_NAME,
        description=f"RAG API service started (Gemini key configured: {is_configured()}).",
    )
    yield


app = FastAPI(title="RAG MVP API", version="0.1.0", lifespan=lifespan)

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
    # Optional so existing callers that only send {"query": ...} keep
    # working unchanged. There's no auth system yet; this lets a caller
    # (e.g. the dashboard) attribute the request to a real identifier in
    # the audit log. Falls back to "local" when omitted.
    user: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


class StatusResponse(BaseModel):
    configured: bool
    chunks_available: bool


class IngestRequest(BaseModel):
    """Optional body for /ingest -- see AskRequest.user for the rationale.

    The endpoint previously took no body at all, so this whole model is
    optional at the call site (`request: IngestRequest | None = None`)
    and POSTing an empty body still works exactly as before.
    """

    user: str | None = None


class IngestResponse(BaseModel):
    message: str


@app.get("/status", response_model=StatusResponse)
def get_status():
    """Report whether the service is ready to answer questions.

    `configured`: whether a Gemini API key is set.
    `chunks_available`: whether anything has been ingested yet.
    """
    from rag_pipeline_core.embedding.vectorstore import _collection

    return StatusResponse(
        configured=is_configured(),
        chunks_available=_collection.count() > 0,
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest | None = None):
    """Ingest every supported file currently in data/raw/."""
    user = request.user if request else None
    try:
        ingest_directory(config.RAW_DATA_DIR, user=user)
    except Exception as e:
        # ingest_directory already logged the pipeline-level failure; this
        # row records that the failure was surfaced to an HTTP caller as
        # a 500, which is the part the pipeline itself can't know about.
        _safe_log(
            event_type="ingest",
            user=user,
            embedding_model=config.EMBEDDING_MODEL_NAME,
            description="POST /ingest returned 500.",
            status="error",
            error=str(e),
        )
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    return IngestResponse(message="Ingestion complete.")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question using the documents already ingested."""
    started = time.perf_counter()

    if not is_configured():
        detail = "No Gemini API key configured. Add GEMINI_API_KEY to .env and restart the service."
        _safe_log(
            event_type="ask",
            user=request.user,
            model=config.GEMINI_MODEL_NAME,
            embedding_model=config.EMBEDDING_MODEL_NAME,
            query=request.query,
            description="POST /ask rejected: service not configured.",
            status="error",
            error=detail,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise HTTPException(status_code=400, detail=detail)

    if not request.query.strip():
        detail = "Query cannot be empty."
        _safe_log(
            event_type="ask",
            user=request.user,
            model=config.GEMINI_MODEL_NAME,
            embedding_model=config.EMBEDDING_MODEL_NAME,
            query=request.query,
            description="POST /ask rejected: empty query.",
            status="error",
            error=detail,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise HTTPException(status_code=400, detail=detail)

    try:
        result = answer_query(request.query, user=request.user)
    except Exception as e:
        # answer_query already logged the pipeline-level failure; this row
        # records the HTTP-level outcome (500) for the same request.
        _safe_log(
            event_type="ask",
            user=request.user,
            model=config.GEMINI_MODEL_NAME,
            embedding_model=config.EMBEDDING_MODEL_NAME,
            query=request.query,
            description="POST /ask returned 500.",
            status="error",
            error=str(e),
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise HTTPException(status_code=500, detail=f"Failed to generate answer: {e}")

    return AskResponse(answer=result["answer"], sources=result["matches"])
