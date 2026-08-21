"""
Orchestrates the ingestion pipeline: turns a source document into stored,
searchable chunks in the vector database.

This is the only function main.py needs to call to ingest a file -- it
hides the parser -> chunker -> embedder -> vectorstore chain behind one
call, so main.py doesn't need to know how any individual stage works.

Note: this currently only covers ingestion (writing documents in).
The query-side flow (embed query -> retrieve -> build context -> generate
answer) will be added here once retriever.py and generator.py exist.
"""

import time
from pathlib import Path

from rag_pipeline_core import config
from rag_pipeline_core.logging_utils import log_event
from rag_pipeline_core.ingestion.parser import parse_document
from rag_pipeline_core.ingestion.chunker import chunk_document
from rag_pipeline_core.embedding.embedder import embed_passages
from rag_pipeline_core.embedding.vectorstore import (
    add_chunks,
    delete_document as _delete_document_chunks,
    list_documents,
)
from rag_pipeline_core.retrieval.retriever import retrieve, build_context
from rag_pipeline_core.generation.generator import generate_answer


def ingest_document(file_path: Path, user: str | None = None) -> int:
    """Parse, chunk, embed, and store a single document.

    Args:
        file_path: path to a .pdf, .docx, or .txt file.
        user: optional identifier of whoever triggered this, recorded in
              the audit log. Defaults to "local" when not supplied.

    Returns:
        The number of chunks written to the vector store (0 if the file
        had no extractable text).
    """
    file_path = Path(file_path)
    started = time.perf_counter()
    # Filename without extension becomes the document_id, e.g.
    # "invoice_2024.pdf" -> "invoice_2024". Used as a prefix for chunk_ids
    # and stored as metadata so retrieval results can be traced back to
    # their source file.
    document_id = file_path.stem

    # The whole body is wrapped so that a failure at any stage still
    # produces an audit row before the exception continues on to the
    # caller -- the exception itself is deliberately re-raised unchanged,
    # so callers see exactly the behavior they saw before logging existed.
    try:
        print(f"[pipeline] Parsing {file_path.name}...")
        pages = parse_document(file_path)
        if not pages:
            print(f"[pipeline] No extractable text found in {file_path.name}, skipping.")
            _log_ingest(user, 0, file_path.name,
                        f"No extractable text found in {file_path.name}, skipped.", started)
            return 0

        print(f"[pipeline] Chunking {len(pages)} page(s)...")
        chunks = chunk_document(pages, document_id)
        if not chunks:
            print(f"[pipeline] Chunking produced no chunks for {file_path.name}, skipping.")
            _log_ingest(user, 0, file_path.name,
                        f"Chunking produced no chunks for {file_path.name}, skipped.", started)
            return 0

        print(f"[pipeline] Embedding {len(chunks)} chunk(s)...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embed_passages(texts)

        # Clear this document's previous chunks before writing the new
        # ones. add_chunks() upserts by chunk_id, which overwrites
        # same-numbered chunks but leaves ORPHANS behind whenever the
        # re-ingested version produces fewer chunks than the old one
        # (e.g. the file was shortened): old chunk _0007 would survive
        # with stale text and still be retrievable. Deleting first makes
        # re-ingestion a true replace.
        removed = _delete_document_chunks(document_id)
        if removed:
            print(f"[pipeline] Replacing {removed} existing chunk(s) for '{document_id}'.")

        print("[pipeline] Writing to vector store...")
        add_chunks(chunks, embeddings)
    except Exception as e:
        _log_ingest(user, "", file_path.name,
                    f"Ingestion of {file_path.name} failed.", started,
                    status="error", error=str(e))
        raise

    print(f"[pipeline] Done: {len(chunks)} chunks stored for '{document_id}'.")
    _log_ingest(user, len(chunks), file_path.name,
                f"Ingested {file_path.name} into {len(chunks)} chunk(s) "
                f"as document '{document_id}'.", started)
    return len(chunks)


def ingest_directory(directory: Path, user: str | None = None) -> None:
    """Ingest every supported file (.pdf, .docx, .txt) in a directory.

    Convenience wrapper for bulk-loading everything in data/raw/ at once.

    Args:
        directory: folder to scan (typically config.RAW_DATA_DIR).
        user: optional identifier of whoever triggered this, recorded in
              the audit log. Defaults to "local" when not supplied.

    Note on logging: each file also produces its own per-file `ingest`
    row via ingest_document(). The row written here is the batch-level
    summary, so a directory of 3 files yields 4 rows in total.
    """
    directory = Path(directory)
    started = time.perf_counter()
    supported_suffixes = {".pdf", ".docx", ".txt"}
    files = [f for f in directory.iterdir() if f.suffix.lower() in supported_suffixes]

    if not files:
        print(f"[pipeline] No supported files found in {directory}.")
        _log_ingest(user, 0, "", f"No supported files found in {directory}.", started)
        return

    total_chunks = 0
    try:
        for file_path in files:
            total_chunks += ingest_document(file_path, user=user)
    except Exception as e:
        _log_ingest(user, total_chunks, "",
                    f"Batch ingestion of {directory} failed after "
                    f"{total_chunks} chunk(s).", started,
                    status="error", error=str(e))
        raise

    print(f"[pipeline] Ingested {len(files)} file(s), {total_chunks} chunk(s) total.")
    _log_ingest(user, total_chunks, "",
                f"Ingested {len(files)} file(s) from {directory}, "
                f"{total_chunks} chunk(s) total.", started)


def answer_query(query: str, user: str | None = None) -> dict:
    """Answer a question using the documents already stored in the vector DB.

    This is the query-side counterpart to ingest_document(): it does not
    touch the parser/chunker/embedder ingestion path at all, it only reads
    from what's already been indexed.

    Args:
        query: the user's raw question text.
        user: optional identifier of whoever asked, recorded in the audit
              log. Defaults to "local" when not supplied.

    Returns:
        {
            "answer": "...",       # the LLM's generated answer
            "matches": [...],      # the chunks that were retrieved (for
                                    # inspecting/debugging what the answer
                                    # was actually grounded in)
        }
    """
    started = time.perf_counter()

    try:
        print(f"[pipeline] Retrieving chunks for query: {query!r}")
        matches = retrieve(query)

        if not matches:
            print("[pipeline] No matching chunks found in the vector store.")
            _log_ask(user, query, 0,
                     "No matching chunks found in the vector store.", started)
            return {"answer": "No relevant documents found. Have you ingested anything yet?", "matches": []}

        print(f"[pipeline] Found {len(matches)} chunk(s), building context...")
        context = build_context(matches)

        print("[pipeline] Generating answer...")
        answer = generate_answer(query, context)
    except Exception as e:
        _log_ask(user, query, "", "Failed to answer query.", started,
                 status="error", error=str(e))
        raise

    _log_ask(user, query, len(matches),
             f"Answered query using {len(matches)} retrieved chunk(s).", started)
    return {"answer": answer, "matches": matches}


def delete_document(document_id: str, user: str | None = None) -> int:
    """Remove a document's chunks from the vector store.

    Deleting a file from data/raw/ does NOT remove it from the vector
    store -- the two are independent, and nothing scans the directory for
    disappearances. This is the explicit way to retract a document, so
    it stops showing up as a source in answers.

    Deliberately NOT automatic: an "anything missing from data/raw/ gets
    purged" sweep would wipe the whole collection the moment the
    directory looked empty for the wrong reason (an unmounted volume, a
    container starting before its bind-mount is ready).

    Args:
        document_id: source filename without its extension, e.g.
                     "test_dokuman" for test_dokuman.txt.
        user: optional identifier of whoever triggered this, recorded in
              the audit log. Defaults to "local" when not supplied.

    Returns:
        The number of chunks removed. 0 means the document wasn't in the
        store -- that is not treated as an error.
    """
    started = time.perf_counter()

    try:
        removed = _delete_document_chunks(document_id)
    except Exception as e:
        _log_delete(user, "", document_id,
                    f"Failed to delete document '{document_id}'.", started,
                    status="error", error=str(e))
        raise

    if removed:
        print(f"[pipeline] Deleted {removed} chunk(s) for '{document_id}'.")
        description = f"Deleted {removed} chunk(s) for document '{document_id}'."
    else:
        print(f"[pipeline] No chunks found for '{document_id}', nothing to delete.")
        description = f"No chunks found for document '{document_id}', nothing deleted."

    _log_delete(user, removed, document_id, description, started)
    return removed


def list_indexed_documents() -> dict[str, int]:
    """Return {document_id: chunk_count} for everything currently indexed.

    Read-only and cheap, so it is not audit-logged.
    """
    return list_documents()


# --- Audit-log helpers -------------------------------------------------
# These sit at the bottom rather than the top so the file still reads as
# parse -> chunk -> embed -> store -> retrieve -> generate from the top.

def _elapsed_ms(started: float) -> int:
    """Milliseconds since `started` (a time.perf_counter() reading)."""
    return int((time.perf_counter() - started) * 1000)


def _log_ingest(user, chunk_count, file, description, started, status="success", error=""):
    """Write one `ingest` audit row.

    Wrapped in its own try/except on top of the one inside log_event, so
    that even a failure while ASSEMBLING the row (not just while writing
    it) can't escape into the pipeline.
    """
    try:
        log_event(
            "ingest",
            user=user,
            embedding_model=config.EMBEDDING_MODEL_NAME,
            chunk_count=chunk_count,
            file=file,
            description=description,
            status=status,
            error=error,
            duration_ms=_elapsed_ms(started),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] WARNING: audit logging failed: {e}")


def _log_delete(user, chunk_count, document_id, description, started,
                status="success", error=""):
    """Write one `delete` audit row. See _log_ingest for the try/except rationale.

    document_id goes in the `file` column: it IS the filename, minus the
    extension, and adding a separate column just for it would break the
    agreed CSV schema.
    """
    try:
        log_event(
            "delete",
            user=user,
            embedding_model=config.EMBEDDING_MODEL_NAME,
            chunk_count=chunk_count,
            file=document_id,
            description=description,
            status=status,
            error=error,
            duration_ms=_elapsed_ms(started),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] WARNING: audit logging failed: {e}")


def _log_ask(user, query, chunk_count, description, started, status="success", error=""):
    """Write one `ask` audit row. See _log_ingest for the try/except rationale."""
    try:
        log_event(
            "ask",
            user=user,
            model=config.GEMINI_MODEL_NAME,
            embedding_model=config.EMBEDDING_MODEL_NAME,
            chunk_count=chunk_count,
            query=query,
            description=description,
            status=status,
            error=error,
            duration_ms=_elapsed_ms(started),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline] WARNING: audit logging failed: {e}")