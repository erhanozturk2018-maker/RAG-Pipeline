"""
Command-line entry point for the RAG pipeline.

Usage:
    python main.py ingest              # ingest everything in data/raw/
    python main.py ask "your question" # answer a question using ingested docs
    python main.py list                # show which documents are indexed
    python main.py delete <doc_id>     # remove a document from the vector store
    python main.py serve               # start the HTTP API (for other apps to call)
"""

import argparse

from rag_pipeline_core import config
from rag_pipeline_core.pipeline import (
    answer_query,
    delete_document,
    ingest_directory,
    list_indexed_documents,
)


def run_ingest(user: str | None = None) -> None:
    """Ingest every supported file found in data/raw/."""
    ingest_directory(config.RAW_DATA_DIR, user=user)


def run_ask(query: str, user: str | None = None) -> None:
    """Answer a single question and print the result."""
    result = answer_query(query, user=user)

    print("\n" + "=" * 60)
    print("ANSWER:")
    print(result["answer"])
    print("=" * 60)

    if result["matches"]:
        print("\nSources used:")
        for match in result["matches"]:
            print(
                f"  - {match['document_id']} (page {match['page_number']}, "
                f"distance {match['distance']:.4f})"
            )


def run_list() -> None:
    """Print every document currently in the vector store."""
    documents = list_indexed_documents()

    if not documents:
        print("No documents indexed yet. Run `python main.py ingest` first.")
        return

    total = sum(documents.values())
    print(f"{len(documents)} document(s) indexed, {total} chunk(s) total:\n")
    for document_id, chunk_count in sorted(documents.items()):
        print(f"  {document_id}  ({chunk_count} chunk(s))")


def run_delete(document_id: str, user: str | None = None) -> None:
    """Remove a document from the vector store and report the result."""
    removed = delete_document(document_id, user=user)

    if removed:
        print(f"Removed '{document_id}' ({removed} chunk(s)) from the vector store.")
    else:
        # Not an error: makes the command safe to re-run, and covers the
        # common mix-up of passing the filename with its extension.
        print(
            f"Nothing to remove -- no document named '{document_id}' is indexed.\n"
            "Run `python main.py list` to see the indexed document ids "
            "(these are filenames WITHOUT the extension)."
        )


def run_serve(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI service (rag_pipeline_core/api.py) using uvicorn, programmatically.

    Equivalent to running `uvicorn rag_pipeline_core.api:app --host ... --port ...`
    by hand,
    but wrapped here so there's a single consistent entry point
    (`python main.py ...`) for every way of using this project.
    """
    import uvicorn

    print(f"[main] Starting API server on http://{host}:{port}")
    print(f"[main] Interactive docs at http://{host}:{port}/docs")
    uvicorn.run("rag_pipeline_core.api:app", host=host, port=port, reload=reload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A small self-hosted RAG pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest every supported file (.pdf, .docx, .txt) in data/raw/",
    )
    # Optional on both subcommands: purely an audit-log attribution, it
    # does not gate anything. Omitting it records the operation as
    # "local", which is what every existing invocation does.
    ingest_parser.add_argument(
        "--user", type=str, default=None,
        help="Identifier recorded in the audit log (default: local)",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", type=str, help="The question to ask")
    ask_parser.add_argument(
        "--user", type=str, default=None,
        help="Identifier recorded in the audit log (default: local)",
    )

    subparsers.add_parser("list", help="List the documents currently indexed")

    delete_parser = subparsers.add_parser(
        "delete", help="Remove a document from the vector store"
    )
    delete_parser.add_argument(
        "document_id",
        type=str,
        help="Source filename WITHOUT its extension (e.g. 'test_dokuman')",
    )
    delete_parser.add_argument(
        "--user", type=str, default=None,
        help="Identifier recorded in the audit log (default: local)",
    )

    serve_parser = subparsers.add_parser("serve", help="Start the HTTP API")
    serve_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    serve_parser.add_argument(
        "--reload", action="store_true", help="Auto-restart on code changes (for development)"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(args.user)
    elif args.command == "ask":
        run_ask(args.query, args.user)
    elif args.command == "list":
        run_list()
    elif args.command == "delete":
        run_delete(args.document_id, args.user)
    elif args.command == "serve":
        run_serve(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()