"""
Command-line entry point for the RAG pipeline.

Usage:
    python main.py ingest              # ingest everything in data/raw/
    python main.py ask "your question" # answer a question using ingested docs
    python main.py serve               # start the HTTP API (for other apps to call)
"""

import argparse

import config
from src.pipeline import ingest_directory, answer_query


def run_ingest() -> None:
    """Ingest every supported file found in data/raw/."""
    ingest_directory(config.RAW_DATA_DIR)


def run_ask(query: str) -> None:
    """Answer a single question and print the result."""
    result = answer_query(query)

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


def run_serve(host: str, port: int, reload: bool) -> None:
    """Start the FastAPI service (api.py) using uvicorn, programmatically.

    Equivalent to running `uvicorn api:app --host ... --port ...` by hand,
    but wrapped here so there's a single consistent entry point
    (`python main.py ...`) for every way of using this project.
    """
    import uvicorn

    print(f"[main] Starting API server on http://{host}:{port}")
    print(f"[main] Interactive docs at http://{host}:{port}/docs")
    uvicorn.run("api:app", host=host, port=port, reload=reload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A small self-hosted RAG pipeline."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "ingest",
        help="Ingest every supported file (.pdf, .docx, .txt) in data/raw/",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", type=str, help="The question to ask")

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
        run_ingest()
    elif args.command == "ask":
        run_ask(args.query)
    elif args.command == "serve":
        run_serve(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()