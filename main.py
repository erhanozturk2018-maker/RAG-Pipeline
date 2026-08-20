"""
Command-line entry point for the RAG pipeline.

Usage:
    python main.py ingest              # ingest everything in data/raw/
    python main.py ask "your question" # answer a question using ingested docs
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A small self-hosted RAG pipeline."
    )
    # subparsers = "ingest" and "ask" are separate subcommands, each can
    # have its own arguments.
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "ingest",
        help="Ingest every supported file (.pdf, .docx, .txt) in data/raw/",
    )

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("query", type=str, help="The question to ask")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest()
    elif args.command == "ask":
        run_ask(args.query)


if __name__ == "__main__":
    main()