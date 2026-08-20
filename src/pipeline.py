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

from pathlib import Path

from src.ingestion.parser import parse_document
from src.ingestion.chunker import chunk_document
from src.embedding.embedder import embed_passages
from src.embedding.vectorstore import add_chunks


def ingest_document(file_path: Path) -> int:
    """Parse, chunk, embed, and store a single document.

    Args:
        file_path: path to a .pdf, .docx, or .txt file.

    Returns:
        The number of chunks written to the vector store (0 if the file
        had no extractable text).
    """
    file_path = Path(file_path)
    # Filename without extension becomes the document_id, e.g.
    # "invoice_2024.pdf" -> "invoice_2024". Used as a prefix for chunk_ids
    # and stored as metadata so retrieval results can be traced back to
    # their source file.
    document_id = file_path.stem

    print(f"[pipeline] Parsing {file_path.name}...")
    pages = parse_document(file_path)
    if not pages:
        print(f"[pipeline] No extractable text found in {file_path.name}, skipping.")
        return 0

    print(f"[pipeline] Chunking {len(pages)} page(s)...")
    chunks = chunk_document(pages, document_id)
    if not chunks:
        print(f"[pipeline] Chunking produced no chunks for {file_path.name}, skipping.")
        return 0

    print(f"[pipeline] Embedding {len(chunks)} chunk(s)...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_passages(texts)

    print("[pipeline] Writing to vector store...")
    add_chunks(chunks, embeddings)

    print(f"[pipeline] Done: {len(chunks)} chunks stored for '{document_id}'.")
    return len(chunks)


def ingest_directory(directory: Path) -> None:
    """Ingest every supported file (.pdf, .docx, .txt) in a directory.

    Convenience wrapper for bulk-loading everything in data/raw/ at once.
    """
    directory = Path(directory)
    supported_suffixes = {".pdf", ".docx", ".txt"}
    files = [f for f in directory.iterdir() if f.suffix.lower() in supported_suffixes]

    if not files:
        print(f"[pipeline] No supported files found in {directory}.")
        return

    total_chunks = 0
    for file_path in files:
        total_chunks += ingest_document(file_path)

    print(f"[pipeline] Ingested {len(files)} file(s), {total_chunks} chunk(s) total.")