"""
Wraps ChromaDB: writes chunks + their embeddings during ingestion, and
retrieves the most similar chunks for a query embedding during retrieval.

Kept as a separate module (not merged into embedder.py) so that swapping
Chroma for a different vector DB later only requires changing this file.
"""

import chromadb

from rag_pipeline_core import config

# --- Client setup ---
# PersistentClient writes to disk (config.CHROMA_PERSIST_DIR) so the
# index survives between runs -- you don't need to re-embed everything
# every time you start the script.
_client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))

# get_or_create_collection: creates the collection on first run, reuses
# it on later runs. "hnsw:space": "cosine" tells Chroma to rank results
# by cosine distance -- the right choice since our embeddings are
# L2-normalized (see embedder.py).
_collection = _client.get_or_create_collection(
    name=config.COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(chunks: list[dict], embeddings: list[list[float]]) -> None:
    """Write a batch of chunks and their embeddings into the collection.

    Args:
        chunks: chunk records from chunker.chunk_document(), each with
                chunk_id, document_id, page_number, text, prev_chunk_id,
                next_chunk_id.
        embeddings: one embedding vector per chunk, same order as chunks
                    (typically the output of embedder.embed_passages()).
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Got {len(chunks)} chunks but {len(embeddings)} embeddings -- "
            "these must be the same length and in the same order."
        )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]

    # Chroma metadata values must be str/int/float/bool -- None is not
    # allowed, so prev/next chunk ids fall back to "" when there is no
    # neighbor (first/last chunk in a document).
    metadatas = [
        {
            "document_id": c["document_id"],
            "page_number": c["page_number"],
            "prev_chunk_id": c["prev_chunk_id"] or "",
            "next_chunk_id": c["next_chunk_id"] or "",
        }
        for c in chunks
    ]

    # upsert (not add): re-running ingestion on the same document
    # overwrites existing chunks with the same id instead of erroring
    # out on duplicates.
    _collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query_similar(query_embedding: list[float], top_k: int = config.TOP_K) -> list[dict]:
    """Find the most similar chunks to a query embedding.

    Args:
        query_embedding: output of embedder.embed_query().
        top_k: how many results to return.

    Returns:
        A list of dicts, ranked most similar first:
        [{"chunk_id", "text", "document_id", "page_number",
          "prev_chunk_id", "next_chunk_id", "distance"}, ...]
        `distance` is cosine distance (0 = identical, 2 = opposite) --
        lower is more similar.
    """
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns everything as a list-of-lists (one outer list per
    # query -- we only ever send one query at a time here), so we unwrap
    # the outer layer and zip the parallel lists back into clean dicts.
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    matches = []
    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        matches.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "document_id": metadata["document_id"],
                "page_number": metadata["page_number"],
                "prev_chunk_id": metadata["prev_chunk_id"] or None,
                "next_chunk_id": metadata["next_chunk_id"] or None,
                "distance": distance,
            }
        )
    return matches


def delete_document(document_id: str) -> int:
    """Remove every chunk belonging to a single document.

    Chroma has no "delete by metadata and tell me how many" call, so this
    counts the matching chunks first and then deletes them -- the count
    is what the caller reports and logs.

    Args:
        document_id: the document's id, i.e. the source filename without
                     its extension (the same value chunker.chunk_document
                     stores in each chunk's metadata).

    Returns:
        How many chunks were deleted (0 if the document wasn't in the
        collection -- deleting something that isn't there is not an
        error, it just does nothing).
    """
    # include=[] because we only need the ids here; fetching documents
    # and metadatas too would pull every chunk's full text into memory
    # for nothing.
    existing = _collection.get(where={"document_id": document_id}, include=[])
    chunk_ids = existing["ids"]

    if not chunk_ids:
        return 0

    _collection.delete(ids=chunk_ids)
    return len(chunk_ids)


def list_documents() -> dict[str, int]:
    """Return every document currently in the collection and its chunk count.

    Used to tell the user what's actually indexed -- which is not
    necessarily the same as what's sitting in data/raw/, since deleting a
    file from disk does not touch the vector store (see
    pipeline.delete_document).

    Returns:
        {"document_id": chunk_count, ...}, e.g. {"test_dokuman": 2}
    """
    records = _collection.get(include=["metadatas"])

    counts: dict[str, int] = {}
    for metadata in records["metadatas"]:
        document_id = metadata["document_id"]
        counts[document_id] = counts.get(document_id, 0) + 1
    return counts
