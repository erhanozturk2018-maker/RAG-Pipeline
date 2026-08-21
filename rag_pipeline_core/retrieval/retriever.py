"""
Retrieval: given a user query, finds the most relevant stored chunks and
formats them into a single context string.

Phase 1 scope: embed the query, run a similarity search, concatenate the
top-k results. No reranker and no context expansion (pulling in
prev/next neighbor chunks) yet -- those are Phase 2 additions once this
baseline is confirmed to work.
"""

from rag_pipeline_core.embedding.embedder import embed_query
from rag_pipeline_core.embedding.vectorstore import query_similar

from rag_pipeline_core import config


def retrieve(query: str, top_k: int = config.TOP_K) -> list[dict]:
    """Find the most relevant chunks for a query.

    Args:
        query: the user's raw question text (no prefix -- embed_query
               adds the required "query: " prefix internally).
        top_k: how many chunks to retrieve.

    Returns:
        Matches ranked most similar first. See vectorstore.query_similar
        for the exact shape of each match dict.
    """
    query_embedding = embed_query(query)
    return query_similar(query_embedding, top_k=top_k)


def build_context(matches: list[dict]) -> str:
    """Join retrieved chunks into a single context string for the LLM.

    Phase 1: simple concatenation, each chunk labeled with its source
    document and page number so the answer can be traced back. No
    deduplication or token-budget trimming yet -- not needed at small
    scale, and will be revisited once context expansion (Phase 2) can
    pull in overlapping neighbor chunks.
    """
    if not matches:
        return ""

    parts = []
    for match in matches:
        source_label = f"[source: {match['document_id']}, page {match['page_number']}]"
        parts.append(f"{source_label}\n{match['text']}")

    return "\n\n---\n\n".join(parts)