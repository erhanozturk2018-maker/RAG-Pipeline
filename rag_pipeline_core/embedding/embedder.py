"""
Turns text into vectors using the embedding model defined in config.py.

Important model-specific detail: the e5 family of models (including
multilingual-e5-large) was trained asymmetrically -- it expects a
"query: " prefix on questions and a "passage: " prefix on the text being
searched over. Without these prefixes the model still returns a vector
(no error), but similarity scores become unreliable. This module adds the
correct prefix automatically so the rest of the pipeline never has to
think about it.

Reference: https://huggingface.co/intfloat/multilingual-e5-large
(model card documents the "query:" / "passage:" convention)
"""

import torch
from sentence_transformers import SentenceTransformer

from rag_pipeline_core import config

# --- Device selection ---
# Use the GPU if available, otherwise fall back to CPU automatically.
_device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[embedder] Using device: {_device}")

# Loaded once at import time and reused -- loading the model from disk
# on every call would be slow and wasteful.
_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=_device)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Embed a list of chunk texts (documents being indexed).

    Args:
        texts: raw chunk texts, WITHOUT any prefix -- this function adds
               the required "passage: " prefix itself.

    Returns:
        A list of embedding vectors (one per input text), L2-normalized
        so that cosine similarity can be computed as a simple dot product.
    """
    prefixed = [f"passage: {text}" for text in texts]
    embeddings = _model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 50,
    )
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single user query.

    Args:
        text: the raw query text, WITHOUT any prefix -- this function adds
              the required "query: " prefix itself.

    Returns:
        A single embedding vector, L2-normalized.
    """
    prefixed = f"query: {text}"
    embedding = _model.encode(prefixed, normalize_embeddings=True)
    return embedding.tolist()