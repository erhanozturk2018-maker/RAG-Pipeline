"""
Central configuration for the RAG pipeline.

Every other module reads its settings from here instead of hardcoding
values, so changing one number (e.g. chunk size) only requires editing
this file.
"""

from pathlib import Path

# --- Paths ---
# Path(__file__) = this config.py file's location.
# .parent = the folder it's in (the repo root, since config.py lives there).
# This makes paths work no matter which directory you run the script from.
BASE_DIR = Path(__file__).parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CHROMA_PERSIST_DIR = BASE_DIR / "storage" / "chroma_db"

# --- Chunking ---
# Measured in TOKENS (not characters), using the embedding model's own
# tokenizer. This matters because the embedding model has a max token
# limit (multilingual-e5-large: 512 tokens) and silently truncates
# anything beyond it -- character counts are an unreliable proxy for
# token counts, especially in agglutinative languages like Turkish where
# a single word can split into several subword tokens.
# 400 tokens leaves headroom below the 512 limit (for special tokens and
# to avoid edge-case overflow). 50 tokens of overlap keeps context that
# would otherwise be lost at a chunk boundary.
CHUNK_SIZE_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# --- Embedding ---
# Multilingual model: works well for Turkish and English alike.
# Runs locally on GPU if available (falls back to CPU automatically).
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# --- Vector store ---
# Name of the Chroma collection where chunks + embeddings are stored.
COLLECTION_NAME = "documents"

# --- Retrieval ---
# How many chunks to retrieve per query in Phase 1 (no reranker yet).
TOP_K = 5

# --- Generation ---
GEMINI_MODEL_NAME = "gemini-2.5-flash"