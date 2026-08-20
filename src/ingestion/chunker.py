"""
Splits parsed document pages into overlapping chunks, sized by TOKEN count
(not character count) -- see config.py for why this matters.

We reuse the embedding model's own tokenizer (via AutoTokenizer) instead of
hand-rolling a tokenization scheme. This guarantees the chunk boundaries we
compute here match exactly what the embedding model will actually see --
no guessing, no silent truncation.

Each chunk also gets metadata that later pipeline stages depend on:
- chunk_id:      unique id, used as the Chroma record id
- document_id:   which source document this chunk came from
- page_number:   which page it came from (for citing sources later)
- prev_chunk_id / next_chunk_id: links to neighboring chunks, used for
  "context expansion" in a later phase (pulling in a chunk's neighbors
  when it's retrieved, to give the LLM more surrounding context)
"""

from transformers import AutoTokenizer

import config

# Loaded once at import time and reused for every document -- loading a
# tokenizer from disk/cache on every call would be wasteful.
_tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)

# We intentionally tokenize the FULL page text before splitting it into
# chunks, purely to measure token counts and offsets -- this untruncated
# sequence is never fed to the model directly (each resulting chunk is
# capped at CHUNK_SIZE_TOKENS, well under the model's limit). Without this
# line, the tokenizer prints a "sequence length > model max" warning every
# time, which looks like a bug but isn't -- this silences that false alarm.
_tokenizer.model_max_length = int(1e9)


def _split_page_into_chunks(text: str) -> list[str]:
    """Split a single page's text into token-sized, overlapping windows.

    Uses the tokenizer's offset mapping to translate token positions back
    to character positions, so each returned chunk is a substring of the
    original text (not a decode of tokens, which can introduce spacing
    artifacts for some tokenizers).
    """
    encoding = _tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoding["offset_mapping"]
    total_tokens = len(offsets)

    if total_tokens == 0:
        return []

    chunks = []
    start_token = 0
    step = config.CHUNK_SIZE_TOKENS - config.CHUNK_OVERLAP_TOKENS

    while start_token < total_tokens:
        end_token = min(start_token + config.CHUNK_SIZE_TOKENS, total_tokens)

        char_start = offsets[start_token][0]
        char_end = offsets[end_token - 1][1]
        chunks.append(text[char_start:char_end].strip())

        if end_token == total_tokens:
            break
        start_token += step

    return [c for c in chunks if c]  # drop any empty chunks


def chunk_document(pages: list[dict], document_id: str) -> list[dict]:
    """Turn a parsed document's pages into a flat list of chunk records.

    Args:
        pages: output of parser.parse_document(), i.e.
               [{"text": ..., "page_number": ...}, ...]
        document_id: an identifier for the source document (e.g. filename).

    Returns:
        A flat list of chunk dicts, each shaped like:
        {
            "chunk_id": "invoice_2024_0007",
            "document_id": "invoice_2024",
            "page_number": 3,
            "text": "...",
            "prev_chunk_id": "invoice_2024_0006",  # or None if first
            "next_chunk_id": "invoice_2024_0008",  # or None if last
        }
    """
    records = []
    for page in pages:
        page_chunks = _split_page_into_chunks(page["text"])
        for chunk_text in page_chunks:
            chunk_index = len(records)
            chunk_id = f"{document_id}_{chunk_index:04d}"
            records.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "page_number": page["page_number"],
                    "text": chunk_text,
                }
            )

    # Second pass: link neighbors now that every chunk_id is known.
    for i, record in enumerate(records):
        record["prev_chunk_id"] = records[i - 1]["chunk_id"] if i > 0 else None
        record["next_chunk_id"] = records[i + 1]["chunk_id"] if i < len(records) - 1 else None

    return records
