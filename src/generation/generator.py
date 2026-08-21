"""
Generation: sends the user's query plus retrieved context to the Gemini
API and returns the answer.

Important separation of concerns: by the time this module runs,
retrieval has already found the relevant chunks. The LLM here does NOT
search for anything -- its only job is to read the given context and
answer the question using it.

Unlike the original CLI version, this module does NOT fail at import
time if no API key is set. The service needs to be able to start up
without a key and let the user configure one through the API -- see
is_configured() and reload_client() below.
"""

import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

import config

load_dotenv()

# Lazily created -- None until a valid key is available.
_client: genai.Client | None = None

_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context below. If the answer is not contained in the "
    "context, say you don't know instead of guessing. Answer in the same "
    "language the question was asked in."
)


def is_configured() -> bool:
    """Whether a usable Gemini API key is currently set."""
    return bool(os.getenv("GEMINI_API_KEY"))


def reload_client() -> None:
    """Re-create the Gemini client from the current GEMINI_API_KEY env var.

    Call this right after writing a new key to .env and updating
    os.environ, so the running process picks up the new key without a
    restart.
    """
    global _client
    api_key = os.getenv("GEMINI_API_KEY")
    _client = genai.Client(api_key=api_key) if api_key else None


# Attempt to initialize at import time too, in case a key is already
# present in .env from a previous run.
reload_client()


def generate_answer(query: str, context: str) -> str:
    """Generate an answer to `query`, grounded in `context`.

    Raises:
        RuntimeError: if no API key has been configured yet.
    """
    if _client is None:
        raise RuntimeError(
            "No Gemini API key configured yet. Call /configure with a "
            "valid key before asking questions."
        )

    if not context:
        return "I couldn't find any relevant information in the documents to answer that."

    prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = _client.models.generate_content(
        model=config.GEMINI_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
        ),
    )
    return response.text