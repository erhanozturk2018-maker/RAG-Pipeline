"""
Generation: sends the user's query plus retrieved context to the Gemini
API and returns the answer.

Important separation of concerns: by the time this module runs,
retrieval has already found the relevant chunks. The LLM here does NOT
search for anything -- its only job is to read the given context and
answer the question using it. See the "Retrieval system vs LLM" split
discussed in the architecture overview.

Uses the new unified Google Gen AI SDK (google-genai). The older
google-generativeai package reached end-of-life and no longer receives
updates or bug fixes.
"""

import os

from google import genai
from google.genai import types
from dotenv import load_dotenv

import config

# Reads the .env file in the project root into environment variables.
# Must be called before os.getenv() below.
load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Create a .env file in the project root "
        "with a line like:\n"
        "GEMINI_API_KEY=your_key_here\n"
        "Get a free key at https://aistudio.google.com/apikey"
    )

_client = genai.Client(api_key=_api_key)

_SYSTEM_INSTRUCTION = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context below. If the answer is not contained in the "
    "context, say you don't know instead of guessing. Answer in the same "
    "language the question was asked in."
)


def generate_answer(query: str, context: str) -> str:
    """Generate an answer to `query`, grounded in `context`.

    Args:
        query: the user's raw question text.
        context: retrieved chunk text, as built by retriever.build_context().

    Returns:
        The model's answer as plain text.
    """
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
