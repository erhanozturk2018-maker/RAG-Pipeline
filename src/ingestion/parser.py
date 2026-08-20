"""
Parses PDF, DOCX, and TXT files into raw text.

Design choice: each parser returns a list of "pages" instead of one big
string. Even for formats without a real page concept (TXT), we wrap the
text in a single-page list so the rest of the pipeline (chunker) can
always assume the same shape: list[{"text": str, "page_number": int}].

Keeping page_number here means it can be carried through as chunk
metadata later -- useful for citing "this answer came from page 4".
"""

from pathlib import Path

from pypdf import PdfReader
from docx import Document as DocxDocument


def parse_pdf(file_path: Path) -> list[dict]:
    """Extract text from a PDF, one entry per page.

    Returns:
        [{"text": "...", "page_number": 1}, {"text": "...", "page_number": 2}, ...]
    """
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:  # skip pages with no extractable text (e.g. pure images)
            pages.append({"text": text, "page_number": i})
    return pages


def parse_docx(file_path: Path) -> list[dict]:
    """Extract text from a DOCX file.

    DOCX has no native "page" concept (pagination depends on the viewer/
    printer), so the whole document is returned as a single page.
    """
    doc = DocxDocument(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    if not text.strip():
        return []
    return [{"text": text, "page_number": 1}]


def parse_txt(file_path: Path) -> list[dict]:
    """Read a plain text file. Also returned as a single page."""
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [{"text": text, "page_number": 1}]


# Maps file extensions to their parser function.
_PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_txt,
}


def parse_document(file_path: Path) -> list[dict]:
    """Dispatch to the right parser based on file extension.

    Raises:
        ValueError: if the file extension isn't supported.
    """
    suffix = file_path.suffix.lower()
    parser = _PARSERS.get(suffix)
    if parser is None:
        supported = ", ".join(_PARSERS.keys())
        raise ValueError(
            f"Unsupported file type '{suffix}' for {file_path.name}. "
            f"Supported types: {supported}"
        )
    return parser(file_path)