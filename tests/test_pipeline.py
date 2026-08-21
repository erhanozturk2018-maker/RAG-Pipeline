"""Smoke tests for the package layout.

tests/test_pipeline.py was previously an empty placeholder. These tests
were added to give `pythonpath = ["."]` in pyproject.toml something real
to prove: with the wrong value, importing rag_pipeline_core fails at
collection time and these tests error out rather than silently passing.

Deliberately kept to import + path checks -- no ingestion or query
behavior is exercised here, since that needs a GPU, a downloaded
embedding model, and a Gemini API key.
"""

from pathlib import Path

from rag_pipeline_core import config
from rag_pipeline_core.logging_utils import FIELDNAMES


def test_package_imports_from_project_root():
    """Every module in the package resolves under the configured pythonpath."""
    import rag_pipeline_core.api  # noqa: F401
    import rag_pipeline_core.embedding.embedder  # noqa: F401
    import rag_pipeline_core.embedding.vectorstore  # noqa: F401
    import rag_pipeline_core.generation.generator  # noqa: F401
    import rag_pipeline_core.ingestion.chunker  # noqa: F401
    import rag_pipeline_core.ingestion.parser  # noqa: F401
    import rag_pipeline_core.pipeline  # noqa: F401
    import rag_pipeline_core.retrieval.retriever  # noqa: F401


def test_base_dir_points_at_project_root_not_the_package():
    """config.BASE_DIR must climb out of rag_pipeline_core/.

    Regression guard for the package move: config.py now lives inside the
    package, so a single .parent would silently point data/ and storage/
    at rag_pipeline_core/data and rag_pipeline_core/storage -- creating an
    empty vector DB in the wrong place with no error.
    """
    assert config.BASE_DIR.name != "rag_pipeline_core"
    assert (config.BASE_DIR / "main.py").is_file()
    assert config.RAW_DATA_DIR == config.BASE_DIR / "data" / "raw"
    assert config.CHROMA_PERSIST_DIR == config.BASE_DIR / "storage" / "chroma_db"


def test_main_is_the_only_root_level_module():
    """main.py is the sole Python file at the project root."""
    root_py = {p.name for p in Path(config.BASE_DIR).glob("*.py")}
    assert root_py == {"main.py"}


def test_audit_log_schema_is_the_agreed_column_order():
    """Downstream readers depend on this exact order -- append, never insert."""
    assert FIELDNAMES == [
        "timestamp",
        "event_type",
        "user",
        "model",
        "embedding_model",
        "chunk_count",
        "file",
        "query",
        "description",
        "status",
        "error",
        "duration_ms",
    ]
