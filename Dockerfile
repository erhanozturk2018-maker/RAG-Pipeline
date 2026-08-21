# Python 3.12 is the newest release satisfying the project's
# requires-python = ">=3.10,<3.13" constraint in pyproject.toml.
#
# Note this is a plain slim image, NOT a CUDA base image. The CUDA
# runtime that torch needs ships inside the cu121 torch wheels that
# pyproject.toml's [tool.uv.sources] already pins, so the only thing
# required from the host is the driver -- which nvidia-container-toolkit
# injects at runtime (see the deploy.resources block in compose.yaml).
FROM python:3.12-slim

# Copy the official uv binary rather than pip-installing it: it's a
# single static binary, so this adds no Python-level dependencies that
# could conflict with the project's own.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- Environment ---
# PYTHONUNBUFFERED: print() output reaches `docker logs` immediately
#   instead of sitting in a block buffer.
# PYTHONDONTWRITEBYTECODE: no .pyc clutter in a layer that's rebuilt anyway.
# UV_LINK_MODE=copy: the venv and the uv cache can land on different
#   filesystems in a container; copying avoids uv's hardlink warning.
# UV_COMPILE_BYTECODE=1: precompile on install so the first request
#   doesn't pay the import-compile cost.
# HF_HOME: sentence-transformers/transformers cache the embedding model
#   here. It sits under /app/storage, which is a host bind-mount, so the
#   ~2GB multilingual-e5-large download survives container restarts
#   instead of being re-fetched every boot.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HF_HOME=/app/storage/hf_cache

# --- Dependency layer ---
# Copied and installed BEFORE the source, so editing a .py file doesn't
# invalidate this layer and force a full re-download of torch + CUDA
# wheels (several GB) on every rebuild.
#
# README.md is copied because pyproject.toml declares `readme = "README.md"`
# and uv reads the metadata even with package = false.
#
# uv.lock is copied when present (the trailing * makes it optional, since
# it's gitignored in this project) -- with it, `uv sync` reproduces exact
# pinned versions; without it, uv resolves fresh.
COPY pyproject.toml README.md uv.lock* ./
RUN uv sync --no-dev

# Put the venv uv just created first on PATH, so `python` at runtime is
# the project interpreter with every dependency already importable. That
# lets the container start without invoking `uv` at all, which would
# otherwise re-check the lockfile (and potentially reach for the network)
# on every single boot.
#
# Deliberately placed AFTER the uv sync layer: an ENV instruction
# invalidates every layer below it, and putting this above would mean any
# edit to it forces a multi-GB re-download of the CUDA wheels.
ENV PATH="/app/.venv/bin:$PATH"

# --- Source layer ---
# Only the code. data/, storage/, logs/ and .env are deliberately NOT
# copied -- they are host bind-mounts declared in compose.yaml, so the
# vector DB and raw documents stay on the host and the API key is
# injected at runtime via env_file rather than baked into a layer.
# .dockerignore is the enforcement for that; this is just the intent.
COPY main.py ./
COPY rag_pipeline_core/ ./rag_pipeline_core/

EXPOSE 8000

# --- Runtime ---
# Goes through the project's own `main.py serve` CLI rather than calling
# uvicorn directly, so the container uses the exact same entry point as a
# local run and there's only one startup path to maintain. `python` here
# is /app/.venv/bin/python via PATH above -- no `uv` at runtime.
#
# --host 0.0.0.0 (not the CLI's 127.0.0.1 default) is required: a
# container's loopback interface is reachable only from inside that
# container, so binding to 127.0.0.1 would make the API invisible to both
# the host's published port and other containers on the network.
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
