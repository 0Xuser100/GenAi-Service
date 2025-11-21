# 🚀 GenAI Service — Project Guide

## Overview

This repository contains a small GenAI service with a FastAPI backend and a Streamlit frontend. The project uses `uv` for environment bootstrapping and can use Qdrant as a vector store (optional).

## Installation & Setup

Install project dependencies and prepare the environment. Two common approaches are shown below.

### Conda (recommended)
```bash
conda create -n genaiservice python=3.11
conda activate genaiservice
pip install -r requirements.txt
```

### Using `uv`
```bash
pip install uv
uv init .
uv add -r requirements.txt
```

### Sync dependencies (if a lockfile exists)
```bash
uv sync
# For strict lockfile enforcement (CI or reproducibility):
uv sync --frozen
```

## Running the services

Start the FastAPI backend (development server with auto-reload):
```bash
uv run uvicorn main:app --reload
# or
uv run fastapi dev
```

In a separate terminal, launch the Streamlit client UI:
```bash
uv run streamlit run client.py
```

By default the FastAPI app listens on `http://127.0.0.1:8000` and Streamlit opens at `http://localhost:8501`.

### Production-style Uvicorn
```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## Qdrant Vector Database (Docker)

Pull the latest Qdrant image:
```bash
docker pull qdrant/qdrant
```

Run Qdrant with persistent local storage:
```bash
docker run -p 6333:6333 -p 6334:6334 -v "%cd%\\qdrant_storage":/qdrant/storage qdrant/qdrant
```

Remove the Qdrant container when done:
```bash
docker rm qdrant
```

## Quick Notes
- The repository contains example code and experiments under `ch2/`, `ch3/`, and a `rag/` helper set. 
- Use the Streamlit UI (`client.py`) to interact with the service locally.

If you'd like, I can also:
- Generate a cleaned README with badges and a project structure section
- Create a `dev` script or `Makefile` to simplify running the backend and UI
- Commit the resolved merge and create a short changelog entry

Tell me which of those you'd like next.
