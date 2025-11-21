# 🚀 GenAI Service — Project Guide

## 📦 Installation & Setup

### 🔧 1) Install `uv`
```bash
pip install uv
```

### 🆕 2) Initialize a New Project

```bash
uv init .
uv add -r requirements.txt
```

### 🔄 3) Sync Dependencies (if lockfile exists)

```bash
uv sync
# For strict lockfile enforcement (CI or reproducibility):
uv sync --frozen
```

## ▶️ Running the Project with uv

### ⚡ FastAPI (Development Mode with Auto-Reload)

```bash
uv run uvicorn main:app --reload
uv run fastapi dev
```

### 🚀 FastAPI (Prod-style Uvicorn)

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

### 🎨 Streamlit Client UI

```bash
uv run streamlit run client.py
```

## 🗄️ Qdrant Vector Database (Docker)

### 📥 Pull Latest Qdrant Image

```bash
docker pull qdrant/qdrant
```

### ▶️ Run Qdrant with Persistent Local Storage

```bash
docker run -p 6333:6333 -p 6334:6334 -v "%cd%\\qdrant_storage":/qdrant/storage qdrant/qdrant
```

### 🗑️ Remove Qdrant Container

```bash
docker rm qdrant
```

If you want, I can also generate:

✅ A full **README.md**  
✅ A version with **badges** (Python, FastAPI, Docker, Qdrant, uv)  
✅ A version with **Project Structure**, **API docs**, or **RAG Architecture Diagram**

Just tell me!
