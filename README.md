# MycoAI Retrieval Backend

FastAPI backend for retrieval, indexing, and scientist-facing data management workflows around `fungal-cv-qdrant`.

## Stack

- Python 3.13
- FastAPI
- Uvicorn
- Ruff
- MyPy
- Pytest
- uv

## Commands

```bash
uv sync --all-groups
uv run mycoai-retrieval-backend
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pytest
```

## API

- `GET /health` - healthcheck endpoint
- `GET /` - service metadata
