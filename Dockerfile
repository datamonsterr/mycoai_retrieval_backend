FROM python:3.13-slim AS builder

RUN pip install --no-cache-dir uv==0.8.0

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --compile-bytecode

COPY src/ ./src/

FROM python:3.13-slim AS runtime

RUN groupadd -r app && useradd -r -g app app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src
COPY --from=builder /app/pyproject.toml .

RUN mkdir -p /data/uploads /data/weights && chown -R app:app /data

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

USER app

EXPOSE 8000

CMD ["uvicorn", "mycoai_retrieval_backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
