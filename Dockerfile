# Minimal image for the offline RAG service. No GPU, no API key required.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    RAG_CONFIG=config/base.yaml

WORKDIR /app

# Install deps first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

# Runtime data: the committed sample corpus and gold set.
COPY config ./config
COPY data/sample ./data/sample
COPY eval/gold ./eval/gold

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "rag_eval.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
