FROM python:3.13-slim

WORKDIR /app

# Install system deps needed by unstructured (PDF parsing)
RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (layer cache)
COPY pyproject.toml uv.lock ./

# Pin Python to 3.13 so uv doesn't grab 3.14 (spacy has no 3.14 wheel yet)
RUN uv sync --frozen --no-dev --python 3.13

# Copy app source
COPY app/ ./app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"

CMD ["uv", "run", "--python", "3.13", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
