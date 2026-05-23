FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --python 3.13

COPY app/ ./app/

# Render uses port 10000 by default
EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-10000}/healthz')"

# Shell form so $PORT is expanded at runtime
CMD uv run --python 3.13 uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
