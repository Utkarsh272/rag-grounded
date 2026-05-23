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

# Verify app imports cleanly at build time — surfaces crash reason in build logs
RUN uv run --python 3.13 python -c "import app.main; print('App imports OK')"

EXPOSE 10000

CMD uv run --python 3.13 uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
