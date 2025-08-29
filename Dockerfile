# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user & dirs
WORKDIR /app
RUN useradd -m appuser
COPY requirements.txt /app/requirements.txt

# Install deps first (better caching)
RUN pip install -r /app/requirements.txt

# Copy source
# We keep your repo layout; the Python package root we will work from is /app/app
COPY app /app/app

# Make sure the 'data' folder exists and is writable (if you later mount over it in dev, that's fine)
RUN mkdir -p /app/app/data && chown -R appuser:appuser /app

# Switch to the package root so imports like "from model import ..." work
WORKDIR /app/app

# Non-root
USER appuser

# Expose API port
EXPOSE 8000

# Default command (FastAPI, no reload for prod)
# Note: we run inside /app/app so relative Path("data/...") resolves to /app/app/data
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]