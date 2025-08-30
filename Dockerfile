# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# System deps (curl used by healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Workdir at repo root so both api and webapp paths resolve
WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the whole repo (API, webapp, data, etc.)
COPY . /app

# Create non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose both API and Streamlit ports
EXPOSE 8000 8501

# Default CMD is the API (compose will override for dev/reload)
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]