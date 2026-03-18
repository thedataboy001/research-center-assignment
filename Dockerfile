# Dockerfile
# Multi-stage build:
#   Stage 1 (builder) — installs dependencies into a virtual environment
#   Stage 2 (runtime) — copies only the venv, keeps the image lean

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first — leverages Docker layer caching.
# If requirements.txt has not changed, this layer is reused.
COPY requirements.txt .

# Create a virtual environment inside the builder stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install all Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create a non-root user for security best practice
RUN groupadd --gid 1001 appgroup && \
    useradd  --uid 1001 --gid appgroup --shell /bin/bash appuser

# Copy virtual environment from builder — no need to reinstall
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source code
COPY app/                ./app/
COPY data/research_centers.csv ./data/research_centers.csv

# Create directories for artefacts and MLflow data
RUN mkdir -p artefacts mlruns /mlflow && \
    chown -R appuser:appgroup /app /mlflow

# Switch to non-root user
USER appuser

# ── Environment variables (overridden by docker-compose or .env) ──────────────
ENV APP_ENV=production
ENV DEBUG=false
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
# Docker will call this every 30s to verify the container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" \
    || exit 1

# ── Start command ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]