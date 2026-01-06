# Multi-stage Dockerfile for Olive Disease Detection API
# Stage 1: Builder - Install dependencies
# Stage 2: Runtime - Minimal production image

# ============================================================================
# STAGE 1: BUILDER
# ============================================================================
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /build

# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-prod.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt


# ============================================================================
# STAGE 2: RUNTIME
# ============================================================================
FROM python:3.11-slim

# Set metadata
LABEL maintainer="Olive Disease Detection Team"
LABEL description="Production-grade FastAPI server for olive disease detection"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    MODEL_PATH="/app/models/best_model.pth" \
    SERVER_HOST="0.0.0.0" \
    SERVER_PORT="8000" \
    WORKERS="4"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser predict.py .
COPY --chown=appuser:appuser config.yaml .

# Create required directories
RUN mkdir -p /app/models /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run FastAPI application
CMD ["python", "-m", "uvicorn", "predict:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4"]