# Complexity Oracle — Container image
#
# Build:   docker build -t complexity-oracle .
# Run:     docker run -p 8080:8080 -e ANTHROPIC_API_KEY=sk-ant-... -e CLOUD_MODE=true complexity-oracle
#
# CLOUD_MODE is NOT baked in — set it at runtime in Railway / Cloud Run so the
# same image works for both local testing and cloud deployment.

FROM python:3.11-slim

WORKDIR /app

# Copy dependency spec first — Docker layer cache means this only re-runs when
# pyproject.toml changes, not on every source change.
COPY pyproject.toml .

# Copy source
COPY complexity_oracle/ ./complexity_oracle/

# Install production dependencies (no dev extras)
RUN pip install --no-cache-dir .

# Cloud Run default port
EXPOSE 8080

# Uvicorn with 2 workers — Cloud Run allocates 1 vCPU per instance by default
CMD ["uvicorn", "complexity_oracle.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
