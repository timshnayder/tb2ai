FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt \
    && pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu

FROM python:3.11-slim AS runner

WORKDIR /app

RUN groupadd -g 999 appuser && \
    useradd -r -u 999 -g appuser appuser

# Copy installed Python packages from the builder stage
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy source code and files
COPY wsgi.py .
COPY tension_ai/ ./tension_ai/
COPY data/ ./data/
COPY modeltraining/ ./modeltraining/

# Set ownership to appuser
RUN chown -R appuser:appuser /app

USER appuser

# Environment variables
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Expose Gunicorn port
EXPOSE 8000

# Run with Gunicorn production server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "wsgi:app"]
