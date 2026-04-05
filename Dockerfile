# CoralGuard AI API — production image (code, UI, thesis PNGs, bundled models + tabular).
# Weights live under app/models/; tabular under data/tabular/ (tracked in Git for deploy).

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY frontend ./frontend
COPY static ./static
COPY artifacts ./artifacts

RUN mkdir -p /app/uploads

# Render and other PaaS set PORT; default 8000 for local Docker
EXPOSE 8000

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
