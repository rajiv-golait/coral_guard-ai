# CoralGuard AI API — production-style image (code + UI + static thesis figures).
# Trained weights (.pth / .pkl) and tabular .npy/.pkl are often gitignored: mount them at run
# (see docker-compose.yml) or copy into the build context before `docker build`.

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
COPY frontend ./frontend
COPY static ./static
COPY artifacts ./artifacts

# Writable upload dir; tabular + models supplied via bind mount or `docker cp`
RUN mkdir -p /app/data/tabular /app/uploads /app/app/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
