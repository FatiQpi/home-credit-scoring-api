FROM python:3.11-slim

# lib_lightgbm.so declare libgomp.so.1 en NEEDED, absente de la roue PyPI.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Avant COPY src/ : preserve le cache de la couche d'installation.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY artifacts/model.pkl artifacts/feature_names.json artifacts/categories.json artifacts/
COPY data/store.parquet data/

RUN useradd --create-home appuser
USER appuser

EXPOSE 7860

# PORT injecte par Cloud Run, 7860 par defaut pour Hugging Face.
# exec : uvicorn en PID 1, seule position ou il recoit SIGTERM.
CMD ["sh", "-c", "exec python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
