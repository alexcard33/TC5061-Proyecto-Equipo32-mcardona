# Containerfile API REST de MiBici con el modelo P1 de demanda.
# Podman o Docker:
#   podman build -t mibici-api:1.0 .            |  docker build -f Containerfile -t mibici-api:1.0 .
#   podman run --rm -p 8000:8000 mibici-api:1.0 |  docker run --rm -p 8000:8000 mibici-api:1.0
#
# Requisito previo: el modelo exportado del Model Registry en models/serving/p1/
# (python -m mibici.modeling.export_model). Esa carpeta está versionada en Git.

# Misma versión menor de Python con la que se entrenó y registró el modelo (3.12).
FROM docker.io/library/python:3.12-slim

LABEL org.opencontainers.image.title="mibici-api" \
      org.opencontainers.image.description="API REST (FastAPI) que sirve el modelo mibici-equipo32-p1-demanda@champion" \
      org.opencontainers.image.source="https://github.com/alexcard33/TC5061-Proyecto-Equipo32-mcardona" \
      org.opencontainers.image.version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Modo de carga del modelo: "local" (empaquetado en la imagen) o "registry"
    # (MLFLOW_TRACKING_URI + MODEL_NAME + MODEL_ALIAS/MODEL_VERSION). Ver api/model_loader.py.
    MODEL_SOURCE=local \
    MODEL_DIR=/app/models/serving/p1 \
    PORT=8000

WORKDIR /app

# Usuario sin privilegios para ejecutar el servicio.
RUN useradd --create-home --uid 10001 appuser

# 1) Dependencias primero: esta capa se reutiliza mientras requirements-api.txt no cambie.
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# 2) Código de la API y modelo exportado del Model Registry.
COPY api/ ./api/
COPY models/serving/p1/ ./models/serving/p1/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck sin curl (no viene en la imagen slim). Podman lo aplica con --format docker.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
