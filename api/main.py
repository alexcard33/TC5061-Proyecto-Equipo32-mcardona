"""API REST de MiBici:

Ejecución local (desde la raíz del repositorio)::

    uvicorn api.main:app --host 0.0.0.0 --port 8000

Documentación interactiva en http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, Request
import numpy as np
import pandas as pd

from api.model_loader import ModeloServido, cargar_modelo
from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PrediccionRequest,
    PrediccionResponse,
)

logger = logging.getLogger("mibici.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo una sola vez al arrancar el servicio."""
    try:
        app.state.modelo = cargar_modelo()
        m = app.state.modelo
        logger.info("Modelo cargado: %s v%s (%s)", m.nombre, m.version, m.fuente)
    except Exception:  # el servicio arranca y /health reporta el problema
        logger.exception("No se pudo cargar el modelo")
        app.state.modelo = None
    yield


app = FastAPI(
    title="MiBici · API de demanda (Equipo 32)",
    description=(
        "Predice cuántos viajes se iniciarán en una estación MiBici el día siguiente. "
        "Sirve el modelo `mibici-equipo32-p1-demanda@champion` del Model Registry de MLflow."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _modelo(request: Request) -> ModeloServido:
    modelo = getattr(request.app.state, "modelo", None)
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado.")
    return modelo


@app.get("/", tags=["general"])
def raiz() -> dict:
    return {"servicio": "mibici-api", "docs": "/docs", "prediccion": "POST /predict"}


@app.get("/health", response_model=HealthResponse, tags=["general"])
def health(request: Request) -> HealthResponse:
    cargado = getattr(request.app.state, "modelo", None) is not None
    return HealthResponse(status="ok" if cargado else "degradado", modelo_cargado=cargado)


@app.get("/model-info", response_model=ModelInfoResponse, tags=["modelo"])
def model_info(request: Request) -> ModelInfoResponse:
    m = _modelo(request)
    return ModelInfoResponse(
        nombre=m.nombre,
        version=m.version,
        alias=m.alias,
        run_id=m.run_id,
        model_uri=m.model_uri,
        fuente=m.fuente,
        columnas=m.columnas,
    )


@app.post("/predict", response_model=PrediccionResponse, tags=["modelo"])
def predict(payload: PrediccionRequest, request: Request) -> PrediccionResponse:
    m = _modelo(request)
    df = pd.DataFrame([obs.model_dump() for obs in payload.instancias])
    df = df[m.columnas].astype("float64")  # mismo orden y tipo que la firma del modelo
    try:
        salida = m.modelo.predict(df)
    except Exception as exc:
        logger.exception("Error al predecir")
        raise HTTPException(status_code=500, detail=f"Error al ejecutar el modelo: {exc}") from exc
    predicciones = np.asarray(salida, dtype="float64").ravel().tolist()
    return PrediccionResponse(
        modelo=m.nombre, version=m.version, alias=m.alias, predicciones=predicciones
    )
