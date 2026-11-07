"""Esquemas de entrada y salida de la API (Pydantic).

Las 12 variables de ``ObservacionP1`` son exactamente la firma (signature) del modelo
``mibici-equipo32-p1-demanda`` registrado en MLflow: una fila = una estación en un día.
"""

from pydantic import BaseModel, Field


class ObservacionP1(BaseModel):
    """Features de una estación-día para predecir los viajes que iniciarán en ella."""

    dia_semana: float = Field(..., description="Día de la semana (0 = lunes ... 6 = domingo).")
    fin_semana: float = Field(..., description="1 si es sábado o domingo, 0 en otro caso.")
    feriado: float = Field(..., description="1 si es día feriado, 0 en otro caso.")
    lag_1: float = Field(..., description="Viajes iniciados en la estación el día anterior.")
    lag_2: float = Field(..., description="Viajes iniciados hace 2 días.")
    lag_7: float = Field(..., description="Viajes iniciados hace 7 días.")
    lag_14: float = Field(..., description="Viajes iniciados hace 14 días.")
    media_7d: float = Field(..., description="Media de viajes diarios de los últimos 7 días.")
    media_28d: float = Field(..., description="Media de viajes diarios de los últimos 28 días.")
    media_mismo_dia_4sem: float = Field(
        ..., description="Media de viajes del mismo día de la semana en las últimas 4 semanas."
    )
    dias_activa: float = Field(..., description="Días transcurridos desde que la estación opera.")
    capacidad: float = Field(..., description="Número de anclajes de la estación.")


class PrediccionRequest(BaseModel):
    """Lote de observaciones a predecir."""

    instancias: list[ObservacionP1] = Field(..., description="Observaciones estación-día.")


class PrediccionResponse(BaseModel):
    """Predicciones del modelo, en el mismo orden que las instancias recibidas."""

    modelo: str
    version: str
    alias: str | None = None
    predicciones: list[float]


class HealthResponse(BaseModel):
    status: str
    modelo_cargado: bool


class ModelInfoResponse(BaseModel):
    nombre: str
    version: str
    alias: str | None = None
    run_id: str | None = None
    model_uri: str
    fuente: str
    columnas: list[str]
