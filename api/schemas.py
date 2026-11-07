"""Esquemas de entrada y salida de la API (Pydantic v2).

Las 12 variables de ``ObservacionP1`` son exactamente la firma (signature) del modelo
``mibici-equipo32-p1-demanda`` registrado en MLflow: una fila = una estación en un día.

Validaciones de entrada:
- Todos los campos son obligatorios y no se aceptan campos desconocidos (``extra="forbid"``).
- Tipos y dominios: ``dia_semana`` entero 0-6; ``fin_semana`` y ``feriado`` binarios 0/1;
  conteos y medias de viajes >= 0, finitos (sin NaN ni infinito) y con tope de cordura;
  ``dias_activa`` entero >= 0; ``capacidad`` entero >= 1.
- Coherencia entre campos: ``fin_semana`` debe corresponder a ``dia_semana`` (5 y 6 = sábado y
  domingo).
- Lote de 1 a ``MAX_INSTANCIAS`` observaciones.

Validaciones de salida: una predicción por instancia, finita y no negativa (es un conteo de
viajes).
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_VIAJES_DIA = 5_000  # tope de cordura: ninguna estación MiBici se acerca a este valor
MAX_DIAS_ACTIVA = 20_000
MAX_CAPACIDAD = 200
MAX_INSTANCIAS = 1_000

Binario = Annotated[int, Field(ge=0, le=1)]
Viajes = Annotated[float, Field(ge=0, le=MAX_VIAJES_DIA, allow_inf_nan=False)]

EJEMPLO_OBSERVACION = {
    "dia_semana": 5,
    "fin_semana": 1,
    "feriado": 0,
    "lag_1": 62,
    "lag_2": 75,
    "lag_7": 31,
    "lag_14": 40,
    "media_7d": 54.0,
    "media_28d": 54.43,
    "media_mismo_dia_4sem": 35.0,
    "dias_activa": 304,
    "capacidad": 15,
}


class ObservacionP1(BaseModel):
    """Features de una estación-día para predecir los viajes que iniciarán en ella."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EJEMPLO_OBSERVACION})

    dia_semana: int = Field(..., ge=0, le=6, description="Día de la semana (0 = lunes ... 6 = domingo).")
    fin_semana: Binario = Field(..., description="1 si es sábado o domingo, 0 en otro caso.")
    feriado: Binario = Field(..., description="1 si es día feriado, 0 en otro caso.")
    lag_1: Viajes = Field(..., description="Viajes iniciados en la estación el día anterior.")
    lag_2: Viajes = Field(..., description="Viajes iniciados hace 2 días.")
    lag_7: Viajes = Field(..., description="Viajes iniciados hace 7 días.")
    lag_14: Viajes = Field(..., description="Viajes iniciados hace 14 días.")
    media_7d: Viajes = Field(..., description="Media de viajes diarios de los últimos 7 días.")
    media_28d: Viajes = Field(..., description="Media de viajes diarios de los últimos 28 días.")
    media_mismo_dia_4sem: Viajes = Field(
        ..., description="Media de viajes del mismo día de la semana en las últimas 4 semanas."
    )
    dias_activa: int = Field(
        ..., ge=0, le=MAX_DIAS_ACTIVA, description="Días transcurridos desde que la estación opera."
    )
    capacidad: int = Field(
        ..., ge=1, le=MAX_CAPACIDAD, description="Número de anclajes de la estación."
    )

    @model_validator(mode="after")
    def fin_semana_coherente(self):
        esperado = 1 if self.dia_semana in (5, 6) else 0
        if self.fin_semana != esperado:
            raise ValueError(
                f"fin_semana={self.fin_semana} no corresponde a dia_semana={self.dia_semana} "
                f"(se esperaba {esperado}; 5 y 6 son sábado y domingo)."
            )
        return self


class PrediccionRequest(BaseModel):
    """Lote de observaciones a predecir."""

    model_config = ConfigDict(
        extra="forbid", json_schema_extra={"example": {"instancias": [EJEMPLO_OBSERVACION]}}
    )

    instancias: list[ObservacionP1] = Field(
        ...,
        min_length=1,
        max_length=MAX_INSTANCIAS,
        description=f"De 1 a {MAX_INSTANCIAS} observaciones estación-día.",
    )


class PrediccionResponse(BaseModel):
    """Predicciones del modelo, en el mismo orden que las instancias recibidas."""

    modelo: str = Field(..., description="Registered model de MLflow.")
    version: str = Field(..., description="Versión del registered model.")
    alias: str | None = Field(None, description="Alias del registry (p. ej. champion).")
    n_instancias: int = Field(..., ge=1)
    predicciones: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]] = Field(
        ..., description="Viajes esperados por instancia (conteo, no negativo)."
    )

    @model_validator(mode="after")
    def una_prediccion_por_instancia(self):
        if len(self.predicciones) != self.n_instancias:
            raise ValueError("El número de predicciones no coincide con el de instancias.")
        return self


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
