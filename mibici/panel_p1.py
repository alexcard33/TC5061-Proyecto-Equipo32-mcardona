"""Panel estación-día de P1 continuo entre 2025 y 2026.

Lo usan el monitoreo de drift, el reentrenamiento de la versión 2 y el A/B testing, para
que los tres trabajen exactamente con las mismas features. No depende de MLflow ni de
Evidently, así que funciona tanto en el entorno principal como en el de monitoreo.
"""

from pathlib import Path

import pandas as pd

from mibici.config import (
    AB_EVAL_END,
    DRIFT_ACTUAL_START,
    P1_FERIADOS,
    P1_HORIZONTE_DIAS,
    P1_LAGS_DIAS,
    P1_MAX_DURATION_SECONDS,
    P1_MIN_DURATION_SECONDS,
    P1_TRAIN_START,
    P1_VENTANAS_MEDIA_DIAS,
)
from mibici.dataset import leer_viajes
from mibici.features import p1_agregar_features, p1_construir_panel_diario


def cargar_viajes_p1() -> pd.DataFrame:
    """Viajes de 2025 y 2026 con el mismo filtro de duración del entrenamiento de P1.

    Es la misma regla de p1_regresion.cargar_viajes. No se importa de ahí porque ese
    módulo necesita MLflow, que no se instala en el entorno de monitoreo.
    """
    viajes = leer_viajes(columnas=["inicio_viaje", "fin_viaje", "origen_id"])
    duracion = (viajes["fin_viaje"] - viajes["inicio_viaje"]).dt.total_seconds()
    return viajes.loc[duracion.between(P1_MIN_DURATION_SECONDS, P1_MAX_DURATION_SECONDS)].dropna()


def cargar_capacidad(ruta: Path) -> pd.Series:
    estaciones = pd.read_parquet(ruta, columns=["id", "dpcapacity"])
    return estaciones.set_index("id")["dpcapacity"]


def cortar(df: pd.DataFrame, inicio, fin) -> pd.DataFrame:
    """Filas con fecha entre inicio (incluido) y fin (excluido)."""
    fecha = df.index.get_level_values("fecha")
    return df.loc[(fecha >= pd.Timestamp(inicio)) & (fecha < pd.Timestamp(fin))]


def construir_features(
    viajes: pd.DataFrame,
    capacidad_ref: pd.Series,
    capacidad_actual: pd.Series,
    fin: str = AB_EVAL_END,
) -> pd.DataFrame:
    """Panel estación-día con las features de P1, desde enero de 2025 hasta fin (excluido).

    El panel empieza en la misma fecha que el entrenamiento de la versión vigente, así que
    las filas de 2025 salen idénticas a las que vio ese modelo, y las de 2026 usan la
    historia continua desde 2025, como pasaría en producción.

    La capacidad se toma del catálogo de cada periodo: el del entrenamiento para 2025 y la
    nomenclatura más reciente para 2026, que ya trae estaciones nuevas y cambios.
    """
    panel = p1_construir_panel_diario(viajes, P1_TRAIN_START, fin)
    df = p1_agregar_features(
        panel, capacidad_ref, P1_FERIADOS, P1_LAGS_DIAS, P1_VENTANAS_MEDIA_DIAS, P1_HORIZONTE_DIAS
    )
    df["capacidad"] = df["capacidad"].astype("float64")

    es_actual = df.index.get_level_values("fecha") >= pd.Timestamp(DRIFT_ACTUAL_START)
    estaciones_2026 = df.index.get_level_values("estacion")[es_actual]
    capacidad_2026 = estaciones_2026.map(capacidad_actual).fillna(0).to_numpy(dtype="float64")
    df.loc[es_actual, "capacidad"] = capacidad_2026
    return df
