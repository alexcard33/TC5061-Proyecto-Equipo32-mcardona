"""Utilidades de MLflow comunes a P1, P2 y P3: tags estándar y artefactos de soporte.

Estructura de artefactos de cada corrida:

    modelo/       modelo con firma (signature) y ejemplo de entrada (log_model)
    plots/        gráficas y mapas de desempeño
    evaluacion/   tablas, métricas, predicciones y reportes
    data/         manifiesto de la ingesta (versión de datos)
    config/       hiperparámetros efectivos y archivos de configuración del proyecto

El dataset de entrenamiento se registra además con ``mlflow.log_input`` (solo metadatos:
esquema, número de filas y digest). Ningún artefacto contiene datos personales de usuarios:
la ingesta elimina el pseudónimo, el género y el año de nacimiento.
"""

import json
from pathlib import Path

import mlflow
import mlflow.data
import pandas as pd

from mibici.config import MANIFEST_PATH, PROJ_ROOT

ETAPAS = ("baseline", "search", "candidate")
ARCHIVOS_CONFIG = ("mibici/config.py", "MLproject", "python_env.yaml", "requirements.txt")
CARPETA_POR_EXTENSION = {".png": "plots", ".html": "plots"}


def tags_estandar(question: str, model_family: str, stage_of_work: str) -> dict:
    """Tags para filtrar y comparar en la UI: question, model_family y stage_of_work."""
    if stage_of_work not in ETAPAS:
        raise ValueError(f"stage_of_work debe ser uno de {ETAPAS}")
    return {"question": question, "model_family": model_family, "stage_of_work": stage_of_work}


def registrar_artefactos(rutas) -> None:
    """Sube cada archivo a plots/ (gráficas y mapas) o a evaluacion/ (tablas y reportes)."""
    for ruta in rutas:
        ruta = Path(ruta)
        carpeta = CARPETA_POR_EXTENSION.get(ruta.suffix.lower(), "evaluacion")
        mlflow.log_artifact(str(ruta), artifact_path=carpeta)


def registrar_config(params_efectivos: dict, extra: dict | None = None) -> None:
    """config/: hiperparámetros efectivos de la corrida y archivos de configuración."""
    contenido = {"params_efectivos": params_efectivos, **(extra or {})}
    mlflow.log_text(
        json.dumps(contenido, indent=2, ensure_ascii=False, default=str),
        "config/params_efectivos.json",
    )
    for relativa in ARCHIVOS_CONFIG:
        ruta = PROJ_ROOT / relativa
        if ruta.exists():
            mlflow.log_artifact(str(ruta), artifact_path="config")


def registrar_datos(x_train: pd.DataFrame, nombre: str, fuente: Path) -> None:
    """data/: manifiesto de la ingesta y dataset de entrenamiento con log_input."""
    mlflow.log_artifact(str(MANIFEST_PATH), artifact_path="data")
    dataset = mlflow.data.from_pandas(
        x_train.reset_index(drop=True), source=Path(fuente).resolve().as_uri(), name=nombre
    )
    mlflow.log_input(dataset, context="training")
