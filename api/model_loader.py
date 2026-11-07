"""Carga del modelo que sirve la API.

Dos modos, elegidos con la variable de entorno ``MODEL_SOURCE``

"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient

DEFAULT_MODEL_NAME = "mibici-equipo32-p1-demanda"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "serving" / "p1"


@dataclass
class ModeloServido:
    modelo: mlflow.pyfunc.PyFuncModel
    nombre: str
    version: str
    alias: str | None
    run_id: str | None
    model_uri: str
    fuente: str
    columnas: list[str] = field(default_factory=list)


def _desde_local(model_dir: Path) -> ModeloServido:
    modelo = mlflow.pyfunc.load_model(str(model_dir))
    info_path = model_dir / "model_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    return ModeloServido(
        modelo=modelo,
        nombre=info.get("registered_model", DEFAULT_MODEL_NAME),
        version=str(info.get("version", "desconocida")),
        alias=info.get("alias"),
        run_id=info.get("run_id") or modelo.metadata.run_id,
        model_uri=info.get("model_uri", str(model_dir)),
        fuente="local",
        columnas=modelo.metadata.get_input_schema().input_names(),
    )


def _desde_registry(nombre: str, alias: str, version: str | None, tracking_uri: str) -> ModeloServido:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    if version:
        mv = client.get_model_version(nombre, version)
        uri = f"models:/{nombre}/{mv.version}"
    else:
        mv = client.get_model_version_by_alias(nombre, alias)
        uri = f"models:/{nombre}@{alias}"
    modelo = mlflow.pyfunc.load_model(uri)
    return ModeloServido(
        modelo=modelo,
        nombre=nombre,
        version=str(mv.version),
        alias=None if version else alias,
        run_id=mv.run_id,
        model_uri=uri,
        fuente="registry",
        columnas=modelo.metadata.get_input_schema().input_names(),
    )


def cargar_modelo() -> ModeloServido:
    """Carga el modelo según las variables de entorno (ver docstring del módulo)."""
    fuente = os.getenv("MODEL_SOURCE", "local").lower()
    if fuente == "registry":
        return _desde_registry(
            nombre=os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME),
            alias=os.getenv("MODEL_ALIAS", "champion"),
            version=os.getenv("MODEL_VERSION") or None,
            tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://host.containers.internal:5000"),
        )
    if fuente != "local":
        raise ValueError(f"MODEL_SOURCE inválido: {fuente!r} (use 'local' o 'registry').")
    return _desde_local(Path(os.getenv("MODEL_DIR", str(DEFAULT_MODEL_DIR))))
