"""Entry point `export`: descarga del Model Registry el modelo que servirá la API.

Recupera el modelo por nombre y alias (o versión) desde el Model Registry de MLflow y lo deja
en ``models/serving/<pregunta>/`` junto con ``model_info.json`` (nombre, versión, alias, run de
origen y tags de linaje). El Containerfile copia esa carpeta a la imagen, de modo que el
contenedor no depende de la ruta local de los artefactos (``file:///C:/...``) ni de que el
servidor de MLflow esté encendido.

Uso::

    # con el servidor de MLflow encendido (make mlflow-ui)
    python -m mibici.modeling.export_model
    # o directamente contra la base SQLite del registry
    python -m mibici.modeling.export_model --tracking-uri sqlite:///mlflow.db
    # una versión concreta en lugar del alias
    python -m mibici.modeling.export_model --version 1
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from loguru import logger
import mlflow
from mlflow import MlflowClient
import typer

from mibici.config import (
    MLFLOW_TRACKING_URI,
    REGISTRO_MODELOS,
    SERVING_ALIAS,
    SERVING_DIR,
    SERVING_QUESTION,
)

app = typer.Typer()


def exportar(
    question: str = SERVING_QUESTION,
    alias: str = SERVING_ALIAS,
    version: str | None = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    destino: Path | None = None,
) -> Path:
    """Descarga ``models:/<nombre>@<alias>`` (o ``/<version>``) y escribe model_info.json."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    nombre = REGISTRO_MODELOS[question]["nombre"]

    if version:
        mv = client.get_model_version(nombre, str(version))
        uri = f"models:/{nombre}/{mv.version}"
    else:
        mv = client.get_model_version_by_alias(nombre, alias)
        uri = f"models:/{nombre}@{alias}"
    logger.info(f"Recuperando {uri} (versión {mv.version}, run {mv.run_id})")

    destino = Path(destino or SERVING_DIR / question)
    if destino.exists():
        shutil.rmtree(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=str(destino.parent / f".tmp_{question}")))
    shutil.move(str(tmp), str(destino))
    shutil.rmtree(destino.parent / f".tmp_{question}", ignore_errors=True)

    # Verificación: el modelo exportado carga como pyfunc.
    modelo = mlflow.pyfunc.load_model(str(destino))
    columnas = modelo.metadata.get_input_schema().input_names()

    info = {
        "question": question,
        "registered_model": nombre,
        "version": str(mv.version),
        "alias": None if version else alias,
        "model_uri": uri,
        "run_id": mv.run_id,
        "aliases": list(getattr(mv, "aliases", []) or []),
        "tags": dict(mv.tags or {}),
        "input_columns": columnas,
        "tracking_uri": tracking_uri,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (destino / "model_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Modelo {nombre} v{mv.version} exportado a {destino}")
    return destino


@app.command()
def main(
    question: str = typer.Option(SERVING_QUESTION, help="p1, p2 o p3."),
    alias: str = typer.Option(SERVING_ALIAS, help="Alias del registry (champion/challenger)."),
    version: str = typer.Option("", help="Versión concreta; si se indica, ignora el alias."),
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI, help="URI del tracking/registry."),
):
    exportar(question.lower(), alias, version or None, tracking_uri)


if __name__ == "__main__":
    app()
