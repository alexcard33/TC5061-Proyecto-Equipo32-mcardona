"""Copia el Model Registry local al servidor de MLflow que levanta Docker Compose.

El MLflow de Compose arranca con un volumen vacío, así que dentro del sistema orquestado no
se veía ningún modelo y la API no podía cargar su modelo con MODEL_SOURCE=registry. Este
script toma los registered models de REGISTRO_MODELOS desde el registry de origen (por
defecto la base local sqlite:///mlflow.db) y los recrea en el de destino:

1. Por cada versión de origen, en orden, crea en el destino una corrida con el mismo nombre,
   los mismos parámetros, métricas y tags, y le sube los artefactos del modelo.
2. Registra esa corrida como versión del mismo registered model, con los tags y la
   descripción de la versión original. Como se recorren en orden y el destino empieza
   vacío, los números de versión coinciden con los del origen.
3. Copia la descripción y los tags del registered model y asigna los mismos aliases.

Se puede correr varias veces: una versión que ya se copió (tag version_origen) no se vuelve
a crear, y los aliases y tags se vuelven a dejar igual que en el origen.

Las corridas del destino guardan en el tag run_id_origen el id de la corrida original, y las
versiones conservan source_run_id, así que el linaje se puede seguir hasta el registry local.

Uso (con el sistema arriba: docker compose up -d)::

    python -m mibici.modeling.copiar_registry
    python -m mibici.modeling.copiar_registry --origen sqlite:///mlflow.db \\
        --destino http://127.0.0.1:5000 --question p1
"""

from pathlib import Path
import tempfile
import time

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric, Param
import typer

from mibici.config import MLFLOW_EXPERIMENT, REGISTRO_MODELOS
from mibici.modeling.mlflow_utils import consola_utf8

app = typer.Typer()

TAG_VERSION_ORIGEN = "version_origen"
TAG_RUN_ORIGEN = "run_id_origen"
TAG_REGISTRY_ORIGEN = "registry_origen"
CARPETA_MODELO = "modelo"


def tags_de_corrida(tags: dict, run_id_origen: str, origen: str) -> dict:
    """Tags de la corrida de origen que se copian al destino.

    Los tags internos de MLflow (mlflow.*) se omiten: el servidor de destino pone los suyos y
    algunos, como el historial de modelos registrados, apuntan a rutas del origen.
    """
    copiados = {k: v for k, v in tags.items() if not k.startswith("mlflow.")}
    copiados[TAG_RUN_ORIGEN] = run_id_origen
    copiados[TAG_REGISTRY_ORIGEN] = origen
    return copiados


def versiones_copiadas(cliente: MlflowClient, nombre: str) -> dict:
    """Versiones que ya existen en el destino, indexadas por su versión de origen."""
    copiadas = {}
    for version in cliente.search_model_versions(f"name = '{nombre}'"):
        origen = (version.tags or {}).get(TAG_VERSION_ORIGEN)
        if origen:
            copiadas[origen] = version
    return copiadas


def asegurar_registered_model(destino: MlflowClient, modelo_origen) -> None:
    if not destino.search_registered_models(f"name = '{modelo_origen.name}'"):
        destino.create_registered_model(modelo_origen.name)
    destino.update_registered_model(
        modelo_origen.name, description=modelo_origen.description or ""
    )
    for clave, valor in (modelo_origen.tags or {}).items():
        destino.set_registered_model_tag(modelo_origen.name, clave, valor)


def copiar_corrida(
    origen: MlflowClient,
    destino: MlflowClient,
    run_id: str,
    uri_origen: str,
    experiment_id: str,
    carpeta_modelo: Path,
) -> str:
    """Recrea en el destino la corrida de origen con sus datos y el modelo. Regresa su id."""
    run = origen.get_run(run_id)
    nueva = destino.create_run(
        experiment_id,
        run_name=run.info.run_name,
        tags=tags_de_corrida(run.data.tags, run_id, uri_origen),
    )
    ahora = int(time.time() * 1000)
    params = [Param(k, v) for k, v in run.data.params.items()]
    metricas = [Metric(k, v, ahora, 0) for k, v in run.data.metrics.items()]
    # log_batch acepta hasta 100 parámetros y 1000 métricas por llamada.
    for i in range(0, len(params), 100):
        destino.log_batch(nueva.info.run_id, params=params[i : i + 100])
    for i in range(0, len(metricas), 1000):
        destino.log_batch(nueva.info.run_id, metrics=metricas[i : i + 1000])
    destino.log_artifacts(nueva.info.run_id, str(carpeta_modelo), artifact_path=CARPETA_MODELO)
    destino.set_terminated(nueva.info.run_id)
    return nueva.info.run_id


def copiar_modelo(
    origen: MlflowClient, destino: MlflowClient, uri_origen: str, nombre: str, experiment_id: str
) -> None:
    modelo = origen.get_registered_model(nombre)
    asegurar_registered_model(destino, modelo)
    ya_copiadas = versiones_copiadas(destino, nombre)
    numero_en_destino = {}

    for version in sorted(
        origen.search_model_versions(f"name = '{nombre}'"), key=lambda v: int(v.version)
    ):
        numero = str(version.version)
        if numero in ya_copiadas:
            nueva = ya_copiadas[numero]
            logger.info(f"{nombre} v{numero}: ya estaba copiada (v{nueva.version} en destino)")
        else:
            with tempfile.TemporaryDirectory() as carpeta:
                # La descarga por models:/ resuelve el registry con la URI global, así que se
                # apunta al origen de forma explícita antes de bajar los artefactos.
                mlflow.set_tracking_uri(uri_origen)
                mlflow.set_registry_uri(uri_origen)
                ruta = mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{nombre}/{numero}", dst_path=carpeta
                )
                run_id = copiar_corrida(
                    origen, destino, version.run_id, uri_origen, experiment_id, Path(ruta)
                )
            fuente = f"{destino.get_run(run_id).info.artifact_uri}/{CARPETA_MODELO}"
            nueva = destino.create_model_version(
                nombre,
                source=fuente,
                run_id=run_id,
                description=version.description or "",
            )
            logger.info(f"{nombre} v{numero}: copiada como v{nueva.version}")

        tags = dict(version.tags or {})
        tags[TAG_VERSION_ORIGEN] = numero
        for clave, valor in tags.items():
            destino.set_model_version_tag(nombre, nueva.version, clave, valor)
        destino.update_model_version(nombre, nueva.version, description=version.description or "")
        numero_en_destino[numero] = str(nueva.version)
        if str(nueva.version) != numero:
            logger.warning(
                f"{nombre}: la v{numero} quedó como v{nueva.version} en el destino. Pasa cuando "
                "el destino ya tenía versiones creadas a mano; con un volumen limpio coinciden."
            )

    for alias, numero in (modelo.aliases or {}).items():
        destino.set_registered_model_alias(nombre, alias, numero_en_destino[str(numero)])
        logger.info(f"{nombre}: @{alias} -> v{numero_en_destino[str(numero)]}")


@app.command()
def main(
    origen: str = typer.Option("sqlite:///mlflow.db", help="Registry de donde se copia."),
    destino: str = typer.Option("http://127.0.0.1:5000", help="MLflow de Docker Compose."),
    question: str = typer.Option("all", help="p1, p2, p3 o all."),
):
    consola_utf8()
    if origen == destino:
        raise typer.BadParameter("El origen y el destino son el mismo registry.")
    cliente_origen = MlflowClient(tracking_uri=origen, registry_uri=origen)
    cliente_destino = MlflowClient(tracking_uri=destino, registry_uri=destino)

    experimento = cliente_destino.get_experiment_by_name(MLFLOW_EXPERIMENT)
    experiment_id = (
        experimento.experiment_id
        if experimento
        else cliente_destino.create_experiment(MLFLOW_EXPERIMENT)
    )

    preguntas = list(REGISTRO_MODELOS) if question == "all" else [question.lower()]
    for q in preguntas:
        copiar_modelo(
            cliente_origen, cliente_destino, origen, REGISTRO_MODELOS[q]["nombre"], experiment_id
        )

    # Comprobación final: el modelo que sirve la API carga desde el registry de destino.
    nombre_p1 = REGISTRO_MODELOS["p1"]["nombre"]
    if "p1" in preguntas:
        mlflow.set_tracking_uri(destino)
        mlflow.set_registry_uri(destino)
        mlflow.pyfunc.load_model(f"models:/{nombre_p1}@champion")
        logger.success(f"Verificado: models:/{nombre_p1}@champion carga desde {destino}")


if __name__ == "__main__":
    app()
