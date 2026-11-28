"""Entry point `register`: lleva el modelo elegido de cada pregunta al Model Registry.

Para cada pregunta definida en REGISTRO_MODELOS (config.py):

1. Crea (o actualiza) el registered model con nombre y descripción.
2. Registra como nueva versión el modelo de la corrida elegida (runs:/<run_id>/modelo),
   salvo que ya exista una versión de esa corrida (el comando es idempotente).
3. Copia a la versión las tags de linaje: run de origen, experimento, commit, versión de datos
   y la etapa del experimento en que se hizo la corrida (run_stage_of_work).
4. Asigna el alias champion (producción) y, si se definió, challenger.
5. Recorre todas las versiones y deja coherentes su alias, su estado_produccion
   (produccion, en_evaluacion o retirado) y su descripción (ver registry_estado.py).
6. Verifica que el modelo en producción carga por nombre y alias (models:/<nombre>@champion).

Uso::

    mlflow run . -e register --env-manager local
    mlflow run . -e register -P question=p2 --env-manager local
"""

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import typer

from mibici.config import (
    ALIAS_CHALLENGER,
    ALIAS_PRODUCCION,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    REGISTRO_MODELOS,
    TEAM_ID,
)
from mibici.modeling.registry_estado import TAG_ETAPA_CORRIDA, sincronizar_versiones

app = typer.Typer()

# Tags que se copian de la corrida a la versión: nombre en la corrida -> nombre en la versión.
# stage_of_work cambia de nombre porque en la versión se leía como si fuera su estado, cuando
# en realidad es la etapa del experimento en que se hizo la corrida.
TAGS_LINAJE = {
    "question": "question",
    "model_family": "model_family",
    "stage_of_work": TAG_ETAPA_CORRIDA,
    "git_commit": "git_commit",
    "git_sucio": "git_sucio",
    "data_version": "data_version",
}


def buscar_run(client: MlflowClient, experiment_id: str, question: str, run_name: str):
    """Corrida terminada más reciente con ese run_name y esa pregunta."""
    runs = client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.question = '{question}' and attributes.run_name = '{run_name}' "
            "and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"No hay una corrida FINISHED de {question} con run_name={run_name!r} en "
            f"{MLFLOW_EXPERIMENT}."
        )
    return runs[0]


def asegurar_registered_model(client: MlflowClient, question: str, cfg: dict) -> None:
    try:
        client.create_registered_model(
            cfg["nombre"],
            description=cfg["descripcion"],
            tags={"question": question, "equipo": TEAM_ID, "lifecycle_scheme": "aliases"},
        )
        logger.info(f"Registered model creado: {cfg['nombre']}")
    except MlflowException:
        client.update_registered_model(cfg["nombre"], description=cfg["descripcion"])


def version_de_run(client: MlflowClient, nombre: str, run):
    """Versión del registered model que viene de `run`; la crea si todavía no existe."""
    run_id = run.info.run_id
    existentes = client.search_model_versions(f"name = '{nombre}' and run_id = '{run_id}'")
    if existentes:
        return existentes[0]
    return mlflow.register_model(f"runs:/{run_id}/modelo", nombre)


def registrar_pregunta(client: MlflowClient, experiment_id: str, question: str) -> None:
    cfg = REGISTRO_MODELOS[question]
    if not cfg.get("champion"):
        raise ValueError(
            f"Falta elegir el modelo de {question}: complete REGISTRO_MODELOS['{question}']"
            "['champion'] en config.py con el run_name de la corrida elegida."
        )
    asegurar_registered_model(client, question, cfg)

    for alias, run_name in (
        (ALIAS_PRODUCCION, cfg["champion"]),
        (ALIAS_CHALLENGER, cfg.get("challenger")),
    ):
        if not run_name:
            continue
        run = buscar_run(client, experiment_id, question, run_name)
        version = version_de_run(client, cfg["nombre"], run)
        tags = {
            destino: run.data.tags[origen]
            for origen, destino in TAGS_LINAJE.items()
            if origen in run.data.tags
        }
        tags.update(
            {
                "source_run_id": run.info.run_id,
                "source_run_name": run_name,
                "source_experiment": MLFLOW_EXPERIMENT,
                "lifecycle_scheme": "aliases",
            }
        )
        if "umbral_decision" in run.data.params:
            tags["umbral_decision"] = run.data.params["umbral_decision"]
        for clave, valor in tags.items():
            client.set_model_version_tag(cfg["nombre"], version.version, clave, str(valor))
        client.set_registered_model_alias(cfg["nombre"], alias, version.version)
        logger.info(f"{cfg['nombre']} v{version.version} <- @{alias} ({run_name})")

    # Con los aliases ya en su lugar, se actualizan estado y descripción de todas las
    # versiones, incluidas las que se quedaron sin alias.
    sincronizar_versiones(client, cfg["nombre"], question)

    uri = f"models:/{cfg['nombre']}@{ALIAS_PRODUCCION}"
    modelo = mlflow.pyfunc.load_model(uri)
    logger.success(f"Verificado: {uri} carga como {type(modelo).__name__}")


@app.command()
def main(question: str = typer.Option("all", help="p1, p2, p3 o all.")):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    experimento = client.get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experimento is None:
        raise ValueError(f"No existe el experimento {MLFLOW_EXPERIMENT}.")
    preguntas = list(REGISTRO_MODELOS) if question == "all" else [question.lower()]
    for q in preguntas:
        registrar_pregunta(client, experimento.experiment_id, q)


if __name__ == "__main__":
    app()
