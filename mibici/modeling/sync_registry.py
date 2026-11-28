"""Entry point `sync_registry`: corrige estado y descripción de las versiones ya registradas.

A diferencia de `register`, no busca corridas ni mueve aliases: solo lee los aliases que ya
tiene cada registered model y, a partir de ellos, deja en cada versión su estado_produccion,
el tag run_stage_of_work, el resultado del A/B (si participó) y una descripción uniforme.
Sirve para poner al día un registry creado con una versión anterior del código sin tener
que volver a entrenar ni a registrar nada.

Al final imprime una tabla en Markdown con todas las versiones, lista para el README.

Uso::

    mlflow run . -e sync_registry --env-manager local
    python -m mibici.modeling.sync_registry --tracking-uri sqlite:///mlflow.db
"""

import mlflow
from mlflow import MlflowClient
import typer

from mibici.config import MLFLOW_TRACKING_URI, REGISTRO_MODELOS
from mibici.modeling.mlflow_utils import consola_utf8
from mibici.modeling.registry_estado import sincronizar_versiones

app = typer.Typer()

COLUMNAS = ("modelo", "version", "alias", "estado", "corrida", "data_version", "metrica")


def tabla_markdown(filas: list[dict]) -> str:
    encabezado = "| " + " | ".join(COLUMNAS) + " |"
    separador = "|" + "---|" * len(COLUMNAS)
    cuerpo = [
        "| " + " | ".join(f"`{f[c]}`" if c == "modelo" else str(f[c]) for c in COLUMNAS) + " |"
        for f in filas
    ]
    return "\n".join([encabezado, separador, *cuerpo])


@app.command()
def main(
    question: str = typer.Option("all", help="p1, p2, p3 o all."),
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI, help="URI del tracking/registry."),
):
    consola_utf8()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    preguntas = list(REGISTRO_MODELOS) if question == "all" else [question.lower()]
    filas = []
    for q in preguntas:
        filas.extend(sincronizar_versiones(client, REGISTRO_MODELOS[q]["nombre"], q))
    print()
    print(tabla_markdown(filas))


if __name__ == "__main__":
    app()
