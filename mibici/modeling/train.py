"""Entry point `train` del MLflow Project: entrena un modelo de P1, P2 o P3.

Ejemplos (con el servidor de tracking levantado y MLFLOW_TRACKING_URI definido)::

    mlflow run . -e train -P question=p2 -P model=hgb \\
        --experiment-name mibici-equipo32 --env-manager local

    mlflow run . -e train -P question=p1 -P model=hgb \\
        -P params=learning_rate=0.05,max_iter=500 \\
        --experiment-name mibici-equipo32 --env-manager local

Modelos disponibles:

    p1: baseline (naive semanal), poisson, hgb, rf, search (malla HGB con nested runs)
    p2: baseline (regresión logística), hgb, rf, search (malla HGB con nested runs)
    p3: baseline (K-Means k=2), kmeans, agglomerative, gmm, search (k de K-Means con nested runs)

Los hiperparámetros por defecto viven en config.py; `params` solo indica los que cambian, como
pares clave=valor separados por comas. Los valores se interpretan como JSON cuando es posible
(números, true/false, null) y si no como texto (p. ej. linkage=average). También se acepta un
objeto JSON al llamar el módulo directamente.
"""

import importlib
import json

from loguru import logger
import typer

from mibici.config import MODELO_DEFAULT

MODULOS = {
    "p1": "mibici.modeling.p1_regresion",
    "p2": "mibici.modeling.p2_clasificacion",
    "p3": "mibici.modeling.p3_clustering",
}

app = typer.Typer()


def _valor(texto: str):
    """Interpreta un valor como JSON (número, booleano, null) o, si no se puede, como texto."""
    alias = {"true": True, "false": False, "none": None, "null": None}
    if texto.lower() in alias:
        return alias[texto.lower()]
    try:
        return json.loads(texto)
    except json.JSONDecodeError:
        return texto


def parsear_params(texto: str) -> dict:
    """Convierte 'a=1,b=0.5,c=ward' (o un objeto JSON) en un diccionario de hiperparámetros."""
    # cmd de Windows puede dejar comillas o ^ alrededor del valor recibido de mlflow run.
    texto = texto.strip().strip("^").strip("'\"").strip()
    if texto.lower() in ("", "none", "{}"):
        return {}
    if texto.startswith("{"):
        params = json.loads(texto)
        if not isinstance(params, dict):
            raise typer.BadParameter("params en JSON debe ser un objeto.")
        return params
    params = {}
    for par in texto.split(","):
        if "=" not in par:
            raise typer.BadParameter(f"Se esperaba clave=valor y llegó {par!r}.")
        clave, valor = par.split("=", 1)
        params[clave.strip()] = _valor(valor.strip())
    return params


@app.command()
def main(
    question: str = typer.Option(..., help="Pregunta a entrenar: p1, p2 o p3."),
    model: str = typer.Option(
        "default", help="Modelo de la pregunta; 'default' toma el de config.py y 'all' entrena todos."
    ),
    params: str = typer.Option(
        "none", help="Hiperparámetros clave=valor separados por comas; se combinan con config.py."
    ),
    run_name: str = typer.Option("auto", help="Nombre del run; 'auto' usa el de config.py."),
):
    question = question.strip().lower()
    if question not in MODULOS:
        raise typer.BadParameter(f"question debe ser una de: {', '.join(MODULOS)}")

    params = parsear_params(params)

    modelo = MODELO_DEFAULT[question] if model == "default" else model
    logger.info(f"train | question={question} model={modelo} params={params}")

    modulo = importlib.import_module(MODULOS[question])
    modulo.ejecutar(
        model=modelo,
        params=params,
        run_name=None if run_name in ("", "auto") else run_name,
    )


if __name__ == "__main__":
    app()
