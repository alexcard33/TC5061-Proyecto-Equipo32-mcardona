"""Versión 2 de P1: el mismo modelo, reentrenado con datos de 2026.

Es el caso típico de Continuous Training. Se usa el mismo algoritmo y los mismos
hiperparámetros que la versión vigente (regresión Poisson lineal, alpha 0.001), pero con una
ventana de entrenamiento que crece hasta junio de 2026. Así el modelo conoce las estaciones
que abrieron en 2026 y el nivel de demanda reciente, y cualquier diferencia contra la versión
vigente se debe solo a los datos.

    entrenamiento     enero 2025 a junio 2026
    validación        julio 2026
    holdout del A/B   agosto a octubre 2026 (aquí no se usa)

El run queda como p1_v2_reentrenado en el experimento mibici-equipo32. Después, el entry
point register lo agrega como nueva versión de mibici-equipo32-p1-demanda con el alias
challenger, porque así está en REGISTRO_MODELOS (config.py):

    python -m mibici.modeling.p1_reentrenamiento
    python -m mibici.modeling.register --question p1
"""

import json

from loguru import logger
import mlflow
from mlflow.models import infer_signature
import pandas as pd
import typer

from mibici.config import (
    AB_EVAL_END,
    AB_EVAL_START,
    ESTACIONES_ACTUAL_PARQUET,
    ESTACIONES_PARQUET,
    MANIFEST_ACTUAL_PATH,
    MANIFEST_PATH,
    MLFLOW_TRACKING_URI,
    P1_HORIZONTE_DIAS,
    P1_LAGS_DIAS,
    P1_MLFLOW_EXPERIMENT,
    P1_MLFLOW_V2_RUN_NAME,
    P1_VENTANAS_MEDIA_DIAS,
    REENTRENAMIENTO_DIR,
    SEED,
    V2_TRAIN_START,
    V2_VAL_START,
    VIAJES_ACTUAL_PARQUET,
)
from mibici.features import P1_FEATURES
from mibici.modeling.mlflow_utils import (
    consola_utf8,
    obtener_git_commit,
    registrar_artefactos,
    registrar_config,
    registrar_datos,
)
from mibici.modeling.p1_regresion import SKOPS_TRUSTED, construir_modelo, evaluar, tags_run
from mibici.panel_p1 import cargar_capacidad, cargar_viajes_p1, construir_features, cortar

app = typer.Typer()

# El baseline no se reentrena: no aprende nada de los datos.
MODELOS_V2 = ("poisson", "hgb", "rf")


def preparar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conjuntos de entrenamiento y validación de la versión 2.

    El panel se corta en AB_EVAL_START, así que los meses del A/B ni siquiera entran al
    cálculo de las features. Igual que en la versión vigente, se quitan las filas que no
    tienen historia suficiente.
    """
    viajes = cargar_viajes_p1()
    df = construir_features(
        viajes,
        cargar_capacidad(ESTACIONES_PARQUET),
        cargar_capacidad(ESTACIONES_ACTUAL_PARQUET),
        fin=AB_EVAL_START,
    )
    train = cortar(df, V2_TRAIN_START, V2_VAL_START).dropna(subset=P1_FEATURES)
    val = cortar(df, V2_VAL_START, AB_EVAL_START).dropna(subset=P1_FEATURES)
    return train, val


def version_de_datos() -> str:
    """La versión 2 usa datos de 2025 y de 2026, así que su versión une ambos manifiestos."""
    ref = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["data_version"]
    actual = json.loads(MANIFEST_ACTUAL_PATH.read_text(encoding="utf-8"))["data_version"]
    return f"{ref}+{actual}"


def entrenar(
    model: str = "poisson",
    run_name: str = P1_MLFLOW_V2_RUN_NAME,
    tracking_uri: str = MLFLOW_TRACKING_URI,
) -> dict:
    """Entrena la versión 2, la evalúa en julio de 2026 y registra el run en MLflow."""
    if model not in MODELOS_V2:
        raise ValueError(f"Modelo no válido para la versión 2: {model!r}. Opciones: {MODELOS_V2}")
    modelo, etapa, familia, _, params_log = construir_modelo(model, {})

    train, val = preparar_datos()
    x_train, y_train = train[P1_FEATURES].astype("float64"), train["viajes"]
    x_val, y_val = val[P1_FEATURES].astype("float64"), val["viajes"]
    estaciones_train = train.index.get_level_values("estacion").nunique()
    logger.info(
        f"Versión 2 ({model}): train {len(train):,} filas y {estaciones_train} estaciones, "
        f"val {len(val):,} filas"
    )

    git_commit, git_sucio = obtener_git_commit()
    if git_sucio:
        logger.warning("Hay cambios sin commit: el run no será reproducible desde git_commit")
    contexto = {
        "git_commit": git_commit,
        "git_sucio": git_sucio,
        "data_version": version_de_datos(),
    }

    modelo.fit(x_train, y_train)
    metricas = {
        **evaluar(y_train, modelo.predict(x_train), "train"),
        **evaluar(y_val, modelo.predict(x_val), "val"),
    }
    REENTRENAMIENTO_DIR.mkdir(parents=True, exist_ok=True)
    ruta_metricas = REENTRENAMIENTO_DIR / f"{run_name}_metricas.json"
    ruta_metricas.write_text(json.dumps(metricas, indent=2), encoding="utf-8")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(P1_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                **tags_run(familia, etapa, contexto),
                "version_modelo": "v2",
                "tipo_entrenamiento": "reentrenamiento con datos de 2026",
            }
        )
        mlflow.log_params(
            {
                **params_log,
                "objetivo": "viajes iniciados por estacion y dia",
                "horizonte_dias": P1_HORIZONTE_DIAS,
                "lags_dias": str(P1_LAGS_DIAS),
                "ventanas_media_dias": str(P1_VENTANAS_MEDIA_DIAS),
                "features": len(P1_FEATURES),
                "train_inicio": V2_TRAIN_START,
                "val_inicio": V2_VAL_START,
                "val_fin": AB_EVAL_START,
                "holdout_ab_inicio": AB_EVAL_START,
                "holdout_ab_fin": AB_EVAL_END,
                "random_state": SEED,
                "data_version": contexto["data_version"],
                "filas_train": len(train),
                "filas_val": len(val),
                "estaciones_train": estaciones_train,
            }
        )
        mlflow.log_metrics(metricas)
        ejemplo = x_val.head(5)
        mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            input_example=ejemplo,
            skops_trusted_types=SKOPS_TRUSTED,
        )
        registrar_artefactos([ruta_metricas])
        registrar_config(params_log)
        registrar_datos(x_train, "p1_v2_train_estacion_dia", VIAJES_ACTUAL_PARQUET)
        mlflow.log_artifact(str(MANIFEST_ACTUAL_PATH), artifact_path="data")

    logger.success(
        f"{run_name} ({run.info.run_id}): val_mae={metricas['val_mae']:.3f} "
        f"val_wape={metricas['val_wape']:.3f}. Para registrarlo como challenger: "
        "python -m mibici.modeling.register --question p1"
    )
    return {"run_id": run.info.run_id, "metricas": metricas}


@app.command()
def main(
    model: str = typer.Option("poisson", help="poisson (igual que la versión vigente), hgb o rf."),
    run_name: str = typer.Option(P1_MLFLOW_V2_RUN_NAME, help="Nombre del run en MLflow."),
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI, help="URI del servidor de tracking."),
):
    consola_utf8()
    entrenar(model, run_name, tracking_uri)


if __name__ == "__main__":
    app()
