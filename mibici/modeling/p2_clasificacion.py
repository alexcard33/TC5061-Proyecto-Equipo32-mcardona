"""P2: clasificación de estaciones MiBici en riesgo de quedarse sin bicis o sin anclajes."""

import itertools
import json
from pathlib import Path
import subprocess

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mibici.config import (
    ESTACIONES_PARQUET,
    MANIFEST_PATH,
    MLFLOW_TRACKING_URI,
    P2_FRACCION_CAPACIDAD,
    P2_HGB_PARAMS,
    P2_HGB_SEARCH_GRID,
    P2_HORIZONTE_HORAS,
    P2_LOGREG_PARAMS,
    P2_MAX_DURATION_SECONDS,
    P2_MIN_DURATION_SECONDS,
    P2_MLFLOW_BASELINE_RUN_NAME,
    P2_MLFLOW_CANDIDATE_RUN_NAME,
    P2_MLFLOW_EXPERIMENT,
    P2_MLFLOW_RF_RUN_NAME,
    P2_MLFLOW_SEARCH_RUN_NAME,
    P2_OUTPUT_DIR,
    P2_QUESTION,
    P2_RF_PARAMS,
    P2_TEST_END,
    P2_TEST_START,
    P2_TRAIN_START,
    P2_VAL_START,
    PROJ_ROOT,
    SEED,
    TEAM_ID,
    VIAJES_PARQUET,
)
from mibici.features import (
    P2_FEATURES,
    p2_agregar_etiqueta,
    p2_agregar_features,
    p2_ajustar_historico,
    p2_aplicar_historico,
    p2_construir_panel,
    p2_historico_pasado,
)
from mibici.modeling.mlflow_utils import (
    registrar_artefactos,
    registrar_config,
    registrar_datos,
    tags_estandar,
)

app = typer.Typer()

# MLflow 3 guarda los modelos de sklearn con skops y pide marcar como confiable este
# tipo interno de HistGradientBoosting (el modelo lo entrenamos nosotros).
SKOPS_TRUSTED = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.tree._tree.Tree",
]


def obtener_git_commit() -> tuple[str, bool]:
    """Commit actual y si hay cambios sin commitear (para los tags git_commit y git_sucio)."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
        ).strip()
        estado = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJ_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido", True
    return commit, bool(estado)


def cargar_viajes(ruta: Path) -> pd.DataFrame:
    """Lee los viajes y deja fuera los de menos de 1 minuto o más de 24 horas."""
    viajes = pd.read_parquet(
        ruta, columns=["inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
    )
    duracion = (viajes["fin_viaje"] - viajes["inicio_viaje"]).dt.total_seconds()
    return viajes.loc[duracion.between(P2_MIN_DURATION_SECONDS, P2_MAX_DURATION_SECONDS)].dropna()


def cargar_capacidad(ruta: Path) -> pd.Series:
    """Capacidad (dpcapacity) de cada estación, indexada por id."""
    estaciones = pd.read_parquet(ruta, columns=["id", "dpcapacity"])
    return estaciones.set_index("id")["dpcapacity"]


def cortar(df: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Filas de [inicio, fin) cuya ventana futura no se pasa de `fin`.

    Si no se cortan, la etiqueta de las últimas horas de un conjunto se calcularía con
    viajes del siguiente.
    """
    hora = df.index.get_level_values("hora")
    limite = pd.Timestamp(fin) - pd.Timedelta(hours=P2_HORIZONTE_HORAS)
    return df.loc[(hora >= pd.Timestamp(inicio)) & (hora < limite)]


def construir_baseline(params: dict) -> Pipeline:
    """Baseline: regresión logística con features escaladas y pesos para la clase rara."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("modelo", LogisticRegression(**params)),
        ]
    )


def construir_candidato(params: dict) -> HistGradientBoostingClassifier:
    """Candidato: HistGradientBoosting, también con class_weight balanceado."""
    return HistGradientBoostingClassifier(**{"random_state": SEED, **params})


def construir_rf(params: dict) -> RandomForestClassifier:
    """Candidato: bosque aleatorio con pesos balanceados por submuestra."""
    return RandomForestClassifier(**{"random_state": SEED, **params})


def elegir_umbral(y_val: pd.Series, prob_val) -> float:
    """Umbral que da el mejor F1 en validación.

    Con pesos balanceados el 0.5 no sirve (salen demasiadas alertas). Nunca se elige con test.
    """
    precision, recall, umbrales = precision_recall_curve(y_val, prob_val)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return float(umbrales[f1[:-1].argmax()])


def evaluar(y, prob, umbral: float, prefijo: str, tipo: pd.Series) -> dict:
    """Métricas de un conjunto con su prefijo (train_, val_ o test_).

    También va el recall por tipo de riesgo y la accuracy de decir siempre "normal", para
    que se vea por qué no usamos accuracy.
    """
    pred = (prob >= umbral).astype(int)
    recall_tipo = {
        f"{prefijo}_recall_{t}": float(pred[(tipo == t).to_numpy()].mean())
        for t in ("sin_bicis", "sin_anclajes")
    }
    return {
        f"{prefijo}_f1": f1_score(y, pred),
        f"{prefijo}_recall": recall_score(y, pred),
        f"{prefijo}_precision": precision_score(y, pred, zero_division=0),
        f"{prefijo}_pr_auc": average_precision_score(y, prob),
        f"{prefijo}_accuracy": accuracy_score(y, pred),
        f"{prefijo}_accuracy_trivial": 1 - y.mean(),  # como si siempre dijera "normal"
        f"{prefijo}_prevalencia": y.mean(),
        **recall_tipo,
    }


def guardar_graficas(nombre: str, y, prob, umbral: float) -> list[Path]:
    """Matriz de confusión, curva PR y classification_report de test, para subirlos a MLflow."""
    pred = (prob >= umbral).astype(int)

    ruta_cm = P2_OUTPUT_DIR / f"{nombre}_matriz_confusion.png"
    ConfusionMatrixDisplay.from_predictions(y, pred, display_labels=["normal", "riesgo"])
    plt.title(f"{nombre} · TEST")
    plt.savefig(ruta_cm, dpi=150, bbox_inches="tight")
    plt.close()

    ruta_pr = P2_OUTPUT_DIR / f"{nombre}_curva_pr.png"
    PrecisionRecallDisplay.from_predictions(y, prob, name=nombre)
    plt.savefig(ruta_pr, dpi=150, bbox_inches="tight")
    plt.close()

    ruta_reporte = P2_OUTPUT_DIR / f"{nombre}_reporte.txt"
    ruta_reporte.write_text(classification_report(y, pred, target_names=["normal", "riesgo"]))
    return [ruta_cm, ruta_pr, ruta_reporte]


def tags_run(familia: str, etapa: str, contexto: dict) -> dict:
    """Tags de equipo, commit y versión de datos más los estándar (question, family, stage)."""
    return {
        "equipo": TEAM_ID,
        "mlflow.user": TEAM_ID,
        "problema": "P2",
        "tipo": "clasificacion",
        "pregunta": P2_QUESTION,
        "git_commit": contexto["git_commit"],
        "git_sucio": str(contexto["git_sucio"]),
        "data_version": contexto["data_version"],
        **tags_estandar("p2", familia, etapa),
    }


def entrenar_y_registrar(nombre_run, etapa, familia, modelo, params_modelo, datos, contexto) -> dict:
    """Entrena, elige el umbral en validación, evalúa los tres conjuntos y sube todo a MLflow."""
    (x_train, y_train, t_train), (x_val, y_val, t_val), (x_test, y_test, t_test) = datos

    modelo.fit(x_train, y_train)
    umbral = elegir_umbral(y_val, modelo.predict_proba(x_val)[:, 1])
    prob_test = modelo.predict_proba(x_test)[:, 1]

    metricas = {
        **evaluar(y_train, modelo.predict_proba(x_train)[:, 1], umbral, "train", t_train),
        **evaluar(y_val, modelo.predict_proba(x_val)[:, 1], umbral, "val", t_val),
        **evaluar(y_test, prob_test, umbral, "test", t_test),
    }
    artefactos = guardar_graficas(nombre_run, y_test, prob_test, umbral)

    with mlflow.start_run(run_name=nombre_run):
        mlflow.set_tags(tags_run(familia, etapa, contexto))
        mlflow.log_params(
            {
                **params_modelo,
                "definicion_riesgo": "abs(neto t+1..t+H) >= fraccion * dpcapacity",
                "horizonte_horas": P2_HORIZONTE_HORAS,
                "fraccion_capacidad": P2_FRACCION_CAPACIDAD,
                "umbral_decision": round(umbral, 4),
                "features": len(P2_FEATURES),
                "train_inicio": P2_TRAIN_START,
                "val_inicio": P2_VAL_START,
                "test_inicio": P2_TEST_START,
                "test_fin": P2_TEST_END,
                "random_state": SEED,
                "data_version": contexto["data_version"],
                **contexto["params_datos"],
            }
        )
        mlflow.log_metrics(metricas)
        ejemplo = x_test.head(5)
        mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
            input_example=ejemplo,
            skops_trusted_types=SKOPS_TRUSTED,
        )
        registrar_artefactos(artefactos)
        registrar_config(params_modelo, {"umbral_decision": umbral})
        registrar_datos(x_train, "p2_train_estacion_hora", VIAJES_PARQUET)

    logger.info(
        f"{nombre_run}: test_f1={metricas['test_f1']:.3f} "
        f"test_recall={metricas['test_recall']:.3f} "
        f"test_precision={metricas['test_precision']:.3f}"
    )
    return metricas


MODELOS = ("baseline", "hgb", "rf")


def preparar_datos(viajes_path: Path = VIAJES_PARQUET):
    """Panel estación x hora, etiqueta, features y partición temporal."""
    P2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Panel estación x hora, etiqueta y features
    viajes = cargar_viajes(viajes_path)
    capacidad = cargar_capacidad(ESTACIONES_PARQUET)
    panel = p2_construir_panel(viajes, P2_TRAIN_START, P2_TEST_END)
    df = p2_agregar_etiqueta(panel, P2_HORIZONTE_HORAS, capacidad, P2_FRACCION_CAPACIDAD)
    df = p2_agregar_features(df, P2_HORIZONTE_HORAS)

    estaciones_viajes = panel.index.get_level_values("estacion").unique()
    estaciones_modelo = df.index.get_level_values("estacion").unique()
    excluidas = estaciones_viajes.difference(estaciones_modelo)
    logger.info(
        f"Estaciones con viajes={len(estaciones_viajes)} | excluidas por capacidad 0 o sin "
        f"catálogo={len(excluidas)}"
    )

    # 2. Partición temporal, sin revolver: TRAIN ene-sep, VAL oct, TEST nov-dic
    train = cortar(df, P2_TRAIN_START, P2_VAL_START)
    val = cortar(df, P2_VAL_START, P2_TEST_START)
    test = cortar(df, P2_TEST_START, P2_TEST_END)

    # en val/test solo entran estaciones que ya tenían viajes en TRAIN
    viajes_train = viajes.loc[viajes["inicio_viaje"] < pd.Timestamp(P2_VAL_START)]
    estaciones_train = pd.Index(viajes_train["origen_id"]).union(viajes_train["destino_id"])
    val = val.loc[val.index.get_level_values("estacion").isin(estaciones_train)]
    test = test.loc[test.index.get_level_values("estacion").isin(estaciones_train)]

    # 3. Perfil histórico, siempre sacado de TRAIN
    historico = p2_ajustar_historico(train)
    train = p2_historico_pasado(train)
    val, test = (p2_aplicar_historico(d, historico) for d in (val, test))

    # fuera filas sin lags completos (la primera semana de cada estación)
    filas_antes = {"train": len(train), "val": len(val), "test": len(test)}
    train, val, test = (d.dropna(subset=P2_FEATURES) for d in (train, val, test))
    for nombre, d in (("train", train), ("val", val), ("test", test)):
        logger.info(
            f"{nombre}: filas={len(d):,} (descartadas sin lags/histórico="
            f"{filas_antes[nombre] - len(d):,}) estaciones="
            f"{d.index.get_level_values('estacion').nunique()} "
            f"prevalencia={d['riesgo'].mean():.2%}"
        )

    datos = tuple(
        (d[P2_FEATURES].astype("float64"), d["riesgo"], d["tipo_riesgo"])
        for d in (train, val, test)
    )

    # 4. MLflow
    git_commit, git_sucio = obtener_git_commit()
    if git_sucio:
        logger.warning("Hay cambios sin commit: el run no será reproducible desde git_commit")
    contexto = {
        "git_commit": git_commit,
        "git_sucio": git_sucio,
        "data_version": json.loads(MANIFEST_PATH.read_text())["data_version"],
        "params_datos": {
            "estaciones_excluidas": len(excluidas),
            "estaciones_train": train.index.get_level_values("estacion").nunique(),
            "filas_train": len(train),
            "filas_val": len(val),
            "filas_test": len(test),
        },
    }
    return datos, contexto


def construir_modelo(nombre: str, params: dict):
    """Devuelve (modelo, etapa, familia, nombre de run por defecto, parámetros a registrar)."""
    if nombre == "baseline":
        p = {**P2_LOGREG_PARAMS, **params}
        params_log = {"modelo": "LogisticRegression", **p}
        return construir_baseline(p), "baseline", "linear", P2_MLFLOW_BASELINE_RUN_NAME, params_log
    if nombre == "hgb":
        p = {**P2_HGB_PARAMS, **params}
        params_log = {"modelo": "HistGradientBoostingClassifier", **p}
        return (
            construir_candidato(p), "candidate", "gradient_boosting",
            P2_MLFLOW_CANDIDATE_RUN_NAME, params_log,
        )
    if nombre == "rf":
        p = {**P2_RF_PARAMS, **params}
        params_log = {"modelo": "RandomForestClassifier", **p}
        return construir_rf(p), "candidate", "random_forest", P2_MLFLOW_RF_RUN_NAME, params_log
    raise ValueError(
        f"Modelo de P2 desconocido: {nombre!r}. Opciones: {', '.join(MODELOS)}, search o all."
    )


def ejecutar_busqueda(run_name, datos, contexto) -> dict:
    """Búsqueda en malla de HGB: un run padre y un run hijo (nested) por combinación.

    Cada combinación elige su umbral en VAL y se compara por val_f1 (desempate val_pr_auc).
    TEST no se calcula para no influir en la elección.
    """
    (x_train, y_train, t_train), (x_val, y_val, t_val), _ = datos
    claves = list(P2_HGB_SEARCH_GRID)
    combinaciones = [
        dict(zip(claves, valores)) for valores in itertools.product(*P2_HGB_SEARCH_GRID.values())
    ]
    resultados = []
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags_run("gradient_boosting", "search", contexto))
        mlflow.log_params(
            {
                "modelo": "HistGradientBoostingClassifier",
                "criterio": "max val_f1; desempate val_pr_auc",
                "combinaciones": len(combinaciones),
                "grid": json.dumps(P2_HGB_SEARCH_GRID),
                "data_version": contexto["data_version"],
            }
        )
        for i, combo in enumerate(combinaciones, start=1):
            params = {**P2_HGB_PARAMS, **combo}
            modelo = construir_candidato(params)
            modelo.fit(x_train, y_train)
            prob_val = modelo.predict_proba(x_val)[:, 1]
            umbral = elegir_umbral(y_val, prob_val)
            metricas = {
                **evaluar(y_train, modelo.predict_proba(x_train)[:, 1], umbral, "train", t_train),
                **evaluar(y_val, prob_val, umbral, "val", t_val),
            }
            with mlflow.start_run(run_name=f"{run_name}_{i:02d}", nested=True) as hijo:
                mlflow.set_tags(tags_run("gradient_boosting", "search", contexto))
                mlflow.log_params(
                    {"modelo": "HistGradientBoostingClassifier", **params, "umbral_decision": round(umbral, 4)}
                )
                mlflow.log_metrics(metricas)
            resultados.append({"run_id": hijo.info.run_id, **combo, **metricas})
            logger.info(
                f"{run_name}_{i:02d} {combo}: val_f1={metricas['val_f1']:.4f} "
                f"val_pr_auc={metricas['val_pr_auc']:.4f}"
            )

        tabla = (
            pd.DataFrame(resultados)
            .sort_values(["val_f1", "val_pr_auc"], ascending=False)
            .reset_index(drop=True)
        )
        mejor = tabla.iloc[0]
        mlflow.log_params({f"mejor_{k}": mejor[k] for k in claves})
        mlflow.log_metrics({"mejor_val_f1": mejor["val_f1"], "mejor_val_pr_auc": mejor["val_pr_auc"]})
        mlflow.set_tag("mejor_run_hijo", mejor["run_id"])
        ruta = P2_OUTPUT_DIR / f"{run_name}_resultados.csv"
        tabla.to_csv(ruta, index=False)
        registrar_artefactos([ruta])
        registrar_config(P2_HGB_PARAMS, {"grid": P2_HGB_SEARCH_GRID})
        registrar_datos(x_train, "p2_train_estacion_hora", VIAJES_PARQUET)

    params_cli = ",".join(f"{k}={mejor[k]}" for k in claves)
    logger.success(
        f"Mejor combinación: {params_cli} (val_f1={mejor['val_f1']:.4f}). Para registrarla "
        f"como candidata: mlflow run . -e train -P question=p2 -P model=hgb -P params={params_cli} "
        f"-P run_name=p2_candidato_hgb_tuneado --experiment-name {P2_MLFLOW_EXPERIMENT} "
        f"--env-manager local"
    )
    return {"mejor": mejor.to_dict(), "resultados": tabla}


def ejecutar(
    model: str = "all",
    params: dict | None = None,
    run_name: str | None = None,
    viajes_path: Path = VIAJES_PARQUET,
    experiment_name: str = P2_MLFLOW_EXPERIMENT,
) -> dict:
    """Entrena y registra un modelo de P2, la búsqueda (model="search") o todos (model="all")."""
    params = params or {}
    if model == "all" and (params or run_name):
        raise ValueError("params y run_name solo se pueden usar al entrenar un modelo concreto.")
    if model == "search" and params:
        raise ValueError("La búsqueda usa P2_HGB_SEARCH_GRID de config.py; no recibe params.")
    nombres = MODELOS if model == "all" else (model,)
    for nombre in nombres:
        if nombre != "search":
            construir_modelo(nombre, params)  # valida nombre e hiperparámetros antes de cargar datos

    logger.info(f"Iniciando pipeline P2 | modelos={', '.join(nombres)} | params={params}")
    datos, contexto = preparar_datos(viajes_path)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    resultados = {}
    for nombre in nombres:
        if nombre == "search":
            resultados[nombre] = ejecutar_busqueda(
                run_name or P2_MLFLOW_SEARCH_RUN_NAME, datos, contexto
            )
            continue
        modelo, etapa, familia, run_default, params_log = construir_modelo(nombre, params)
        resultados[nombre] = entrenar_y_registrar(
            run_name or run_default, etapa, familia, modelo, params_log, datos, contexto
        )
    logger.success("Pipeline P2 terminado")
    return resultados


@app.command()
def main(
    model: str = typer.Option("all", help="baseline, hgb, rf, search o all."),
    params_json: str = typer.Option("{}", help="Hiperparámetros en JSON (se combinan con config.py)."),
    run_name: str = typer.Option("", help="Nombre del run; vacío usa el de config.py."),
    viajes_path: Path = VIAJES_PARQUET,
    experiment_name: str = P2_MLFLOW_EXPERIMENT,
):
    """Pipeline de P2: datos, partición, entrenamiento y registro en MLflow."""
    ejecutar(model, json.loads(params_json), run_name or None, viajes_path, experiment_name)

if __name__ == "__main__":
    app()
