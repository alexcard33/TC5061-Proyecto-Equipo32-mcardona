"""P1: regresión de viajes iniciados por estación y día (Johanna Camargo).

Pregunta: ¿cuántos viajes se iniciarán en cada estación el día siguiente?

Pipeline reproducible de extremo a extremo:

1. Lee ``data/processed/viajes.parquet`` (salida de ``python -m mibici.dataset``)
   y descarta viajes de menos de 60 s o más de 24 h.
2. Construye el panel diario estación × fecha y las variables de
   ``mibici.features`` (rezagos >= 1 día, medias móviles, calendario).
3. Partición temporal: TRAIN ene-sep 2025, VAL oct 2025, TEST nov-dic 2025.
4. Entrena y registra en MLflow (experimento común ``mibici-equipo32``):
   baseline ingenuo semanal, candidato lineal Poisson y candidato
   HistGradientBoosting con pérdida Poisson.

Ejecución::

    python -m mibici.modeling.p1_regresion                  # todos los modelos
    python -m mibici.modeling.p1_regresion --model poisson  # un modelo
    python -m mibici.modeling.p1_regresion --model search   # búsqueda HGB (nested runs)
    mlflow run . -e train -P question=p1 -P model=hgb       # desde el MLflow Project
"""

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
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import typer

from mibici.config import (
    ESTACIONES_PARQUET,
    MANIFEST_PATH,
    MLFLOW_TRACKING_URI,
    P1_BASELINE_LAG_DIAS,
    P1_FERIADOS,
    P1_HGB_PARAMS,
    P1_HGB_SEARCH_GRID,
    P1_HORIZONTE_DIAS,
    P1_LAGS_DIAS,
    P1_MAX_DURATION_SECONDS,
    P1_MIN_DURATION_SECONDS,
    P1_MLFLOW_BASELINE_RUN_NAME,
    P1_MLFLOW_CANDIDATE_RUN_NAME,
    P1_MLFLOW_EXPERIMENT,
    P1_MLFLOW_LINEAL_RUN_NAME,
    P1_MLFLOW_RF_RUN_NAME,
    P1_MLFLOW_SEARCH_RUN_NAME,
    P1_OUTPUT_DIR,
    P1_POISSON_PARAMS,
    P1_QUESTION,
    P1_RF_PARAMS,
    P1_TEST_END,
    P1_TEST_START,
    P1_TRAIN_START,
    P1_VAL_START,
    P1_VENTANAS_MEDIA_DIAS,
    PROJ_ROOT,
    SEED,
    TEAM_ID,
    VIAJES_PARQUET,
)
from mibici.features import (
    P1_FEATURES,
    p1_agregar_features,
    p1_construir_panel_diario,
)
from mibici.modeling.mlflow_utils import (
    registrar_artefactos,
    registrar_config,
    registrar_datos,
    tags_estandar,
)

app = typer.Typer()

SKOPS_TRUSTED = [
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.tree._tree.Tree",
]
DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]


class BaselineNaiveSemanal:
    """Predice para el día D los viajes del mismo día de la semana anterior (``lag_7``)."""

    def __init__(self, columna: str):
        self.columna = columna

    def fit(self, x, y=None):
        return self

    def predict(self, x):
        return x[self.columna].to_numpy(dtype="float64")


def obtener_git_commit() -> tuple[str, bool]:
    """Devuelve el commit actual y si hay cambios sin confirmar."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True).strip()
        estado = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJ_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "desconocido", True
    return commit, bool(estado)


def cargar_viajes(ruta: Path) -> pd.DataFrame:
    """Lee los viajes procesados y aplica el filtro de duración de P1."""
    viajes = pd.read_parquet(ruta, columns=["inicio_viaje", "fin_viaje", "origen_id"])
    duracion = (viajes["fin_viaje"] - viajes["inicio_viaje"]).dt.total_seconds()
    return viajes.loc[duracion.between(P1_MIN_DURATION_SECONDS, P1_MAX_DURATION_SECONDS)].dropna()


def cargar_capacidad(ruta: Path) -> pd.Series:
    """Capacidad instalada (``dpcapacity``) por id de estación."""
    estaciones = pd.read_parquet(ruta, columns=["id", "dpcapacity"])
    return estaciones.set_index("id")["dpcapacity"]


def cortar(df: pd.DataFrame, inicio: str, fin: str) -> pd.DataFrame:
    """Filas con fecha en ``[inicio, fin)``."""
    fecha = df.index.get_level_values("fecha")
    return df.loc[(fecha >= pd.Timestamp(inicio)) & (fecha < pd.Timestamp(fin))]


def construir_lineal(params: dict) -> Pipeline:
    """Candidato 1: regresión Poisson lineal (enlace log).

    Los conteos (rezagos y medias) entran como ``log1p`` para que el modelo sea
    multiplicativo y estable; el día de la semana entra en one-hot.
    """
    conteos = [c for c in P1_FEATURES if c.startswith(("lag_", "media_"))]
    otras = [c for c in P1_FEATURES if c not in conteos and c != "dia_semana"]
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("dia", OneHotEncoder(handle_unknown="ignore"), ["dia_semana"]),
                        (
                            "conteos",
                            Pipeline([("log", FunctionTransformer(np.log1p)), ("esc", StandardScaler())]),
                            conteos,
                        ),
                        ("num", StandardScaler(), otras),
                    ]
                ),
            ),
            ("modelo", PoissonRegressor(**params)),
        ]
    )


def construir_hgb(params: dict) -> HistGradientBoostingRegressor:
    """Candidato 2: gradient boosting con pérdida Poisson (conteos no negativos)."""
    return HistGradientBoostingRegressor(**{"random_state": SEED, **params})


def construir_rf(params: dict) -> RandomForestRegressor:
    """Candidato 3: bosque aleatorio (no lineal, sin supuesto de distribución)."""
    return RandomForestRegressor(**{"random_state": SEED, **params})


def evaluar(y: pd.Series, pred: np.ndarray, prefijo: str) -> dict:
    """Métricas de regresión a nivel estación-día y de total diario del sistema."""
    pred = np.clip(pred, 0, None)
    total = pd.DataFrame({"y": y.to_numpy(), "p": pred}, index=y.index).groupby(level="fecha").sum()
    return {
        f"{prefijo}_mae": mean_absolute_error(y, pred),
        f"{prefijo}_rmse": float(np.sqrt(mean_squared_error(y, pred))),
        f"{prefijo}_wape": float(np.abs(y - pred).sum() / y.sum()),
        f"{prefijo}_r2": r2_score(y, pred),
        f"{prefijo}_sesgo": float((pred - y).mean()),
        f"{prefijo}_mape_total_dia": float((np.abs(total["p"] - total["y"]) / total["y"]).mean()),
    }


def guardar_graficas(nombre: str, test: pd.DataFrame, pred: np.ndarray, base: np.ndarray) -> list[Path]:
    """Serie del total diario en TEST y MAE por día de la semana (modelo vs. baseline)."""
    df = test[["viajes", "dia_semana"]].copy()
    df["pred"] = np.clip(pred, 0, None)
    df["base"] = base

    total = df.groupby(level="fecha")[["viajes", "pred", "base"]].sum()
    ruta_serie = P1_OUTPUT_DIR / f"{nombre}_total_diario_test.png"
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(total.index, total["viajes"], color="#1f2937", lw=2, label="Real")
    ax.plot(total.index, total["base"], color="#9ca3af", lw=1.5, ls="--", label="Baseline (t-7 días)")
    if nombre != P1_MLFLOW_BASELINE_RUN_NAME:
        ax.plot(total.index, total["pred"], color="#2563eb", lw=1.8, label=nombre)
    ax.set_title("P1 · Viajes totales por día en TEST (nov-dic 2025)")
    ax.set_ylabel("Viajes iniciados")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(ruta_serie, dpi=150, bbox_inches="tight")
    plt.close(fig)

    df["err_pred"] = (df["viajes"] - df["pred"]).abs()
    df["err_base"] = (df["viajes"] - df["base"]).abs()
    por_dia = df.groupby("dia_semana")[["err_base", "err_pred"]].mean()
    ruta_dia = P1_OUTPUT_DIR / f"{nombre}_mae_por_dia_test.png"
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(por_dia))
    ax.bar(x - 0.2, por_dia["err_base"], 0.4, color="#9ca3af", label="Baseline (t-7 días)")
    ax.bar(x + 0.2, por_dia["err_pred"], 0.4, color="#2563eb", label=nombre)
    ax.set_xticks(x, [DIAS[int(i)] for i in por_dia.index])
    ax.set_ylabel("MAE (viajes por estación-día)")
    ax.set_title("P1 · Error absoluto medio por día de la semana (TEST)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(ruta_dia, dpi=150, bbox_inches="tight")
    plt.close(fig)

    muestra = df.sample(min(20_000, len(df)), random_state=SEED)
    limite = float(max(muestra["viajes"].max(), muestra["pred"].max()))
    ruta_disp = P1_OUTPUT_DIR / f"{nombre}_real_vs_predicho_test.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.scatter(muestra["viajes"], muestra["pred"], s=4, alpha=0.25, color="#2563eb")
    ax1.plot([0, limite], [0, limite], color="#1f2937", lw=1, ls="--")
    ax1.set_xlabel("Viajes reales (estación-día)")
    ax1.set_ylabel("Viajes predichos")
    ax1.set_title("P1 · Real vs. predicho (TEST, muestra de 20 mil)")
    ax1.grid(alpha=0.3)
    ax2.hist(df["viajes"] - df["pred"], bins=60, color="#2563eb", alpha=0.85)
    ax2.axvline(0, color="#1f2937", lw=1, ls="--")
    ax2.set_xlabel("Residuo (real - predicho)")
    ax2.set_ylabel("Estación-días")
    ax2.set_title("P1 · Distribución de residuos (TEST)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta_disp, dpi=150, bbox_inches="tight")
    plt.close(fig)

    ruta_pred = P1_OUTPUT_DIR / f"{nombre}_predicciones_test.csv"
    df[["viajes", "pred"]].reset_index().to_csv(ruta_pred, index=False)
    return [ruta_serie, ruta_dia, ruta_disp, ruta_pred]


def guardar_importancias(modelo, x_val: pd.DataFrame, y_val: pd.Series, nombre: str) -> list[Path]:
    """Importancia por permutación en VAL (muestra fija de 40 mil filas)."""
    muestra = x_val.sample(min(40_000, len(x_val)), random_state=SEED)
    res = permutation_importance(
        modelo, muestra, y_val.loc[muestra.index], scoring="neg_mean_absolute_error",
        n_repeats=3, random_state=SEED,
    )
    imp = pd.Series(res.importances_mean, index=x_val.columns).sort_values()
    ruta_csv = P1_OUTPUT_DIR / f"{nombre}_importancia_permutacion.csv"
    imp.rename("aumento_mae").to_csv(ruta_csv)
    ruta_png = P1_OUTPUT_DIR / f"{nombre}_importancia_permutacion.png"
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(imp.index, imp.values, color="#2563eb")
    ax.set_xlabel("Aumento del MAE al permutar la variable (VAL)")
    ax.set_title("P1 · Importancia de variables (HGB Poisson)")
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(ruta_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [ruta_png, ruta_csv]


def tags_run(familia: str, etapa: str, contexto: dict) -> dict:
    """Tags de equipo, commit y versión de datos más los estándar (question, family, stage)."""
    return {
        "equipo": TEAM_ID,
        "mlflow.user": TEAM_ID,
        "problema": "P1",
        "tipo": "regresion",
        "pregunta": P1_QUESTION,
        "git_commit": contexto["git_commit"],
        "git_sucio": str(contexto["git_sucio"]),
        "data_version": contexto["data_version"],
        **tags_estandar("p1", familia, etapa),
    }


def entrenar_y_registrar(
    nombre_run, etapa, familia, modelo, params_modelo, datos, contexto, base_test
) -> dict:
    """Ajusta en TRAIN, evalúa en TRAIN/VAL/TEST y registra todo en MLflow."""
    (x_train, y_train), (x_val, y_val), (x_test, y_test), test = datos

    modelo.fit(x_train, y_train)
    pred = {n: modelo.predict(x) for n, x in (("train", x_train), ("val", x_val), ("test", x_test))}
    metricas = {
        **evaluar(y_train, pred["train"], "train"),
        **evaluar(y_val, pred["val"], "val"),
        **evaluar(y_test, pred["test"], "test"),
    }
    artefactos = guardar_graficas(nombre_run, test, pred["test"], base_test)
    if isinstance(modelo, HistGradientBoostingRegressor):
        artefactos += guardar_importancias(modelo, x_val, y_val, nombre_run)
    ruta_metricas = P1_OUTPUT_DIR / f"{nombre_run}_metricas.json"
    ruta_metricas.write_text(json.dumps(metricas, indent=2))
    artefactos.append(ruta_metricas)

    with mlflow.start_run(run_name=nombre_run):
        mlflow.set_tags(tags_run(familia, etapa, contexto))
        mlflow.log_params(
            {
                **params_modelo,
                "objetivo": "viajes iniciados por estacion y dia",
                "horizonte_dias": P1_HORIZONTE_DIAS,
                "lags_dias": str(P1_LAGS_DIAS),
                "ventanas_media_dias": str(P1_VENTANAS_MEDIA_DIAS),
                "features": len(P1_FEATURES),
                "train_inicio": P1_TRAIN_START,
                "val_inicio": P1_VAL_START,
                "test_inicio": P1_TEST_START,
                "test_fin": P1_TEST_END,
                "random_state": SEED,
                "data_version": contexto["data_version"],
                **contexto["params_datos"],
            }
        )
        mlflow.log_metrics(metricas)
        if not isinstance(modelo, BaselineNaiveSemanal):
            ejemplo = x_test.head(5)
            mlflow.sklearn.log_model(
                modelo,
                name="modelo",
                signature=infer_signature(ejemplo, modelo.predict(ejemplo)),
                input_example=ejemplo,
                skops_trusted_types=SKOPS_TRUSTED,
            )
        registrar_artefactos(artefactos)
        registrar_config(params_modelo)
        registrar_datos(x_train, "p1_train_estacion_dia", VIAJES_PARQUET)

    logger.info(
        f"{nombre_run}: val_mae={metricas['val_mae']:.3f} test_mae={metricas['test_mae']:.3f} "
        f"test_rmse={metricas['test_rmse']:.3f} test_wape={metricas['test_wape']:.3f}"
    )
    return metricas


MODELOS = ("baseline", "poisson", "hgb", "rf")


def preparar_datos(viajes_path: Path = VIAJES_PARQUET):
    """Panel diario, features y partición temporal (TRAIN ene-sep, VAL oct, TEST nov-dic)."""
    P1_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    viajes = cargar_viajes(viajes_path)
    capacidad = cargar_capacidad(ESTACIONES_PARQUET)
    panel = p1_construir_panel_diario(viajes, P1_TRAIN_START, P1_TEST_END)
    df = p1_agregar_features(
        panel, capacidad, P1_FERIADOS, P1_LAGS_DIAS, P1_VENTANAS_MEDIA_DIAS, P1_HORIZONTE_DIAS
    )

    train = cortar(df, P1_TRAIN_START, P1_VAL_START)
    val = cortar(df, P1_VAL_START, P1_TEST_START)
    test = cortar(df, P1_TEST_START, P1_TEST_END)

    filas_antes = {"train": len(train), "val": len(val), "test": len(test)}
    train, val, test = (d.dropna(subset=P1_FEATURES) for d in (train, val, test))
    for nombre, d in (("train", train), ("val", val), ("test", test)):
        logger.info(
            f"{nombre}: filas={len(d):,} (sin historia suficiente={filas_antes[nombre] - len(d):,}) "
            f"estaciones={d.index.get_level_values('estacion').nunique()} "
            f"media={d['viajes'].mean():.2f} viajes/estacion-dia"
        )

    datos = (
        (train[P1_FEATURES].astype("float64"), train["viajes"]),
        (val[P1_FEATURES].astype("float64"), val["viajes"]),
        (test[P1_FEATURES].astype("float64"), test["viajes"]),
        test,
    )
    base_test = test[f"lag_{P1_BASELINE_LAG_DIAS}"].to_numpy(dtype="float64")

    git_commit, git_sucio = obtener_git_commit()
    if git_sucio:
        logger.warning("Hay cambios sin commit: el run no será reproducible desde git_commit")
    contexto = {
        "git_commit": git_commit,
        "git_sucio": git_sucio,
        "data_version": json.loads(MANIFEST_PATH.read_text())["data_version"],
        "params_datos": {
            "estaciones_test": test.index.get_level_values("estacion").nunique(),
            "filas_train": len(train),
            "filas_val": len(val),
            "filas_test": len(test),
        },
    }
    return datos, base_test, contexto


def construir_modelo(nombre: str, params: dict):
    """Devuelve (modelo, etapa, familia, nombre de run por defecto, parámetros a registrar)."""
    if nombre == "baseline":
        if params:
            raise ValueError("El baseline de P1 no recibe hiperparámetros.")
        modelo = BaselineNaiveSemanal(f"lag_{P1_BASELINE_LAG_DIAS}")
        params_log = {"modelo": "NaiveEstacional", "lag_dias": P1_BASELINE_LAG_DIAS}
        return modelo, "baseline", "naive", P1_MLFLOW_BASELINE_RUN_NAME, params_log
    if nombre == "poisson":
        p = {**P1_POISSON_PARAMS, **params}
        params_log = {
            "modelo": "PoissonRegressor",
            **p,
            "transformacion_conteos": "log1p",
            "escalado": "StandardScaler",
        }
        return construir_lineal(p), "candidate", "glm_poisson", P1_MLFLOW_LINEAL_RUN_NAME, params_log
    if nombre == "hgb":
        p = {**P1_HGB_PARAMS, **params}
        params_log = {"modelo": "HistGradientBoostingRegressor", **p}
        return construir_hgb(p), "candidate", "gradient_boosting", P1_MLFLOW_CANDIDATE_RUN_NAME, params_log
    if nombre == "rf":
        p = {**P1_RF_PARAMS, **params}
        params_log = {"modelo": "RandomForestRegressor", **p}
        return construir_rf(p), "candidate", "random_forest", P1_MLFLOW_RF_RUN_NAME, params_log
    raise ValueError(
        f"Modelo de P1 desconocido: {nombre!r}. Opciones: {', '.join(MODELOS)}, search o all."
    )


def ejecutar_busqueda(run_name, datos, contexto) -> dict:
    """Búsqueda en malla de HGB: un run padre y un run hijo (nested) por combinación.

    Se selecciona con VAL (menor val_mae); TEST no se calcula para no influir en la elección.
    """
    (x_train, y_train), (x_val, y_val), _, _ = datos
    claves = list(P1_HGB_SEARCH_GRID)
    combinaciones = [
        dict(zip(claves, valores)) for valores in itertools.product(*P1_HGB_SEARCH_GRID.values())
    ]
    resultados = []
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags_run("gradient_boosting", "search", contexto))
        mlflow.log_params(
            {
                "modelo": "HistGradientBoostingRegressor",
                "criterio": "min val_mae",
                "combinaciones": len(combinaciones),
                "grid": json.dumps(P1_HGB_SEARCH_GRID),
                "data_version": contexto["data_version"],
            }
        )
        for i, combo in enumerate(combinaciones, start=1):
            params = {**P1_HGB_PARAMS, **combo}
            modelo = construir_hgb(params)
            modelo.fit(x_train, y_train)
            metricas = {
                **evaluar(y_train, modelo.predict(x_train), "train"),
                **evaluar(y_val, modelo.predict(x_val), "val"),
            }
            with mlflow.start_run(run_name=f"{run_name}_{i:02d}", nested=True) as hijo:
                mlflow.set_tags(tags_run("gradient_boosting", "search", contexto))
                mlflow.log_params({"modelo": "HistGradientBoostingRegressor", **params})
                mlflow.log_metrics(metricas)
            resultados.append({"run_id": hijo.info.run_id, **combo, **metricas})
            logger.info(f"{run_name}_{i:02d} {combo}: val_mae={metricas['val_mae']:.4f}")

        tabla = pd.DataFrame(resultados).sort_values("val_mae").reset_index(drop=True)
        mejor = tabla.iloc[0]
        mlflow.log_params({f"mejor_{k}": mejor[k] for k in claves})
        mlflow.log_metrics({"mejor_val_mae": mejor["val_mae"], "mejor_train_mae": mejor["train_mae"]})
        mlflow.set_tag("mejor_run_hijo", mejor["run_id"])
        ruta = P1_OUTPUT_DIR / f"{run_name}_resultados.csv"
        tabla.to_csv(ruta, index=False)
        registrar_artefactos([ruta])
        registrar_config(P1_HGB_PARAMS, {"grid": P1_HGB_SEARCH_GRID})
        registrar_datos(x_train, "p1_train_estacion_dia", VIAJES_PARQUET)

    params_cli = ",".join(f"{k}={mejor[k]}" for k in claves)
    logger.success(
        f"Mejor combinación: {params_cli} (val_mae={mejor['val_mae']:.4f}). Para registrarla "
        f"como candidata: mlflow run . -e train -P question=p1 -P model=hgb -P params={params_cli} "
        f"-P run_name=p1_candidato_hgb_tuneado --experiment-name {P1_MLFLOW_EXPERIMENT} "
        f"--env-manager local"
    )
    return {"mejor": mejor.to_dict(), "resultados": tabla}


def ejecutar(
    model: str = "all",
    params: dict | None = None,
    run_name: str | None = None,
    viajes_path: Path = VIAJES_PARQUET,
    experiment_name: str = P1_MLFLOW_EXPERIMENT,
) -> dict:
    """Entrena y registra un modelo de P1, la búsqueda (model="search") o todos (model="all")."""
    params = params or {}
    if model == "all" and (params or run_name):
        raise ValueError("params y run_name solo se pueden usar al entrenar un modelo concreto.")
    if model == "search" and params:
        raise ValueError("La búsqueda usa P1_HGB_SEARCH_GRID de config.py; no recibe params.")
    nombres = MODELOS if model == "all" else (model,)
    for nombre in nombres:
        if nombre != "search":
            construir_modelo(nombre, params)  # valida nombre e hiperparámetros antes de cargar datos

    logger.info(f"Iniciando pipeline P1 | modelos={', '.join(nombres)} | params={params}")
    datos, base_test, contexto = preparar_datos(viajes_path)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    resultados = {}
    for nombre in nombres:
        if nombre == "search":
            resultados[nombre] = ejecutar_busqueda(
                run_name or P1_MLFLOW_SEARCH_RUN_NAME, datos, contexto
            )
            continue
        modelo, etapa, familia, run_default, params_log = construir_modelo(nombre, params)
        resultados[nombre] = entrenar_y_registrar(
            run_name or run_default, etapa, familia, modelo, params_log, datos, contexto, base_test
        )
    logger.success("Pipeline P1 terminado")
    return resultados


@app.command()
def main(
    model: str = typer.Option("all", help="baseline, poisson, hgb, rf, search o all."),
    params_json: str = typer.Option("{}", help="Hiperparámetros en JSON (se combinan con config.py)."),
    run_name: str = typer.Option("", help="Nombre del run; vacío usa el de config.py."),
    viajes_path: Path = VIAJES_PARQUET,
    experiment_name: str = P1_MLFLOW_EXPERIMENT,
):
    ejecutar(model, json.loads(params_json), run_name or None, viajes_path, experiment_name)

if __name__ == "__main__":
    app()
