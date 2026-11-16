"""A/B testing de P1: versión vigente (champion) contra versión reentrenada (challenger).

Las dos versiones se cargan del Model Registry por su alias y se evalúan sobre el mismo
conjunto: el holdout de agosto a octubre de 2026, que ninguna vio al entrenar. Como cada
estación-día recibe la predicción de ambas, la comparación es pareada y la diferencia no
depende de qué estaciones o días le tocaron a cada versión.

Se reporta:

    métricas de champion, challenger y del baseline ingenuo (mismo día de la semana anterior)
    por periodo (agosto a octubre completo y cada mes) y por segmento de estaciones
    (las que la versión vigente conoció al entrenar y las nuevas)
    la diferencia de MAE con un intervalo de confianza bootstrap que remuestrea días
    completos, porque los errores de un mismo día están correlacionados
    una prueba de Wilcoxon sobre la diferencia diaria de MAE, como segunda opinión

Regla de decisión, fijada antes de ver los resultados: el challenger se promueve solo si el
intervalo de confianza de la diferencia de MAE (challenger menos champion) queda
completamente debajo de cero. Si no hay diferencia significativa se mantiene la versión
vigente, porque cambiar de modelo tiene costo y no habría evidencia de mejora.

Salidas en reports/ab_test/ y un run p1_ab_test en MLflow:

    python -m mibici.modeling.p1_ab_test
"""

import json
from pathlib import Path
import tempfile

from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import typer

from mibici.config import (
    AB_BOOTSTRAP_MUESTRAS,
    AB_DIR,
    AB_EVAL_END,
    AB_EVAL_START,
    AB_NIVEL_CONFIANZA,
    ALIAS_CHALLENGER,
    ALIAS_PRODUCCION,
    ESTACIONES_ACTUAL_PARQUET,
    ESTACIONES_PARQUET,
    MLFLOW_TRACKING_URI,
    P1_BASELINE_LAG_DIAS,
    P1_MLFLOW_AB_RUN_NAME,
    P1_MLFLOW_EXPERIMENT,
    P1_TRAIN_START,
    P1_VAL_START,
    REGISTRO_MODELOS,
    SEED,
    TEAM_ID,
)
from mibici.features import P1_FEATURES
from mibici.modeling.mlflow_utils import consola_utf8, obtener_git_commit, registrar_artefactos
from mibici.modeling.p1_reentrenamiento import version_de_datos
from mibici.modeling.p1_regresion import evaluar
from mibici.panel_p1 import cargar_capacidad, cargar_viajes_p1, construir_features, cortar

app = typer.Typer()

METRICAS = ["mae", "rmse", "wape", "r2", "sesgo", "mape_total_dia"]
COLORES = {"champion": "#2563eb", "challenger": "#dc2626", "baseline": "#9ca3af"}


# Datos


def preparar_holdout() -> tuple[pd.DataFrame, set]:
    """Holdout del A/B y el conjunto de estaciones que la versión vigente vio al entrenar."""
    viajes = cargar_viajes_p1()
    df = construir_features(
        viajes,
        cargar_capacidad(ESTACIONES_PARQUET),
        cargar_capacidad(ESTACIONES_ACTUAL_PARQUET),
        fin=AB_EVAL_END,
    )
    holdout = cortar(df, AB_EVAL_START, AB_EVAL_END).dropna(subset=P1_FEATURES)
    entrenamiento_v1 = cortar(df, P1_TRAIN_START, P1_VAL_START).dropna(subset=P1_FEATURES)
    return holdout, set(entrenamiento_v1.index.get_level_values("estacion"))


# Modelos


def cargar_version(client: MlflowClient, nombre: str, alias: str):
    """Versión del registry con ese alias y su modelo listo para predecir."""
    version = client.get_model_version_by_alias(nombre, alias)
    modelo = mlflow.pyfunc.load_model(f"models:/{nombre}@{alias}")
    return version, modelo


def predecir(modelo, x: pd.DataFrame) -> np.ndarray:
    """Predicción como arreglo plano, sin valores negativos (igual que en la evaluación)."""
    return np.clip(np.asarray(modelo.predict(x), dtype="float64").ravel(), 0, None)


# Comparación


def metricas_por_segmento(
    holdout: pd.DataFrame, predicciones: dict[str, np.ndarray], vistas: set
) -> pd.DataFrame:
    """Métricas de cada versión por periodo y por segmento de estaciones."""
    y = holdout["viajes"]
    mes = np.asarray(holdout.index.get_level_values("fecha").strftime("%Y-%m"))
    conocida = holdout.index.get_level_values("estacion").isin(list(vistas))
    todas = np.ones(len(holdout), dtype=bool)

    periodos = {"ago-oct": todas, **{m: mes == m for m in sorted(set(mes))}}
    segmentos = {"todas": todas, "conocidas": conocida, "nuevas": ~conocida}

    filas = []
    for version, pred in predicciones.items():
        for periodo, en_periodo in periodos.items():
            for segmento, en_segmento in segmentos.items():
                mascara = en_periodo & en_segmento
                if not mascara.any():
                    continue
                valores = evaluar(y[mascara], pred[mascara], "m")
                filas.append(
                    {
                        "version": version,
                        "periodo": periodo,
                        "segmento": segmento,
                        "filas": int(mascara.sum()),
                        **{clave[2:]: valor for clave, valor in valores.items()},
                    }
                )
    return pd.DataFrame(filas)


def errores_diarios(holdout: pd.DataFrame, pred_champion, pred_challenger) -> pd.DataFrame:
    """Suma de errores absolutos de cada versión por día, y filas de ese día."""
    y = holdout["viajes"].to_numpy(dtype="float64")
    errores = pd.DataFrame(
        {
            "champion": np.abs(y - pred_champion),
            "challenger": np.abs(y - pred_challenger),
        },
        index=holdout.index,
    )
    diario = errores.groupby(level="fecha").sum()
    diario["filas"] = errores.groupby(level="fecha").size()
    return diario


def comparar(
    diario: pd.DataFrame,
    muestras: int = AB_BOOTSTRAP_MUESTRAS,
    nivel: float = AB_NIVEL_CONFIANZA,
    seed: int = SEED,
) -> dict:
    """Diferencia de MAE (challenger menos champion) con bootstrap por días y Wilcoxon.

    Un valor negativo significa que el challenger se equivoca menos. Se remuestrean días
    completos y no filas sueltas, porque los errores de las estaciones en un mismo día están
    correlacionados (lluvia, feriados, eventos) y remuestrear filas daría un intervalo
    demasiado estrecho.
    """
    suma_1 = diario["champion"].to_numpy()
    suma_2 = diario["challenger"].to_numpy()
    filas = diario["filas"].to_numpy()

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(diario), size=(muestras, len(diario)))
    total_1 = suma_1[indices].sum(axis=1)
    total_2 = suma_2[indices].sum(axis=1)
    deltas = (total_2 - total_1) / filas[indices].sum(axis=1)
    alfa = 1 - nivel

    mae_dia_1, mae_dia_2 = suma_1 / filas, suma_2 / filas
    diferencia_dia = mae_dia_2 - mae_dia_1
    p_wilcoxon = float(wilcoxon(diferencia_dia).pvalue) if np.any(diferencia_dia != 0) else 1.0

    mae_1, mae_2 = suma_1.sum() / filas.sum(), suma_2.sum() / filas.sum()
    return {
        "mae_champion": float(mae_1),
        "mae_challenger": float(mae_2),
        "delta_mae": float(mae_2 - mae_1),
        "delta_mae_pct": float((mae_2 - mae_1) / mae_1),
        "ic_inferior": float(np.quantile(deltas, alfa / 2)),
        "ic_superior": float(np.quantile(deltas, 1 - alfa / 2)),
        "nivel_confianza": nivel,
        "muestras_bootstrap": muestras,
        "prob_challenger_mejor": float((deltas < 0).mean()),
        "dias": int(len(diario)),
        "dias_challenger_mejor": int((diferencia_dia < 0).sum()),
        "p_wilcoxon": p_wilcoxon,
    }


def decidir(comparacion: dict) -> dict:
    """Aplica la regla de decisión fijada antes de correr el A/B."""
    if comparacion["ic_superior"] < 0:
        return {
            "version_vigente": "challenger",
            "motivo": (
                "El challenger reduce el MAE y todo el intervalo de confianza de la diferencia "
                "queda debajo de cero: la mejora es significativa y conviene promoverlo."
            ),
        }
    if comparacion["ic_inferior"] > 0:
        return {
            "version_vigente": "champion",
            "motivo": (
                "El challenger aumenta el MAE y todo el intervalo de confianza queda arriba de "
                "cero: es significativamente peor y se mantiene la versión vigente."
            ),
        }
    return {
        "version_vigente": "champion",
        "motivo": (
            "El intervalo de confianza incluye el cero: no hay evidencia de que el challenger "
            "sea mejor, así que se mantiene la versión vigente."
        ),
    }


# Reportes


def graficar(holdout: pd.DataFrame, predicciones: dict, etiquetas: dict, ruta: Path) -> Path:
    """Total diario real contra predicho y MAE diario de cada versión."""
    df = pd.DataFrame(
        {"real": holdout["viajes"].to_numpy(dtype="float64"), **predicciones}, index=holdout.index
    )
    total = df.groupby(level="fecha").sum()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(total.index, total["real"], color="#1f2937", lw=2, label="Real")
    for version in predicciones:
        estilo = "--" if version == "baseline" else "-"
        ax1.plot(
            total.index,
            total[version],
            color=COLORES[version],
            ls=estilo,
            lw=1.5,
            label=etiquetas[version],
        )
        mae_dia = (df[version] - df["real"]).abs().groupby(level="fecha").mean()
        ax2.plot(
            mae_dia.index,
            mae_dia.to_numpy(),
            color=COLORES[version],
            ls=estilo,
            lw=1.5,
            label=etiquetas[version],
        )
    ax1.set_title("P1: viajes totales por día en el holdout del A/B (ago-oct 2026)")
    ax1.set_ylabel("Viajes iniciados")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.set_title("Error absoluto medio por día (viajes por estación-día)")
    ax2.set_ylabel("MAE")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def tabla_markdown(
    metricas: pd.DataFrame, etiquetas: dict, comparacion: dict, decision: dict
) -> str:
    """Tabla comparativa principal (agosto a octubre, todas las estaciones) para el reporte."""
    principal = metricas[(metricas["periodo"] == "ago-oct") & (metricas["segmento"] == "todas")]
    principal = principal.set_index("version")
    versiones = list(principal.index)
    lineas = [
        "| Métrica | " + " | ".join(etiquetas[v] for v in versiones) + " |",
        "|---|" + "---|" * len(versiones),
    ]
    for metrica in METRICAS:
        valores = " | ".join(f"{principal.loc[v, metrica]:.3f}" for v in versiones)
        lineas.append(f"| {metrica} | {valores} |")
    nivel = int(comparacion["nivel_confianza"] * 100)
    lineas += [
        "",
        f"Diferencia de MAE (challenger menos champion): {comparacion['delta_mae']:.3f} "
        f"({comparacion['delta_mae_pct']:+.1%}), IC {nivel}% "
        f"[{comparacion['ic_inferior']:.3f}, {comparacion['ic_superior']:.3f}], "
        f"p de Wilcoxon {comparacion['p_wilcoxon']:.4f}, el challenger gana "
        f"{comparacion['dias_challenger_mejor']} de {comparacion['dias']} días.",
        "",
        f"Decisión: queda como vigente el {decision['version_vigente']}. {decision['motivo']}",
    ]
    return "\n".join(lineas) + "\n"


# Ejecución


def ejecutar(tracking_uri: str = MLFLOW_TRACKING_URI) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    nombre = REGISTRO_MODELOS["p1"]["nombre"]
    v_champion, m_champion = cargar_version(client, nombre, ALIAS_PRODUCCION)
    v_challenger, m_challenger = cargar_version(client, nombre, ALIAS_CHALLENGER)
    etiquetas = {
        "champion": f"champion (versión {v_champion.version})",
        "challenger": f"challenger (versión {v_challenger.version})",
        "baseline": f"baseline (lag {P1_BASELINE_LAG_DIAS} días)",
    }
    logger.info(f"A/B: {etiquetas['champion']} contra {etiquetas['challenger']}")

    holdout, vistas = preparar_holdout()
    x = holdout[P1_FEATURES].astype("float64")
    predicciones = {
        "champion": predecir(m_champion, x),
        "challenger": predecir(m_challenger, x),
        "baseline": x[f"lag_{P1_BASELINE_LAG_DIAS}"].to_numpy(dtype="float64"),
    }
    estaciones = holdout.index.get_level_values("estacion")
    es_nueva = ~estaciones.isin(list(vistas))
    logger.info(
        f"Holdout {AB_EVAL_START} a {AB_EVAL_END}: {len(holdout):,} filas, "
        f"{estaciones.nunique()} estaciones ({es_nueva.sum():,} filas de "
        "estaciones nuevas)"
    )

    metricas = metricas_por_segmento(holdout, predicciones, vistas)
    diario = errores_diarios(holdout, predicciones["champion"], predicciones["challenger"])
    comparacion = comparar(diario)
    decision = decidir(comparacion)

    AB_DIR.mkdir(parents=True, exist_ok=True)
    rutas = {
        "metricas": AB_DIR / "p1_ab_metricas.csv",
        "diario": AB_DIR / "p1_ab_errores_diarios.csv",
        "grafica": AB_DIR / "p1_ab_diario.png",
        "tabla": AB_DIR / "p1_ab_tabla.md",
        "resumen": AB_DIR / "p1_ab_resumen.json",
    }
    metricas.to_csv(rutas["metricas"], index=False)
    diario.to_csv(rutas["diario"])
    graficar(holdout, predicciones, etiquetas, rutas["grafica"])
    rutas["tabla"].write_text(
        tabla_markdown(metricas, etiquetas, comparacion, decision), encoding="utf-8"
    )

    versiones = {
        rol: {
            "version": v.version,
            "run_id": v.run_id,
            "run_name": (v.tags or {}).get("source_run_name"),
        }
        for rol, v in (("champion", v_champion), ("challenger", v_challenger))
    }
    resumen = {
        "modelo_registrado": nombre,
        "versiones": versiones,
        "holdout": {
            "inicio": AB_EVAL_START,
            "fin_exclusivo": AB_EVAL_END,
            "filas": len(holdout),
            "estaciones": int(estaciones.nunique()),
            "filas_estaciones_nuevas": int(es_nueva.sum()),
        },
        "comparacion": comparacion,
        "decision": decision,
        "regla": "promover el challenger solo si el IC de la diferencia de MAE queda debajo de 0",
    }
    rutas["resumen"].write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    git_commit, git_sucio = obtener_git_commit()
    mlflow.set_experiment(P1_MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=P1_MLFLOW_AB_RUN_NAME) as run:
        mlflow.set_tags(
            {
                "equipo": TEAM_ID,
                "problema": "P1",
                "question": "p1",
                "tipo": "ab_test",
                "git_commit": git_commit,
                "git_sucio": str(git_sucio),
                "data_version": version_de_datos(),
                "decision": decision["version_vigente"],
            }
        )
        mlflow.log_params(
            {
                "modelo_registrado": nombre,
                "champion_version": v_champion.version,
                "challenger_version": v_challenger.version,
                "holdout_inicio": AB_EVAL_START,
                "holdout_fin": AB_EVAL_END,
                "filas_holdout": len(holdout),
                "bootstrap_muestras": AB_BOOTSTRAP_MUESTRAS,
                "nivel_confianza": AB_NIVEL_CONFIANZA,
                "random_state": SEED,
            }
        )
        principal = metricas[metricas["segmento"] == "todas"]
        mlflow.log_metrics(
            {
                f"{fila.version}_{fila.periodo}_{metrica}": float(getattr(fila, metrica))
                for fila in principal.itertuples()
                for metrica in METRICAS
            }
        )
        mlflow.log_metrics(
            {k: float(v) for k, v in comparacion.items() if isinstance(v, (int, float))}
        )
        registrar_artefactos(rutas.values())
        with tempfile.TemporaryDirectory() as carpeta:
            ruta_pred = Path(carpeta) / "p1_ab_predicciones.csv"
            tabla_predicciones = pd.DataFrame(
                {"viajes": holdout["viajes"].to_numpy(), **predicciones}, index=holdout.index
            )
            tabla_predicciones.reset_index().to_csv(ruta_pred, index=False)
            mlflow.log_artifact(str(ruta_pred), artifact_path="evaluacion")

    # Deja la evidencia del A/B en ambas versiones del registry.
    for v in (v_champion, v_challenger):
        client.set_model_version_tag(nombre, v.version, "ab_test_run_id", run.info.run_id)
        client.set_model_version_tag(
            nombre, v.version, "ab_test_decision", decision["version_vigente"]
        )

    logger.success(
        f"MAE champion {comparacion['mae_champion']:.3f} | challenger "
        f"{comparacion['mae_challenger']:.3f} | diferencia {comparacion['delta_mae']:+.3f} "
        f"IC [{comparacion['ic_inferior']:.3f}, {comparacion['ic_superior']:.3f}]. "
        f"Vigente: {decision['version_vigente']}. Reportes en {AB_DIR}"
    )
    return resumen


@app.command()
def main(
    tracking_uri: str = typer.Option(MLFLOW_TRACKING_URI, help="URI del servidor de tracking."),
):
    consola_utf8()
    ejecutar(tracking_uri)


if __name__ == "__main__":
    app()
