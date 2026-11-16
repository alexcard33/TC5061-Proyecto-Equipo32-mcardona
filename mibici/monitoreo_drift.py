"""Monitoreo de data drift del modelo P1 con Evidently.

Compara la referencia (enero a septiembre de 2025, la ventana con la que se entrenó la
versión vigente de P1) contra el periodo actual (los mismos meses de 2026). Se comparan
las features que recibe el modelo, construidas con mibici.features, y la variable
objetivo: viajes por estación y día.

Genera en reports/drift/:

    p1_drift_reporte.html       reporte de Evidently: drift por variable y calidad de datos
    p1_drift_reporte.json       el mismo reporte en JSON
    p1_drift_por_variable.csv   método, umbral, score y si hubo drift en cada variable
    p1_drift_estaciones_comunes.csv   lo mismo, solo con estaciones que ya existían
    p1_drift_por_mes.csv        lo mismo para cada mes de 2026 contra la referencia
    p1_drift_por_mes.png        mapa de calor del drift mensual
    resumen.json                cifras principales para el reporte

Evidently 0.7.23 todavía no funciona con pandas 3, así que este módulo corre en su propio
entorno (requirements-monitoreo.txt), como la validación con DeepChecks:

    python -m mibici.monitoreo_drift
    python -m mibici.monitoreo_drift --sin-por-mes
"""

from __future__ import annotations

import json
from pathlib import Path

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import typer

from mibici.config import (
    AB_EVAL_END,
    DRIFT_ACTUAL_END,
    DRIFT_ACTUAL_START,
    DRIFT_DIR,
    DRIFT_REF_END,
    DRIFT_REF_START,
    ESTACIONES_ACTUAL_PARQUET,
    ESTACIONES_PARQUET,
    MONITOREO_QUESTION,
)
from mibici.features import P1_FEATURES
from mibici.panel_p1 import cargar_capacidad, cargar_viajes_p1, construir_features, cortar

app = typer.Typer()

OBJETIVO = "viajes"
COLUMNAS = P1_FEATURES + [OBJETIVO]
DRIFT_CATEGORICAS = ["dia_semana", "fin_semana", "feriado"]

# El mapa de calor corta la escala en este valor; si no, dias_activa (score cercano a 5)
# hace que todas las demás celdas se vean blancas aunque tengan drift.
VMAX_MAPA = 0.3

# Métricas de Evidently que se copian al resumen como indicadores de calidad de datos.
METRICAS_CALIDAD = [
    "RowCount",
    "ColumnCount",
    "DuplicatedRowCount",
    "DatasetMissingValueCount",
    "EmptyRowsCount",
    "EmptyColumnsCount",
    "ConstantColumnsCount",
    "AlmostConstantColumnsCount",
]


# Datos (el panel continuo vive en mibici.panel_p1)


def separar_periodos(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conjuntos de referencia y actual.

    Igual que en el entrenamiento, se quitan las filas sin historia suficiente (los
    primeros 28 días de cada estación), porque el modelo no puede predecir sobre ellas.
    """
    ref = cortar(df, DRIFT_REF_START, DRIFT_REF_END).dropna(subset=P1_FEATURES)
    actual = cortar(df, DRIFT_ACTUAL_START, DRIFT_ACTUAL_END).dropna(subset=P1_FEATURES)
    return ref[COLUMNAS], actual[COLUMNAS]


# Evidently


def definicion_datos() -> DataDefinition:
    numericas = [c for c in COLUMNAS if c not in DRIFT_CATEGORICAS]
    return DataDefinition(numerical_columns=numericas, categorical_columns=DRIFT_CATEGORICAS)


def a_dataset(df: pd.DataFrame) -> Dataset:
    datos = df.reset_index(drop=True).astype("float64")
    datos[DRIFT_CATEGORICAS] = datos[DRIFT_CATEGORICAS].astype("int64")
    return Dataset.from_pandas(datos, data_definition=definicion_datos())


def generar_reporte(ref: pd.DataFrame, actual: pd.DataFrame, calidad: bool = True):
    """Corre el reporte de Evidently y devuelve el Snapshot.

    Ojo con el orden: run recibe primero los datos actuales y después la referencia.
    """
    presets = [DataDriftPreset(), DataSummaryPreset()] if calidad else [DataDriftPreset()]
    return Report(presets, include_tests=True).run(a_dataset(actual), a_dataset(ref))


def _tipo(config: dict) -> str:
    return str(config.get("type", "")).split(":")[-1]


def _estado(prueba: dict) -> str:
    estado = prueba["status"]
    return getattr(estado, "value", estado)


def resumen_por_variable(resultados: dict) -> pd.DataFrame:
    """Una fila por variable con el método, el umbral, el score y si hay drift.

    Recibe snapshot.dict(). Evidently elige el método según el tipo y el tamaño de cada
    columna; con muestras grandes usa distancias (Wasserstein para numéricas y
    Jensen-Shannon para categóricas) y hay drift cuando el score supera el umbral. La
    columna drift sale directamente de la prueba de Evidently, así que también es correcta
    si el método es un p-valor.
    """
    scores = {m["id"]: m["value"] for m in resultados["metrics"]}
    filas = []
    for prueba in resultados.get("tests", []):
        params = prueba["metric_config"]["params"]
        if _tipo(params) != "ValueDrift":
            continue
        score = scores.get(prueba["metric_config"]["metric_id"], float("nan"))
        filas.append(
            {
                "variable": params["column"],
                "metodo": params.get("method"),
                "umbral": params.get("threshold"),
                "score": float(score),
                "drift": _estado(prueba) == "FAIL",
            }
        )
    tabla = pd.DataFrame(filas)
    return tabla.sort_values(["drift", "score"], ascending=False, ignore_index=True)


def proporcion_drift(resultados: dict) -> dict:
    """Cuántas columnas tienen drift y qué proporción representan."""
    for metrica in resultados["metrics"]:
        if _tipo(metrica["config"]) == "DriftedColumnsCount":
            return {
                "columnas_con_drift": int(metrica["value"]["count"]),
                "proporcion": float(metrica["value"]["share"]),
            }
    return {}


def resumen_calidad(resultados: dict) -> dict:
    """Indicadores de calidad del periodo actual y las pruebas de calidad que no pasaron."""
    metricas = {}
    for m in resultados["metrics"]:
        tipo = _tipo(m["config"])
        # Evidently repite algunas métricas por tipo de columna (ColumnCount trae el total,
        # las numéricas, las categóricas, etc.). La primera es la del conjunto completo.
        if tipo in METRICAS_CALIDAD:
            metricas.setdefault(tipo, m["value"])
    fallidas = [
        p["name"]
        for p in resultados.get("tests", [])
        if _estado(p) == "FAIL"
        and _tipo(p["metric_config"]["params"]) not in ("ValueDrift", "DriftedColumnsCount")
    ]
    return {"metricas": metricas, "pruebas_fallidas": fallidas}


def composicion(ref: pd.DataFrame, actual: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Separa el cambio de demanda del cambio en la red de estaciones.

    Repite la comparación solo con las estaciones que existen en ambos periodos. Si el
    drift de viajes desaparece aquí, lo que cambió fue la red (estaciones nuevas con menos
    demanda) y no el comportamiento de las estaciones que el modelo ya conocía.
    """
    comunes = set(ref.index.get_level_values("estacion")) & set(
        actual.index.get_level_values("estacion")
    )
    ref_c = ref[ref.index.get_level_values("estacion").isin(comunes)]
    es_comun = actual.index.get_level_values("estacion").isin(comunes)
    act_c, act_nuevas = actual[es_comun], actual[~es_comun]

    tabla = resumen_por_variable(generar_reporte(ref_c, act_c, calidad=False).dict())
    resumen = {
        "estaciones_comunes": len(comunes),
        "media_viajes_comunes_referencia": round(float(ref_c[OBJETIVO].mean()), 3),
        "media_viajes_comunes_actual": round(float(act_c[OBJETIVO].mean()), 3),
        "media_viajes_estaciones_nuevas": (
            round(float(act_nuevas[OBJETIVO].mean()), 3) if len(act_nuevas) else None
        ),
        "variables_con_drift_solo_comunes": tabla.loc[tabla["drift"], "variable"].tolist(),
        "score_viajes_solo_comunes": float(tabla.set_index("variable").loc[OBJETIVO, "score"]),
    }
    return resumen, tabla


def drift_por_mes(
    df: pd.DataFrame, ref: pd.DataFrame, inicio: str = DRIFT_ACTUAL_START, fin: str = AB_EVAL_END
) -> pd.DataFrame:
    """Drift de cada mes contra la referencia completa.

    Sirve para ver desde cuándo aparece el drift y si sigue en los meses del A/B testing.
    Aquí la referencia es fija, como en un monitoreo en producción, así que sí se cuela algo
    de estacionalidad; el reporte principal la evita comparando los mismos meses.
    """
    tablas = []
    for mes in pd.date_range(inicio, fin, freq="MS", inclusive="left"):
        actual = cortar(df, mes, mes + pd.offsets.MonthBegin(1)).dropna(subset=P1_FEATURES)
        if actual.empty:
            logger.warning(f"Sin datos para {mes:%Y-%m}, se omite")
            continue
        resultados = generar_reporte(ref, actual[COLUMNAS], calidad=False).dict()
        tabla = resumen_por_variable(resultados)
        tabla.insert(0, "mes", f"{mes:%Y-%m}")
        tabla["filas"] = len(actual)
        # El score no tiene signo; la media dice si la demanda subió o bajó ese mes.
        tabla["media_viajes_mes"] = round(float(actual[OBJETIVO].mean()), 3)
        tablas.append(tabla)
    return pd.concat(tablas, ignore_index=True)


def graficar_por_mes(por_mes: pd.DataFrame, ruta: Path) -> Path:
    """Mapa de calor del score de drift por mes; el asterisco marca drift detectado."""
    scores = por_mes.pivot(index="variable", columns="mes", values="score").reindex(COLUMNAS)
    drift = por_mes.pivot(index="variable", columns="mes", values="drift").reindex(COLUMNAS)

    fig, ax = plt.subplots(figsize=(scores.shape[1] + 3, 0.45 * scores.shape[0] + 1.5))
    imagen = ax.imshow(
        scores.to_numpy(dtype="float64"), cmap="Reds", vmin=0, vmax=VMAX_MAPA, aspect="auto"
    )
    ax.set_xticks(range(scores.shape[1]), scores.columns, rotation=45, ha="right")
    ax.set_yticks(range(scores.shape[0]), scores.index)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            valor = scores.iat[i, j]
            if pd.notna(valor):
                marca = "*" if bool(drift.iat[i, j]) else ""
                color = "white" if valor > 0.6 * VMAX_MAPA else "black"
                ax.text(
                    j, i, f"{valor:.2f}{marca}", ha="center", va="center", fontsize=7, color=color
                )
    fig.colorbar(imagen, ax=ax, extend="max", label="Score de drift de Evidently")
    ax.set_title(
        "P1: drift de cada mes de 2026 contra la referencia (ene-sep 2025)\n"
        f"* Evidently detecta drift. Escala cortada en {VMAX_MAPA} (el umbral es 0.1)"
    )
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return ruta


def _describir(df: pd.DataFrame, inicio: str, fin: str) -> dict:
    return {
        "inicio": inicio,
        "fin_exclusivo": fin,
        "filas": len(df),
        "estaciones": int(df.index.get_level_values("estacion").nunique()),
        "media_viajes": round(float(df[OBJETIVO].mean()), 3),
    }


@app.command()
def main(
    por_mes: bool = typer.Option(
        True, "--por-mes/--sin-por-mes", help="Calcula también el drift de cada mes de 2026."
    ),
):
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    viajes = cargar_viajes_p1()
    df = construir_features(
        viajes, cargar_capacidad(ESTACIONES_PARQUET), cargar_capacidad(ESTACIONES_ACTUAL_PARQUET)
    )
    ref, actual = separar_periodos(df)
    filas_sin_historia = len(cortar(df, DRIFT_ACTUAL_START, DRIFT_ACTUAL_END)) - len(actual)
    logger.info(f"Referencia: {len(ref):,} filas. Actual: {len(actual):,} filas.")

    snapshot = generar_reporte(ref, actual)
    snapshot.save_html(str(DRIFT_DIR / "p1_drift_reporte.html"))
    snapshot.save_json(str(DRIFT_DIR / "p1_drift_reporte.json"))
    resultados = snapshot.dict()

    por_variable = resumen_por_variable(resultados)
    por_variable.to_csv(DRIFT_DIR / "p1_drift_por_variable.csv", index=False)

    estaciones_ref = set(ref.index.get_level_values("estacion"))
    estaciones_act = set(actual.index.get_level_values("estacion"))
    nuevas = sorted(estaciones_act - estaciones_ref)
    primera_salida = (
        pd.Series(df.index.get_level_values("fecha"), index=df.index.get_level_values("estacion"))
        .groupby(level=0)
        .min()
    )
    aperturas = primera_salida.loc[nuevas].dt.strftime("%Y-%m").value_counts().sort_index()

    resumen_composicion, tabla_comunes = composicion(ref, actual)
    tabla_comunes.to_csv(DRIFT_DIR / "p1_drift_estaciones_comunes.csv", index=False)
    resumen = {
        "modelo": MONITOREO_QUESTION,
        "referencia": _describir(ref, DRIFT_REF_START, DRIFT_REF_END),
        "actual": _describir(actual, DRIFT_ACTUAL_START, DRIFT_ACTUAL_END),
        "filas_actual_sin_historia": filas_sin_historia,
        "estaciones_nuevas": [int(e) for e in nuevas],
        "estaciones_nuevas_por_mes_de_primera_salida": aperturas.to_dict(),
        "estaciones_sin_actividad": sorted(int(e) for e in estaciones_ref - estaciones_act),
        "drift": proporcion_drift(resultados),
        "variables_con_drift": por_variable.loc[por_variable["drift"], "variable"].tolist(),
        "composicion": resumen_composicion,
        "calidad": resumen_calidad(resultados),
    }

    if por_mes:
        mensual = drift_por_mes(df, ref)
        mensual.to_csv(DRIFT_DIR / "p1_drift_por_mes.csv", index=False)
        graficar_por_mes(mensual, DRIFT_DIR / "p1_drift_por_mes.png")
        resumen["proporcion_variables_con_drift_por_mes"] = (
            mensual.groupby("mes")["drift"].mean().round(3).to_dict()
        )
        resumen["media_viajes_por_mes"] = (
            mensual.groupby("mes")["media_viajes_mes"].first().to_dict()
        )

    (DRIFT_DIR / "resumen.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )
    logger.success(
        f"Drift en {resumen['drift'].get('columnas_con_drift')} de {len(COLUMNAS)} variables: "
        f"{', '.join(resumen['variables_con_drift']) or 'ninguna'}. Reportes en {DRIFT_DIR}"
    )


if __name__ == "__main__":
    app()
