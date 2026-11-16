"""Monitoreo de drift con Evidently (mibici.monitoreo_drift).

Requiere el entorno de requirements-monitoreo.txt. En el entorno principal estas pruebas
se omiten solas, porque Evidently no es compatible con pandas 3.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("evidently", reason="Requiere el entorno de requirements-monitoreo.txt")

from mibici.config import (  # noqa: E402
    DRIFT_ACTUAL_END,
    DRIFT_ACTUAL_START,
    DRIFT_REF_END,
    DRIFT_REF_START,
    SEED,
)
from mibici.features import P1_FEATURES  # noqa: E402
from mibici.monitoreo_drift import (  # noqa: E402
    COLUMNAS,
    DRIFT_CATEGORICAS,
    composicion,
    cortar,
    drift_por_mes,
    generar_reporte,
    graficar_por_mes,
    proporcion_drift,
    resumen_calidad,
    resumen_por_variable,
    separar_periodos,
)


def _features_sinteticas(inicio, fin, n_estaciones=10, seed=SEED):
    """Panel estación-día con todas las columnas que usa el monitoreo."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range(inicio, fin, freq="D", inclusive="left")
    indice = pd.MultiIndex.from_product(
        [range(1, n_estaciones + 1), fechas], names=["estacion", "fecha"]
    )
    fecha = indice.get_level_values("fecha")
    df = pd.DataFrame(index=indice)
    df["dia_semana"] = fecha.dayofweek
    df["fin_semana"] = (fecha.dayofweek >= 5).astype(int)
    df["feriado"] = rng.binomial(1, 0.05, len(indice))
    for columna in COLUMNAS:
        if columna not in DRIFT_CATEGORICAS:
            df[columna] = rng.poisson(10, len(indice)).astype(float)
    return df[COLUMNAS]


def test_separar_periodos_respeta_fechas_y_quita_filas_sin_historia():
    df = _features_sinteticas("2025-01-01", "2026-11-01")
    df.iloc[:5, df.columns.get_loc("lag_14")] = np.nan  # estación 1, del 1 al 5 de enero 2025

    ref, actual = separar_periodos(df)

    f_ref = ref.index.get_level_values("fecha")
    f_act = actual.index.get_level_values("fecha")
    assert f_ref.min() >= pd.Timestamp(DRIFT_REF_START)
    assert f_ref.max() < pd.Timestamp(DRIFT_REF_END)
    assert f_act.min() >= pd.Timestamp(DRIFT_ACTUAL_START)
    assert f_act.max() < pd.Timestamp(DRIFT_ACTUAL_END)
    assert len(ref) == len(cortar(df, DRIFT_REF_START, DRIFT_REF_END)) - 5
    assert not ref[P1_FEATURES].isna().any().any()
    assert list(ref.columns) == COLUMNAS


def test_resumen_por_variable_detecta_el_drift_inyectado():
    ref = _features_sinteticas(DRIFT_REF_START, DRIFT_REF_END)
    actual = _features_sinteticas(DRIFT_ACTUAL_START, DRIFT_ACTUAL_END, seed=SEED + 1)
    actual["lag_1"] += 30
    actual["viajes"] += 30

    resultados = generar_reporte(ref, actual).dict()
    tabla = resumen_por_variable(resultados)

    assert set(tabla["variable"]) == set(COLUMNAS)
    assert {"metodo", "umbral", "score", "drift"} <= set(tabla.columns)
    con_drift = set(tabla.loc[tabla["drift"], "variable"])
    assert {"lag_1", "viajes"} <= con_drift
    assert "capacidad" not in con_drift

    proporcion = proporcion_drift(resultados)
    assert proporcion["columnas_con_drift"] >= 2
    assert 0 < proporcion["proporcion"] <= 1


def test_resumen_calidad_detecta_faltantes():
    ref = _features_sinteticas(DRIFT_REF_START, DRIFT_REF_END)
    actual = _features_sinteticas(DRIFT_ACTUAL_START, DRIFT_ACTUAL_END, seed=SEED + 1)
    actual.iloc[:20, actual.columns.get_loc("media_7d")] = np.nan

    calidad = resumen_calidad(generar_reporte(ref, actual).dict())

    assert calidad["metricas"]["RowCount"] == len(actual)
    assert calidad["metricas"]["ColumnCount"] == len(COLUMNAS)  # el total, no el de texto
    assert calidad["metricas"]["DatasetMissingValueCount"]["count"] == 20
    assert any("missing" in nombre.lower() for nombre in calidad["pruebas_fallidas"])


def test_drift_por_mes_y_grafica(tmp_path):
    df = _features_sinteticas("2025-01-01", "2026-03-01")
    ref = cortar(df, DRIFT_REF_START, DRIFT_REF_END)

    por_mes = drift_por_mes(df, ref, inicio="2026-01-01", fin="2026-03-01")

    assert sorted(por_mes["mes"].unique()) == ["2026-01", "2026-02"]
    assert len(por_mes) == 2 * len(COLUMNAS)
    assert por_mes["media_viajes_mes"].notna().all()
    ruta = graficar_por_mes(por_mes, tmp_path / "por_mes.png")
    assert ruta.exists() and ruta.stat().st_size > 0


def test_composicion_separa_estaciones_nuevas():
    """Si solo cambian las estaciones nuevas, las comunes no deben mostrar drift."""
    ref = _features_sinteticas(DRIFT_REF_START, DRIFT_REF_END, n_estaciones=10)
    actual = _features_sinteticas(
        DRIFT_ACTUAL_START, DRIFT_ACTUAL_END, n_estaciones=14, seed=SEED + 1
    )
    nuevas = actual.index.get_level_values("estacion") > 10
    actual.loc[nuevas, "viajes"] = 0.0  # estaciones nuevas casi sin demanda

    resumen, tabla = composicion(ref, actual)

    assert resumen["estaciones_comunes"] == 10
    assert resumen["media_viajes_estaciones_nuevas"] == 0
    assert "viajes" not in resumen["variables_con_drift_solo_comunes"]
    assert set(tabla["variable"]) == set(COLUMNAS)
