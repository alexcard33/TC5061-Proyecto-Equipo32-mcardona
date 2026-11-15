"""Pruebas de los periodos del monitoreo de drift y del A/B testing (``mibici.config``).

Verifican que los conjuntos estén bien definidos: sin traslapes que provoquen fuga de
información y con los datos y feriados necesarios para el periodo actual.
"""

import pandas as pd

from mibici.config import (
    AB_EVAL_END,
    AB_EVAL_START,
    DRIFT_ACTUAL_END,
    DRIFT_ACTUAL_START,
    DRIFT_REF_END,
    DRIFT_REF_START,
    P1_FERIADOS,
    P1_TEST_END,
    P1_VAL_START,
    V2_TRAIN_START,
    V2_VAL_START,
    VIAJES_ACTUAL_URLS,
    VIAJES_URLS,
)

ts = pd.Timestamp


def _meses(inicio: str, fin_exclusivo: str) -> set[str]:
    rango = pd.period_range(ts(inicio), ts(fin_exclusivo) - pd.Timedelta(days=1), freq="M")
    return {m.strftime("%Y_%m") for m in rango}


def test_referencia_es_la_ventana_de_entrenamiento_de_p1():
    assert ts(DRIFT_REF_END) == ts(P1_VAL_START)
    assert _meses(DRIFT_REF_START, DRIFT_REF_END) <= set(VIAJES_URLS)


def test_actual_es_posterior_a_la_referencia_y_mismos_meses_del_anio():
    assert ts(DRIFT_REF_END) <= ts(DRIFT_ACTUAL_START) < ts(DRIFT_ACTUAL_END)
    meses_ref = {m[-2:] for m in _meses(DRIFT_REF_START, DRIFT_REF_END)}
    meses_act = {m[-2:] for m in _meses(DRIFT_ACTUAL_START, DRIFT_ACTUAL_END)}
    assert meses_ref == meses_act  # comparación interanual, sin efecto de estacionalidad


def test_holdout_ab_no_lo_ve_ninguna_version_al_entrenar():
    assert ts(V2_TRAIN_START) < ts(V2_VAL_START) < ts(AB_EVAL_START) < ts(AB_EVAL_END)
    assert ts(AB_EVAL_START) >= ts(P1_TEST_END)  # v1 no usó datos posteriores a 2025


def test_urls_del_periodo_actual_cubren_drift_y_ab():
    necesarios = _meses(DRIFT_ACTUAL_START, DRIFT_ACTUAL_END) | _meses(AB_EVAL_START, AB_EVAL_END)
    assert necesarios <= set(VIAJES_ACTUAL_URLS)


def test_feriados_cubren_el_periodo_actual():
    feriados = pd.to_datetime(P1_FERIADOS)
    en_actual = feriados[(feriados >= ts(DRIFT_ACTUAL_START)) & (feriados < ts(AB_EVAL_END))]
    assert len(en_actual) > 0
    assert feriados.is_unique
