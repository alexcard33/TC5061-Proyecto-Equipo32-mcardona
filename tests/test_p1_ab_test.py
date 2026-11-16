"""A/B testing de P1 (mibici.modeling.p1_ab_test).

Cubren la comparación pareada, la regla de decisión y las tablas sin necesitar datos reales
ni servidor de MLflow.
"""

import json

import numpy as np
import pandas as pd
import pytest

from mibici.config import AB_DIR, AB_EVAL_END, AB_EVAL_START, REGISTRO_MODELOS
from mibici.features import P1_FEATURES
from mibici.modeling import p1_ab_test as ab
from tests.sinteticos import generar_viajes


def _holdout_sintetico(n_estaciones=6, seed=0):
    """Holdout estación-día de agosto a octubre de 2026 con la variable viajes."""
    rng = np.random.default_rng(seed)
    fechas = pd.date_range(AB_EVAL_START, AB_EVAL_END, freq="D", inclusive="left")
    indice = pd.MultiIndex.from_product(
        [range(1, n_estaciones + 1), fechas], names=["estacion", "fecha"]
    )
    return pd.DataFrame({"viajes": rng.poisson(20, len(indice)).astype(float)}, index=indice)


def _predicciones(holdout, error_champion, error_challenger, seed=1):
    rng = np.random.default_rng(seed)
    y = holdout["viajes"].to_numpy()
    ruido = rng.normal(0, 1, len(y))
    return y + error_champion + ruido, y + error_challenger + ruido


def test_challenger_claramente_mejor_se_promueve():
    holdout = _holdout_sintetico()
    champion, challenger = _predicciones(holdout, error_champion=5, error_challenger=1)

    comparacion = ab.comparar(ab.errores_diarios(holdout, champion, challenger))
    decision = ab.decidir(comparacion)

    assert comparacion["delta_mae"] < 0
    assert comparacion["ic_superior"] < 0
    assert comparacion["p_wilcoxon"] < 0.05
    assert decision["version_vigente"] == "challenger"


def test_challenger_peor_no_se_promueve():
    holdout = _holdout_sintetico()
    champion, challenger = _predicciones(holdout, error_champion=1, error_challenger=5)

    comparacion = ab.comparar(ab.errores_diarios(holdout, champion, challenger))

    assert comparacion["ic_inferior"] > 0
    assert ab.decidir(comparacion)["version_vigente"] == "champion"


def test_sin_diferencia_se_mantiene_la_version_vigente():
    holdout = _holdout_sintetico()
    champion, _ = _predicciones(holdout, error_champion=2, error_challenger=2)

    comparacion = ab.comparar(ab.errores_diarios(holdout, champion, champion.copy()))

    assert comparacion["delta_mae"] == 0
    assert comparacion["ic_inferior"] <= 0 <= comparacion["ic_superior"]
    assert comparacion["p_wilcoxon"] == 1.0
    assert ab.decidir(comparacion)["version_vigente"] == "champion"


def test_bootstrap_es_reproducible():
    holdout = _holdout_sintetico()
    champion, challenger = _predicciones(holdout, error_champion=3, error_challenger=2)
    diario = ab.errores_diarios(holdout, champion, challenger)

    assert ab.comparar(diario) == ab.comparar(diario)


def test_metricas_por_segmento_cubre_periodos_y_estaciones():
    holdout = _holdout_sintetico(n_estaciones=6)
    champion, challenger = _predicciones(holdout, error_champion=3, error_challenger=2)
    predicciones = {"champion": champion, "challenger": challenger}

    tabla = ab.metricas_por_segmento(holdout, predicciones, vistas={1, 2, 3, 4})

    assert set(tabla["periodo"]) == {"ago-oct", "2026-08", "2026-09", "2026-10"}
    assert set(tabla["segmento"]) == {"todas", "conocidas", "nuevas"}
    assert {"mae", "rmse", "wape", "r2", "sesgo"} <= set(tabla.columns)
    total = tabla.query("version == 'champion' and periodo == 'ago-oct' and segmento == 'todas'")
    assert int(total["filas"].iloc[0]) == len(holdout)
    nuevas = tabla.query("version == 'champion' and periodo == 'ago-oct' and segmento == 'nuevas'")
    assert int(nuevas["filas"].iloc[0]) == len(holdout) // 3  # estaciones 5 y 6 de 6


def test_tabla_markdown_incluye_metricas_y_decision():
    holdout = _holdout_sintetico()
    champion, challenger = _predicciones(holdout, error_champion=3, error_challenger=1)
    predicciones = {"champion": champion, "challenger": challenger}
    metricas = ab.metricas_por_segmento(holdout, predicciones, vistas={1, 2, 3})
    comparacion = ab.comparar(ab.errores_diarios(holdout, champion, challenger))
    etiquetas = {"champion": "champion (versión 1)", "challenger": "challenger (versión 3)"}

    texto = ab.tabla_markdown(metricas, etiquetas, comparacion, ab.decidir(comparacion))

    assert "| mae |" in texto and "champion (versión 1)" in texto
    assert "Decisión: queda como vigente el challenger" in texto


def test_el_champion_configurado_es_el_ganador_del_ab():
    """La configuración del registry debe coincidir con la evidencia versionada del A/B."""
    ruta = AB_DIR / "p1_ab_resumen.json"
    if not ruta.exists():
        pytest.skip("Todavía no se ha corrido el A/B (python -m mibici.modeling.p1_ab_test)")
    resumen = json.loads(ruta.read_text(encoding="utf-8"))
    rol_ganador = resumen["decision"]["version_vigente"]
    ganador = resumen["versiones"][rol_ganador]["run_name"]
    assert REGISTRO_MODELOS["p1"]["champion"] == ganador


@pytest.fixture
def holdout_real(monkeypatch):
    # Cuatro estaciones con viajes de junio a octubre de 2026; solo la 1 tiene además viajes
    # en marzo de 2025, dentro de la ventana de entrenamiento de la versión vigente.
    viajes = pd.concat(
        [
            generar_viajes(n_estaciones=4, dias=150, inicio="2026-06-01"),
            generar_viajes(n_estaciones=4, dias=20, inicio="2025-03-01").assign(origen_id=1),
        ],
        ignore_index=True,
    )
    capacidad = pd.Series({i: 20 for i in range(1, 5)})
    monkeypatch.setattr(ab, "cargar_viajes_p1", lambda: viajes)
    monkeypatch.setattr(ab, "cargar_capacidad", lambda ruta: capacidad)
    return ab.preparar_holdout()


def test_preparar_holdout_respeta_fechas_y_marca_estaciones_vistas(holdout_real):
    holdout, vistas = holdout_real
    fecha = holdout.index.get_level_values("fecha")

    assert fecha.min() >= pd.Timestamp(AB_EVAL_START)
    assert fecha.max() < pd.Timestamp(AB_EVAL_END)
    assert not holdout[P1_FEATURES].isna().any().any()
    # Solo la estación 1 tiene viajes en la ventana de entrenamiento de la versión vigente.
    assert vistas == {1}
