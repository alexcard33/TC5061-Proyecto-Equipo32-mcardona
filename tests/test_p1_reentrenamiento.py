"""Versión 2 de P1 (mibici.modeling.p1_reentrenamiento): particiones sin fuga y configuración.

Como el resto de las pruebas de modelado, no necesita datos reales ni servidor de MLflow.
"""

import io
import sys

import pandas as pd
import pytest

from mibici.config import (
    AB_EVAL_START,
    ALIAS_CHALLENGER,
    P1_MLFLOW_LINEAL_RUN_NAME,
    P1_MLFLOW_V2_RUN_NAME,
    REGISTRO_MODELOS,
    V2_TRAIN_START,
    V2_VAL_START,
)
from mibici.features import P1_FEATURES
from mibici.modeling import p1_reentrenamiento as v2
from mibici.modeling.mlflow_utils import consola_utf8
from tests.sinteticos import generar_viajes


@pytest.fixture
def particiones(monkeypatch):
    # Viajes de mediados de abril a septiembre de 2026: cubren train, val y el holdout del A/B.
    viajes = generar_viajes(n_estaciones=5, dias=150, inicio="2026-04-15")
    capacidad = pd.Series({i: 20 for i in range(1, 6)})
    monkeypatch.setattr(v2, "cargar_viajes_p1", lambda: viajes)
    monkeypatch.setattr(v2, "cargar_capacidad", lambda ruta: capacidad)
    return v2.preparar_datos()


def test_particiones_respetan_fechas_y_no_tocan_el_holdout(particiones):
    train, val = particiones
    f_train = train.index.get_level_values("fecha")
    f_val = val.index.get_level_values("fecha")

    assert f_train.min() >= pd.Timestamp(V2_TRAIN_START)
    assert f_train.max() < pd.Timestamp(V2_VAL_START)
    assert f_val.min() >= pd.Timestamp(V2_VAL_START)
    assert f_val.max() < pd.Timestamp(AB_EVAL_START)
    assert len(train) > 0 and len(val) > 0


def test_particiones_sin_filas_incompletas(particiones):
    train, val = particiones
    assert not train[P1_FEATURES].isna().any().any()
    assert not val[P1_FEATURES].isna().any().any()


def test_v2_usa_el_mismo_modelo_que_la_version_original():
    _, _, familia, run_original, params = v2.construir_modelo("poisson", {})
    assert familia == "glm_poisson"
    assert run_original == P1_MLFLOW_LINEAL_RUN_NAME
    assert params["alpha"] == 0.001 and params["max_iter"] == 1000


def test_registro_tras_la_promocion():
    """Después del A/B la versión reentrenada es la vigente y la original queda de challenger."""
    assert ALIAS_CHALLENGER == "challenger"
    assert REGISTRO_MODELOS["p1"]["champion"] == P1_MLFLOW_V2_RUN_NAME
    assert REGISTRO_MODELOS["p1"]["challenger"] == P1_MLFLOW_LINEAL_RUN_NAME


def test_baseline_no_se_reentrena():
    with pytest.raises(ValueError, match="no válido"):
        v2.entrenar(model="baseline")


def test_consola_utf8_permite_el_emoji_que_imprime_mlflow(monkeypatch):
    """En una consola cp1252 el enlace de MLflow con emoji fallaba al cerrar el run."""
    salida = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", salida)
    monkeypatch.setattr(sys, "stderr", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))

    consola_utf8()
    sys.stdout.write("\U0001f3c3 View run")
    sys.stdout.flush()

    assert salida.buffer.getvalue().decode("utf-8") == "\U0001f3c3 View run"
