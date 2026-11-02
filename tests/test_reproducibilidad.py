"""Reproducibilidad del entrenamiento: misma semilla => mismas predicciones y métricas.

"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score

from mibici.config import SEED
from mibici.features import (
    P1_FEATURES,
    P2_FEATURES,
    p1_agregar_features,
    p1_construir_panel_diario,
    p2_agregar_etiqueta,
    p2_agregar_features,
    p2_construir_panel,
    p2_historico_pasado,
    p3_construir_features,
)
from mibici.modeling import p1_regresion as p1
from mibici.modeling import p2_clasificacion as p2
from mibici.modeling import p3_clustering as p3
from tests.sinteticos import generar_viajes

# Hiperparámetros reducidos para que la suite sea rápida; la semilla y la lógica
# de construcción son las mismas que en producción.
PARAMS_RAPIDOS = {
    "hgb": {"max_iter": 30},
    "rf": {"n_estimators": 20, "n_jobs": 1},
}


# ---------- Datos: el feature engineering también debe ser determinista ----------
def _datos_p1(seed=SEED):
    viajes = generar_viajes(dias=70, seed=seed)
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-03-12")
    capacidad = pd.Series({i: 15 + i for i in range(1, 7)})
    df = p1_agregar_features(panel, capacidad, ["2025-02-03"], [1, 2, 7, 14], [7, 28])
    df = df.dropna(subset=P1_FEATURES)
    train = p1.cortar(df, "2025-01-01", "2025-02-25")
    val = p1.cortar(df, "2025-02-25", "2025-03-12")
    return (
        (train[P1_FEATURES].astype("float64"), train["viajes"]),
        (val[P1_FEATURES].astype("float64"), val["viajes"]),
    )


def _datos_p2(seed=SEED):
    viajes = generar_viajes(dias=21, seed=seed)
    capacidad = pd.Series({i: 8 for i in range(1, 7)})
    panel = p2_construir_panel(viajes, "2025-01-01", "2025-01-22")
    df = p2_agregar_features(p2_agregar_etiqueta(panel, 2, capacidad, 0.4), 2)
    df = p2_historico_pasado(df).dropna(subset=P2_FEATURES)
    hora = df.index.get_level_values("hora")
    corte = pd.Timestamp("2025-01-16")
    train, val = df.loc[hora < corte], df.loc[hora >= corte]
    return (
        (train[P2_FEATURES].astype("float64"), train["riesgo"]),
        (val[P2_FEATURES].astype("float64"), val["riesgo"], val["tipo_riesgo"]),
    )


def _datos_p3(seed=SEED):
    viajes = generar_viajes(n_estaciones=40, dias=60, viajes_por_dia=400, seed=seed)
    features, _ = p3_construir_features(viajes, pd.Timestamp("2025-03-01 23:59:59"))
    x, _ = p3.preparar_features(features)
    return x


def test_feature_engineering_es_determinista():
    for construir in (_datos_p1, _datos_p2):
        a, b = construir(), construir()
        for parte_a, parte_b in zip(a, b):
            for objeto_a, objeto_b in zip(parte_a, parte_b):
                if isinstance(objeto_a, pd.DataFrame):
                    pd.testing.assert_frame_equal(objeto_a, objeto_b)
                else:
                    pd.testing.assert_series_equal(objeto_a, objeto_b)
    pd.testing.assert_frame_equal(_datos_p3(), _datos_p3())


# ---------- P1: regresión ----------
def _entrenar_p1(nombre, datos, semilla=None):
    modelo, *_ = p1.construir_modelo(nombre, PARAMS_RAPIDOS.get(nombre, {}))
    if semilla is not None:
        modelo.set_params(random_state=semilla)
    (x_tr, y_tr), (x_val, y_val) = datos
    modelo.fit(x_tr, y_tr)
    pred = modelo.predict(x_val)
    return pred, p1.evaluar(y_val, pred, "val")


@pytest.mark.parametrize("nombre", ["baseline", "poisson", "hgb", "rf"])
def test_p1_misma_semilla_mismas_predicciones_y_metricas(nombre):
    pred_a, met_a = _entrenar_p1(nombre, _datos_p1())
    pred_b, met_b = _entrenar_p1(nombre, _datos_p1())
    np.testing.assert_array_equal(pred_a, pred_b)
    assert met_a == met_b
    assert np.isfinite(list(met_a.values())).all()


def test_p1_otra_semilla_cambia_el_bosque():
    datos = _datos_p1()
    pred_a, _ = _entrenar_p1("rf", datos)
    pred_b, _ = _entrenar_p1("rf", datos, semilla=SEED + 1)
    assert not np.array_equal(pred_a, pred_b)


# ---------- P2: clasificación ----------
def _entrenar_p2(nombre, datos, semilla=None):
    modelo, *_ = p2.construir_modelo(nombre, PARAMS_RAPIDOS.get(nombre, {}))
    if semilla is not None:
        modelo.set_params(random_state=semilla)
    (x_tr, y_tr), (x_val, y_val, tipo) = datos
    modelo.fit(x_tr, y_tr)
    prob = modelo.predict_proba(x_val)[:, 1]
    umbral = p2.elegir_umbral(y_val, prob)
    return prob, umbral, p2.evaluar(y_val, prob, umbral, "val", tipo)


@pytest.mark.parametrize("nombre", ["baseline", "hgb", "rf"])
def test_p2_misma_semilla_mismas_probabilidades_umbral_y_metricas(nombre):
    datos = _datos_p2()
    assert datos[0][1].nunique() == 2, "los datos sintéticos deben tener ambas clases"
    prob_a, umbral_a, met_a = _entrenar_p2(nombre, datos)
    prob_b, umbral_b, met_b = _entrenar_p2(nombre, _datos_p2())
    np.testing.assert_array_equal(prob_a, prob_b)
    assert umbral_a == umbral_b
    assert met_a == met_b


def test_p2_otra_semilla_cambia_el_bosque():
    datos = _datos_p2()
    prob_a, *_ = _entrenar_p2("rf", datos)
    prob_b, *_ = _entrenar_p2("rf", datos, semilla=SEED + 1)
    assert not np.array_equal(prob_a, prob_b)


# ---------- P3: clustering ----------
@pytest.mark.parametrize("nombre", ["kmeans", "gmm", "agglomerative"])
def test_p3_misma_semilla_mismos_grupos(nombre):
    x = _datos_p3()
    _, crear, _, _ = p3.config_candidato(nombre, {})
    etiquetas_a = p3.predecir(crear().fit(x), x)
    etiquetas_b = p3.predecir(crear().fit(_datos_p3()), x)
    np.testing.assert_array_equal(etiquetas_a, etiquetas_b)  # misma numeración, no solo ARI=1
    assert len(np.unique(etiquetas_a)) == len(p3.COLORES_PERFIL)


def test_p3_semilla_fija_aun_con_una_sola_inicializacion():
    """Con n_init=1 el resultado depende solo de la semilla: debe repetirse exacto."""
    x = _datos_p3()
    _, crear, _, _ = p3.config_candidato("kmeans", {"n_init": 1})
    a, b = crear().fit(x), crear().fit(x)
    np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)
    assert adjusted_rand_score(a.labels_, b.labels_) == 1.0


def test_modelos_clonados_conservan_la_semilla():
    """``clone`` (usado en búsquedas y en predecir) no pierde el random_state."""
    for modelo in (
        p1.construir_modelo("hgb", {})[0],
        p2.construir_modelo("rf", {})[0],
        p3.config_candidato("gmm", {})[1](),
    ):
        assert clone(modelo).get_params()["random_state"] == SEED
