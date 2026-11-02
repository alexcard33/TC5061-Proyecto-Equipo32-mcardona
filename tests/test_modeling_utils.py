"""Pruebas de las utilidades de entrenamiento y evaluación (``mibici.modeling``).

Cubren las piezas que no requieren datos reales ni un servidor de MLflow:
parseo de hiperparámetros, tags, particiones temporales, métricas, elección de
umbral y la fábrica de modelos de cada pregunta.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import AgglomerativeClustering, KMeans
import typer

from mibici.config import P2_HORIZONTE_HORAS, SEED
from mibici.features import P3_MODEL_FEATURES
from mibici.modeling import p1_regresion as p1
from mibici.modeling import p2_clasificacion as p2
from mibici.modeling import p3_clustering as p3
from mibici.modeling.mlflow_utils import ETAPAS, tags_estandar
from mibici.modeling.train import _valor, parsear_params


# ---------- train.parsear_params ----------
@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("learning_rate=0.1,max_iter=200", {"learning_rate": 0.1, "max_iter": 200}),
        ("linkage=ward, early_stopping=false", {"linkage": "ward", "early_stopping": False}),
        ('{"n_clusters": 4, "n_init": 20}', {"n_clusters": 4, "n_init": 20}),
        ("max_depth=none", {"max_depth": None}),
        ("'alpha=0.001'", {"alpha": 0.001}),  # comillas que deja cmd de Windows
        ("^alpha=0.001^", {"alpha": 0.001}),
    ],
)
def test_parsear_params_formatos_validos(texto, esperado):
    assert parsear_params(texto) == esperado


@pytest.mark.parametrize("texto", ["", "  ", "none", "{}", "''"])
def test_parsear_params_vacio(texto):
    assert parsear_params(texto) == {}


@pytest.mark.parametrize("texto", ["learning_rate", "a=1,b", "[1, 2]"])
def test_parsear_params_invalido(texto):
    with pytest.raises(typer.BadParameter):
        parsear_params(texto)


def test_parsear_params_conserva_igual_en_el_valor():
    assert parsear_params("expr=a=b") == {"expr": "a=b"}


@pytest.mark.parametrize(
    "texto, esperado",
    [("3", 3), ("0.5", 0.5), ("TRUE", True), ("Null", None), ("ward", "ward"), ("[1,2]", [1, 2])],
)
def test_valor(texto, esperado):
    assert _valor(texto) == esperado


# ---------- mlflow_utils.tags_estandar ----------
@pytest.mark.parametrize("etapa", ETAPAS)
def test_tags_estandar_validos(etapa):
    assert tags_estandar("p1", "glm", etapa) == {
        "question": "p1", "model_family": "glm", "stage_of_work": etapa,
    }  # fmt: skip


def test_tags_estandar_etapa_invalida():
    with pytest.raises(ValueError):
        tags_estandar("p1", "glm", "produccion")


# ---------- P1: cortar, evaluar, baseline y fábrica de modelos ----------
def _serie_diaria(valores, inicio="2025-01-01", estaciones=(1,)):
    fechas = pd.date_range(inicio, periods=len(valores) // len(estaciones), freq="D")
    idx = pd.MultiIndex.from_product([list(estaciones), fechas], names=["estacion", "fecha"])
    return pd.Series(valores, index=idx, dtype="float64", name="viajes")


def test_p1_cortar_intervalo_semiabierto():
    df = _serie_diaria(range(10)).to_frame()
    corte = p1.cortar(df, "2025-01-03", "2025-01-06")
    fechas = corte.index.get_level_values("fecha")
    assert fechas.min() == pd.Timestamp("2025-01-03")
    assert fechas.max() == pd.Timestamp("2025-01-05")  # el fin es exclusivo


def test_p1_evaluar_prediccion_perfecta():
    y = _serie_diaria([3, 5, 2, 8, 4, 6], estaciones=(1, 2))
    m = p1.evaluar(y, y.to_numpy(), "val")
    assert m["val_mae"] == 0 and m["val_rmse"] == 0 and m["val_wape"] == 0
    assert m["val_r2"] == pytest.approx(1.0) and m["val_sesgo"] == 0
    assert m["val_mape_total_dia"] == 0


def test_p1_evaluar_valores_conocidos_y_recorte_de_negativos():
    y = _serie_diaria([2.0, 4.0])
    m = p1.evaluar(y, np.array([-3.0, 6.0]), "test")  # -3 se recorta a 0
    assert m["test_mae"] == pytest.approx(2.0)
    assert m["test_wape"] == pytest.approx(4 / 6)
    assert m["test_sesgo"] == pytest.approx(0.0)
    assert set(m) == {
        f"test_{k}" for k in ("mae", "rmse", "wape", "r2", "sesgo", "mape_total_dia")
    }


def test_p1_baseline_devuelve_lag_7():
    x = pd.DataFrame({"lag_7": [1, 5, 0], "lag_1": [9, 9, 9]})
    modelo = p1.BaselineNaiveSemanal("lag_7").fit(x)
    np.testing.assert_array_equal(modelo.predict(x), [1.0, 5.0, 0.0])
    assert modelo.predict(x).dtype == np.float64


@pytest.mark.parametrize(
    "nombre, etapa",
    [
        ("baseline", "baseline"),
        ("poisson", "candidate"),
        ("hgb", "candidate"),
        ("rf", "candidate"),
    ],
)
def test_p1_construir_modelo(nombre, etapa):
    modelo, etapa_obtenida, familia, run_name, params_log = p1.construir_modelo(nombre, {})
    assert etapa_obtenida == etapa and etapa_obtenida in ETAPAS
    assert run_name.startswith("p1_") and "modelo" in params_log
    assert hasattr(modelo, "fit") and hasattr(modelo, "predict")


def test_p1_construir_modelo_sobrescribe_params_y_fija_semilla():
    modelo, *_, params_log = p1.construir_modelo("hgb", {"learning_rate": 0.2})
    assert modelo.learning_rate == 0.2 and params_log["learning_rate"] == 0.2
    assert modelo.random_state == SEED


def test_p1_construir_modelo_errores():
    with pytest.raises(ValueError, match="desconocido"):
        p1.construir_modelo("xgboost", {})
    with pytest.raises(ValueError, match="baseline"):
        p1.construir_modelo("baseline", {"lag": 1})


# ---------- P2: cortar, umbral, evaluar y fábrica de modelos ----------
def test_p2_cortar_descarta_horas_cuya_ventana_cruza_el_fin():
    horas = pd.date_range("2025-01-01 00:00", "2025-01-01 23:00", freq="h")
    df = pd.DataFrame(
        {"x": 1}, index=pd.MultiIndex.from_product([[1], horas], names=["estacion", "hora"])
    )
    corte = p2.cortar(df, "2025-01-01", "2025-01-02")
    assert corte.index.get_level_values("hora").max() == pd.Timestamp("2025-01-02") - pd.Timedelta(
        hours=P2_HORIZONTE_HORAS + 1
    )
    assert len(corte) == 24 - P2_HORIZONTE_HORAS


def test_p2_elegir_umbral_separa_clases_perfectamente():
    y = pd.Series([0, 0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    umbral = p2.elegir_umbral(y, prob)
    assert 0.3 < umbral <= 0.7
    assert ((prob >= umbral).astype(int) == y.to_numpy()).all()


def test_p2_evaluar_metricas_y_accuracy_trivial():
    y = pd.Series([0, 0, 0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.6, 0.1, 0.8, 0.3])
    tipo = pd.Series(["normal", "normal", "normal", "normal", "sin_bicis", "sin_anclajes"])
    m = p2.evaluar(y, prob, 0.5, "val", tipo)
    assert m["val_precision"] == pytest.approx(0.5)
    assert m["val_recall"] == pytest.approx(0.5)
    assert m["val_f1"] == pytest.approx(0.5)
    assert m["val_accuracy_trivial"] == pytest.approx(4 / 6)
    assert m["val_prevalencia"] == pytest.approx(2 / 6)
    assert m["val_recall_sin_bicis"] == 1.0 and m["val_recall_sin_anclajes"] == 0.0


def test_p2_evaluar_sin_predicciones_positivas_no_falla():
    """Caso límite: umbral tan alto que nunca alerta -> precision 0 sin excepción."""
    y = pd.Series([0, 1, 0, 1])
    tipo = pd.Series(["normal", "sin_bicis", "normal", "sin_anclajes"])
    m = p2.evaluar(y, np.array([0.1, 0.2, 0.3, 0.4]), 0.99, "test", tipo)
    assert m["test_precision"] == 0 and m["test_recall"] == 0


@pytest.mark.parametrize("nombre", ["baseline", "hgb", "rf"])
def test_p2_construir_modelo(nombre):
    modelo, etapa, familia, run_name, params_log = p2.construir_modelo(nombre, {})
    assert etapa in ETAPAS and run_name.startswith("p2_")
    assert hasattr(modelo, "predict_proba")


def test_p2_construir_modelo_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        p2.construir_modelo("svm", {})


# ---------- P3: preparación, perfiles, predicción y candidatos ----------
@pytest.fixture
def features_p3():
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(rng.random((20, len(P3_MODEL_FEATURES))), columns=P3_MODEL_FEATURES)


def test_p3_preparar_features_estandariza_y_reutiliza_scaler(features_p3):
    x, scaler = p3.preparar_features(features_p3)
    np.testing.assert_allclose(x.mean(), 0, atol=1e-12)
    np.testing.assert_allclose(x.std(ddof=0), 1)
    # aplicar el scaler de TRAIN a otros datos no lo reajusta
    otros = features_p3 * 2
    x2, scaler2 = p3.preparar_features(otros, scaler=scaler)
    assert scaler2 is scaler
    assert not np.allclose(x2.mean(), 0)


def test_p3_preparar_features_aplica_log1p_a_intensidades(features_p3):
    f = features_p3.copy()
    f["salidas_dia"] = [0.0, 1e6] * 10  # valores extremos: el log los acerca
    x, _ = p3.preparar_features(f)
    assert x["salidas_dia"].abs().max() < 1.01  # dos grupos simétricos


def test_p3_asignar_perfiles():
    f = pd.DataFrame(0.1, index=range(4), columns=P3_MODEL_FEATURES)
    f.loc[0, ["salidas_dia", "llegadas_dia"]] = 50  # alta intensidad
    f.loc[1, "share_fin_semana"] = 0.9  # fin de semana
    f.loc[2, ["salida_manana_share", "llegada_tarde_share"]] = 0.8  # sale en la mañana
    perfiles = p3.asignar_perfiles(f, pd.Series([10, 11, 12, 13]))
    assert perfiles == {
        10: "Alta intensidad balanceada",
        11: "Fin de semana y mediodia",
        12: "Salidas matutinas / llegadas vespertinas",
        13: "Llegadas matutinas / salidas vespertinas",
    }
    assert set(perfiles.values()) == set(p3.COLORES_PERFIL)


def test_p3_predecir_con_y_sin_metodo_predict(features_p3):
    x, _ = p3.preparar_features(features_p3)
    kmeans = KMeans(n_clusters=3, n_init=5, random_state=SEED).fit(x)
    np.testing.assert_array_equal(p3.predecir(kmeans, x), kmeans.labels_)
    agglo = AgglomerativeClustering(n_clusters=3).fit(x)
    assert len(set(p3.predecir(agglo, x))) == 3  # se reagrupa x


@pytest.mark.parametrize("nombre", ["kmeans", "agglomerative", "gmm"])
def test_p3_config_candidato(nombre):
    familia, crear, params_log, run_name = p3.config_candidato(nombre, {})
    assert params_log["k"] == len(p3.COLORES_PERFIL)
    assert run_name.startswith("p3_")
    assert hasattr(crear(), "fit")


def test_p3_config_candidato_errores():
    with pytest.raises(ValueError, match="desconocido"):
        p3.config_candidato("dbscan", {})
    with pytest.raises(ValueError, match="k=3"):
        p3.config_candidato("kmeans", {"n_clusters": 3})
