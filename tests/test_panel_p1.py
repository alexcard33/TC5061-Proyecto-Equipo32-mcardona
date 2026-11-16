"""Panel continuo de P1 (mibici.panel_p1), compartido por drift, reentrenamiento y A/B.

No depende de MLflow ni de Evidently, así que corre en el entorno principal y en el de
monitoreo.
"""

import pandas as pd

from mibici.config import DRIFT_ACTUAL_START
from mibici.features import P1_FEATURES
from mibici.panel_p1 import construir_features, cortar
from tests.sinteticos import generar_viajes


def test_construir_features_usa_la_capacidad_de_cada_periodo():
    viajes = generar_viajes(n_estaciones=4, dias=90, inicio="2025-11-15")
    capacidad_ref = pd.Series({i: 10 for i in range(1, 5)})
    capacidad_actual = pd.Series({i: 25 for i in range(1, 5)})

    df = construir_features(viajes, capacidad_ref, capacidad_actual, fin="2026-02-13")

    fecha = df.index.get_level_values("fecha")
    assert (df.loc[fecha < pd.Timestamp(DRIFT_ACTUAL_START), "capacidad"] == 10).all()
    assert (df.loc[fecha >= pd.Timestamp(DRIFT_ACTUAL_START), "capacidad"] == 25).all()
    assert set(P1_FEATURES) <= set(df.columns)


def test_construir_features_no_incluye_fechas_desde_fin():
    """Los datos posteriores a fin no entran al panel, aunque vengan en los viajes."""
    viajes = generar_viajes(n_estaciones=3, dias=60, inicio="2026-06-15")
    capacidad = pd.Series({i: 20 for i in range(1, 4)})

    df = construir_features(viajes, capacidad, capacidad, fin="2026-07-20")

    assert df.index.get_level_values("fecha").max() < pd.Timestamp("2026-07-20")


def test_cortar_incluye_inicio_y_excluye_fin():
    fechas = pd.date_range("2026-07-30", "2026-08-03")
    indice = pd.MultiIndex.from_product([[1], fechas], names=["estacion", "fecha"])
    df = pd.DataFrame({"viajes": range(len(fechas))}, index=indice)

    corte = cortar(df, "2026-07-31", "2026-08-02")

    assert list(corte.index.get_level_values("fecha")) == [
        pd.Timestamp("2026-07-31"),
        pd.Timestamp("2026-08-01"),
    ]
