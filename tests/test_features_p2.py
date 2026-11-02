"""Pruebas de las variables y la etiqueta de P2 (riesgo de vaciado/llenado por hora)."""

import numpy as np
import pandas as pd
import pytest

from mibici.features import (
    P2_FEATURES,
    p2_agregar_etiqueta,
    p2_agregar_features,
    p2_ajustar_historico,
    p2_aplicar_historico,
    p2_construir_panel,
    p2_historico_pasado,
)


def _viaje(inicio, fin, origen, destino):
    return {
        "inicio_viaje": pd.Timestamp(inicio),
        "fin_viaje": pd.Timestamp(fin),
        "origen_id": origen,
        "destino_id": destino,
    }


@pytest.fixture
def viajes_simples():
    return pd.DataFrame(
        [
            _viaje("2025-01-01 08:10", "2025-01-01 08:40", 1, 2),
            _viaje("2025-01-01 08:20", "2025-01-01 09:05", 1, 2),
            _viaje("2025-01-01 10:00", "2025-01-01 10:30", 2, 1),
        ]
    )


# ---------- p2_construir_panel ----------
def test_panel_horario_cuenta_salidas_llegadas_y_neto(viajes_simples):
    panel = p2_construir_panel(viajes_simples, "2025-01-01", "2025-01-02")
    assert panel.loc[(1, pd.Timestamp("2025-01-01 08:00")), "salidas"] == 2
    assert panel.loc[(2, pd.Timestamp("2025-01-01 08:00")), "llegadas"] == 1
    assert panel.loc[(2, pd.Timestamp("2025-01-01 09:00")), "llegadas"] == 1
    assert panel.loc[(1, pd.Timestamp("2025-01-01 08:00")), "neto"] == -2
    assert (panel["neto"] == panel["llegadas"] - panel["salidas"]).all()


def test_panel_rellena_horas_vacias_y_empieza_en_primera_actividad(viajes_simples):
    panel = p2_construir_panel(viajes_simples, "2025-01-01", "2025-01-02")
    est1 = panel.loc[1]
    est2 = panel.loc[2]
    assert est1.index.min() == pd.Timestamp("2025-01-01 08:00")
    assert est2.index.min() == pd.Timestamp("2025-01-01 08:00")  # su primera llegada
    assert len(est1) == 16  # 08:00..23:00, contiguo
    assert est1.loc[pd.Timestamp("2025-01-01 12:00"), ["salidas", "llegadas"]].sum() == 0
    assert panel.isna().sum().sum() == 0


def test_panel_conserva_flujo_total(viajes_sinteticos):
    panel = p2_construir_panel(viajes_sinteticos, "2025-01-01", "2025-03-05")
    assert panel["salidas"].sum() == len(viajes_sinteticos)
    assert panel["llegadas"].sum() == len(viajes_sinteticos)
    assert panel["neto"].sum() == 0


# ---------- p2_agregar_etiqueta ----------
@pytest.fixture
def panel_manual():
    horas = pd.date_range("2025-01-06 00:00", periods=6, freq="h")
    neto = [0, -3, -3, 1, 4, 4]
    idx = pd.MultiIndex.from_product([[1], horas], names=["estacion", "hora"])
    return pd.DataFrame({"salidas": [0] * 6, "llegadas": [0] * 6, "neto": neto}, index=idx)


def test_etiqueta_umbral_y_tipo(panel_manual):
    df = p2_agregar_etiqueta(panel_manual, horizonte=2, capacidad=pd.Series({1: 10}), fraccion=0.4)
    # neto_futuro(t) = neto(t+1) + neto(t+2); umbral = 4
    assert df["neto_futuro"].tolist() == [-6, -2, 5, 8]
    assert df["riesgo"].tolist() == [1, 0, 1, 1]
    assert df["tipo_riesgo"].tolist() == ["sin_bicis", "normal", "sin_anclajes", "sin_anclajes"]


def test_etiqueta_umbral_es_inclusivo(panel_manual):
    """Caso límite: |neto_futuro| igual al umbral ya es riesgo."""
    df = p2_agregar_etiqueta(panel_manual, horizonte=1, capacidad=pd.Series({1: 10}), fraccion=0.3)
    fila = df.loc[(1, pd.Timestamp("2025-01-06 00:00"))]
    assert fila["neto_futuro"] == -3 and fila["riesgo"] == 1


def test_etiqueta_descarta_horas_sin_ventana_futura(panel_manual):
    df = p2_agregar_etiqueta(panel_manual, horizonte=2, capacidad=pd.Series({1: 10}), fraccion=0.4)
    assert len(df) == len(panel_manual) - 2
    assert df.index.get_level_values("hora").max() == pd.Timestamp("2025-01-06 03:00")


def test_etiqueta_descarta_capacidad_cero_o_desconocida(panel_manual):
    for capacidad in (pd.Series({1: 0}), pd.Series({99: 10})):
        df = p2_agregar_etiqueta(panel_manual, 2, capacidad, 0.4)
        assert df.empty


# ---------- p2_agregar_features ----------
@pytest.mark.parametrize("horizonte", [0, 24, -1])
def test_features_rechaza_horizonte_invalido(panel_manual, horizonte):
    with pytest.raises(ValueError):
        p2_agregar_features(panel_manual.assign(capacidad=10), horizonte)


def test_features_rezagos_y_calendario(viajes_sinteticos, capacidad_sintetica):
    panel = p2_construir_panel(viajes_sinteticos, "2025-01-01", "2025-03-05")
    df = p2_agregar_etiqueta(panel, 2, capacidad_sintetica, 0.4)
    df = p2_agregar_features(df, 2)

    est = df.loc[2]
    t = pd.Timestamp("2025-02-10 09:00")  # lunes
    assert est.loc[t, "hora_dia"] == 9 and est.loc[t, "dia_semana"] == 0
    assert est.loc[t, "fin_semana"] == 0
    neto = panel.loc[2, "neto"]
    assert est.loc[t, "neto_h1"] == neto.loc[t - pd.Timedelta(hours=1)]
    assert est.loc[t, "neto_3h"] == neto.loc[t - pd.Timedelta(hours=2) : t].sum()
    assert est.loc[t, "neto_3h_rel"] == pytest.approx(est.loc[t, "neto_3h"] / 17)
    # misma ventana futura de ayer: neto de t-23 h y t-22 h
    ayer = neto.loc[t - pd.Timedelta(hours=23)] + neto.loc[t - pd.Timedelta(hours=22)]
    assert est.loc[t, "neto_futuro_ayer"] == ayer


def test_features_no_usan_el_futuro(viajes_sinteticos, capacidad_sintetica):
    """Si se altera el flujo posterior a t, las variables en t no cambian."""
    panel = p2_construir_panel(viajes_sinteticos, "2025-01-01", "2025-03-05")
    t = pd.Timestamp("2025-02-10 09:00")
    alterado = panel.copy()
    futuro = alterado.index.get_level_values("hora") > t
    alterado.loc[futuro, "neto"] = 999

    def features(p):
        df = p2_agregar_features(p2_agregar_etiqueta(p, 2, capacidad_sintetica, 0.4), 2)
        cols = [c for c in P2_FEATURES if c in df.columns]
        return df.loc[(2, t), cols]

    pd.testing.assert_series_equal(features(panel), features(alterado))


# ---------- histórico ----------
@pytest.fixture
def train_historico():
    horas = pd.date_range("2025-01-06 08:00", periods=4, freq="D")  # 4 lunes-jueves a las 8
    idx = pd.MultiIndex.from_product([[1], horas], names=["estacion", "hora"])
    return pd.DataFrame(
        {
            "riesgo": [1, 0, 1, 0],
            "neto_futuro": [-6.0, 0.0, 4.0, 2.0],
            "hora_dia": 8,
            "fin_semana": 0,
        },
        index=idx,
    )


def test_historico_pasado_solo_usa_filas_previas(train_historico):
    df = p2_historico_pasado(train_historico)
    assert np.isnan(df["tasa_riesgo_hist"].iloc[0])  # primera fila: sin pasado
    assert df["tasa_riesgo_hist"].iloc[1:].tolist() == [1.0, 0.5, pytest.approx(2 / 3)]
    assert df["neto_futuro_medio_hist"].iloc[1:].tolist() == [-6.0, -3.0, pytest.approx(-2 / 3)]


def test_historico_ajustado_y_aplicado_sin_recalcular(train_historico):
    historico = p2_ajustar_historico(train_historico)
    assert historico.loc[0, "tasa_riesgo_hist"] == 0.5
    assert historico.loc[0, "neto_futuro_medio_hist"] == 0.0

    val = pd.DataFrame(
        {"hora_dia": [8, 9], "fin_semana": [0, 0], "riesgo": [1, 1]},
        index=pd.MultiIndex.from_tuples(
            [(1, pd.Timestamp("2025-02-03 08:00")), (1, pd.Timestamp("2025-02-03 09:00"))],
            names=["estacion", "hora"],
        ),
    )
    unido = p2_aplicar_historico(val, historico)
    assert unido["tasa_riesgo_hist"].iloc[0] == 0.5  # viene de TRAIN, no de VAL
    assert np.isnan(unido["tasa_riesgo_hist"].iloc[1])  # grupo sin historia en TRAIN
    assert unido.index.names == ["estacion", "hora"]
