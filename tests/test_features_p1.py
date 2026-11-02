"""Pruebas de las variables de P1: el panel es contiguo y no hay fuga de información."""

import pandas as pd
import pytest

from mibici.features import LeakageError, p1_agregar_features, p1_construir_panel_diario


@pytest.fixture
def viajes():
    # Estación 1: 1 viaje el día d (d = 1..20). Estación 2 empieza el día 10.
    filas = []
    for d in range(1, 21):
        filas += [{"inicio_viaje": pd.Timestamp(2025, 1, d, 8), "origen_id": 1}] * d
        if d >= 10:
            filas.append({"inicio_viaje": pd.Timestamp(2025, 1, d, 9), "origen_id": 2})
    return pd.DataFrame(filas)


def test_panel_contiguo_desde_primera_actividad(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    assert panel.loc[1].shape[0] == 20
    assert panel.loc[2].index.min() == pd.Timestamp("2025-01-10")
    assert panel.loc[(1, pd.Timestamp("2025-01-05")), "viajes"] == 5


def test_rezagos_solo_usan_dias_anteriores(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    df = p1_agregar_features(panel, pd.Series({1: 20, 2: 15}), [], [1, 7], [7])
    fila = df.loc[(1, pd.Timestamp("2025-01-15"))]
    assert fila["lag_1"] == 14
    assert fila["lag_7"] == 8
    assert fila["media_7d"] == pytest.approx(sum(range(8, 15)) / 7)  # días 8..14, no incluye el 15


def test_rezago_menor_al_horizonte_lanza_error(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    with pytest.raises(LeakageError):
        p1_agregar_features(panel, pd.Series(dtype=float), [], [0, 7], [7], horizonte=1)


# ---------- Casos adicionales de P1 ----------
def test_panel_rellena_con_cero_los_dias_sin_viajes():
    viajes = pd.DataFrame(
        {
            "inicio_viaje": pd.to_datetime(["2025-01-01 08:00", "2025-01-04 10:00"]),
            "origen_id": [7, 7],
        }
    )
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-06")
    assert panel.loc[7, "viajes"].tolist() == [1, 0, 0, 1, 0]


def test_panel_excluye_el_dia_final_y_viajes_fuera_de_rango(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-10")
    fechas = panel.index.get_level_values("fecha")
    assert fechas.max() == pd.Timestamp("2025-01-09")  # el fin es exclusivo
    assert 2 not in panel.index.get_level_values("estacion")  # empieza el día 10


def test_panel_cuenta_viajes_por_dia_calendario():
    """Un viaje a las 23:59 cuenta para ese día y uno a las 00:00 para el siguiente."""
    viajes = pd.DataFrame(
        {
            "inicio_viaje": pd.to_datetime(["2025-01-01 23:59:59", "2025-01-02 00:00:00"]),
            "origen_id": [1, 1],
        }
    )
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-03")
    assert panel["viajes"].tolist() == [1, 1]


def test_calendario_y_feriados(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    df = p1_agregar_features(panel, pd.Series({1: 20}), ["2025-01-06"], [1], [7])
    fila_lunes_feriado = df.loc[(1, pd.Timestamp("2025-01-06"))]
    fila_sabado = df.loc[(1, pd.Timestamp("2025-01-04"))]
    assert fila_lunes_feriado["dia_semana"] == 0 and fila_lunes_feriado["feriado"] == 1
    assert fila_sabado["fin_semana"] == 1 and fila_sabado["feriado"] == 0
    assert df["feriado"].sum() == 1  # solo la estación 1 existe el 6 de enero


def test_medias_moviles_requieren_ventana_completa(viajes):
    """Caso límite: sin 7 días previos completos la media queda NaN (no se inventa)."""
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    df = p1_agregar_features(panel, pd.Series({1: 20, 2: 15}), [], [1], [7])
    est1 = df.loc[1, "media_7d"]
    assert est1.iloc[:7].isna().all()
    assert est1.iloc[7] == pytest.approx(sum(range(1, 8)) / 7)  # días 1..7


def test_rezagos_no_cruzan_estaciones(viajes):
    """El primer día de la estación 2 no debe heredar viajes de la estación 1."""
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    df = p1_agregar_features(panel, pd.Series({1: 20, 2: 15}), [], [1], [7])
    assert pd.isna(df.loc[(2, pd.Timestamp("2025-01-10")), "lag_1"])


def test_dias_activa_y_capacidad_faltante(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    df = p1_agregar_features(panel, pd.Series({1: 20}), [], [1], [7])
    assert df.loc[2, "dias_activa"].tolist() == list(range(11))
    assert (df.loc[1, "capacidad"] == 20).all()
    assert (df.loc[2, "capacidad"] == 0).all()  # estación fuera del catálogo -> 0


def test_horizonte_mayor_exige_rezagos_mayores(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    with pytest.raises(LeakageError):
        p1_agregar_features(panel, pd.Series(dtype=float), [], [1, 7], [7], horizonte=2)
    df = p1_agregar_features(panel, pd.Series(dtype=float), [], [2, 7], [7], horizonte=2)
    assert "lag_2" in df.columns


def test_agregar_features_no_modifica_el_panel(viajes):
    panel = p1_construir_panel_diario(viajes, "2025-01-01", "2025-01-21")
    copia = panel.copy()
    p1_agregar_features(panel, pd.Series({1: 20}), [], [1], [7])
    pd.testing.assert_frame_equal(panel, copia)
