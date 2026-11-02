"""Pruebas de las variables de P3 (perfil de uso de cada estación para clustering)."""

import numpy as np
import pandas as pd
import pytest

from mibici.features import (
    P3_MODEL_FEATURES,
    p3_agregar_entropias,
    p3_construir_features,
    p3_entropia_normalizada,
    p3_filtrar_viajes,
)

FIN = pd.Timestamp("2025-03-01 23:59:59")


# ---------- p3_filtrar_viajes ----------
def test_filtrar_viajes_limites_inclusivos():
    inicio = pd.Timestamp("2025-01-01 08:00")
    duraciones = [59, 60, 3600, 86400, 86401]
    viajes = pd.DataFrame(
        {
            "inicio_viaje": [inicio] * 5,
            "fin_viaje": [inicio + pd.Timedelta(seconds=s) for s in duraciones],
        }
    )
    filtrados = p3_filtrar_viajes(viajes, 60, 86400)
    assert filtrados.index.tolist() == [1, 2, 3]


def test_filtrar_viajes_descarta_duracion_negativa_y_nat():
    inicio = pd.Timestamp("2025-01-01 08:00")
    viajes = pd.DataFrame(
        {
            "inicio_viaje": [inicio, inicio, inicio],
            "fin_viaje": [
                inicio - pd.Timedelta(minutes=5),
                pd.NaT,
                inicio + pd.Timedelta(minutes=5),
            ],
        }
    )
    assert p3_filtrar_viajes(viajes).index.tolist() == [2]


def test_filtrar_viajes_no_modifica_la_entrada(viajes_sinteticos):
    copia = viajes_sinteticos.copy()
    p3_filtrar_viajes(viajes_sinteticos)
    pd.testing.assert_frame_equal(viajes_sinteticos, copia)


# ---------- p3_construir_features ----------
def test_features_tienen_columnas_y_rangos_validos(viajes_sinteticos):
    features, cobertura = p3_construir_features(viajes_sinteticos, FIN)
    assert list(features.columns) == P3_MODEL_FEATURES
    assert features.notna().all().all()
    assert (features["salidas_dia"] > 0).all() and (features["llegadas_dia"] > 0).all()
    assert features["balance_relativo"].between(-1, 1).all()
    assert features["share_fin_semana"].between(0, 1).all()
    for tipo in ("salida", "llegada"):
        bloques = [f"{tipo}_{b}_share" for b in ("manana", "mediodia", "tarde", "noche")]
        np.testing.assert_allclose(features[bloques].sum(axis=1), 1.0)
    assert set(cobertura.columns) >= {
        "estacion",
        "dias_historial",
        "dias_sin_actividad",
        "eligible_p3",
    }


def test_features_capturan_el_patron_horario(viajes_sinteticos):
    """Las estaciones pares salen en la mañana y las impares en la tarde (ver conftest)."""
    features, _ = p3_construir_features(viajes_sinteticos, FIN)
    pares = features.loc[features.index % 2 == 0]
    impares = features.loc[features.index % 2 == 1]
    assert (pares["salida_manana_share"] > 0.8).all()
    assert (impares["salida_tarde_share"] > 0.8).all()


def _viajes_estacion(estacion, dias, inicio="2025-01-01 08:00"):
    inicio = pd.Timestamp(inicio)
    t = [inicio + pd.Timedelta(days=d) for d in dias]
    return pd.DataFrame(
        {
            "inicio_viaje": t,
            "fin_viaje": [x + pd.Timedelta(minutes=10) for x in t],
            "origen_id": estacion,
            "destino_id": estacion,
        }
    )


def test_elegibilidad_excluye_inactivas_y_con_poca_historia():
    viajes = pd.concat(
        [
            _viajes_estacion(1, range(0, 59)),  # activa hasta el final: elegible
            _viajes_estacion(2, range(0, 10)),  # inactiva ~50 días > 30 (umbral de prueba)
            _viajes_estacion(3, range(50, 59)),  # solo 9 días de historial < 30
        ],
        ignore_index=True,
    )
    features, cobertura = p3_construir_features(
        viajes, FIN, max_inactivity_days=30, min_history_days=30
    )
    assert features.index.tolist() == [1]
    elegibles = cobertura.set_index("estacion")["eligible_p3"]
    assert elegibles.to_dict() == {1: True, 2: False, 3: False}


def test_sin_estaciones_elegibles_lanza_error():
    viajes = _viajes_estacion(1, range(0, 5))
    with pytest.raises(ValueError, match="No hay estaciones elegibles"):
        p3_construir_features(viajes, FIN, min_history_days=30)


def test_intensidad_normalizada_por_dias_de_exposicion():
    """Una estación que abrió a mitad del periodo no se ve 'menos intensa' por eso."""
    fin = pd.Timestamp("2025-01-31 23:59:59")
    vieja = _viajes_estacion(1, range(0, 31))  # 1 salida diaria desde el día 1
    nueva = _viajes_estacion(2, range(0, 16), inicio="2025-01-16 08:00")  # 1 diaria desde el 16
    features, _ = p3_construir_features(
        pd.concat([vieja, nueva], ignore_index=True), fin, min_history_days=10
    )
    assert features.loc[1, "salidas_dia"] == pytest.approx(1.0)
    assert features.loc[2, "salidas_dia"] == pytest.approx(1.0)


# ---------- entropías ----------
@pytest.mark.parametrize(
    "valores, esperado",
    [
        ([5, 5, 5, 5], 0.0),  # una sola categoría
        ([1, 2, 3, 4], 1.0),  # uniforme: máxima diversidad
        ([1, 2], 1.0),
    ],
)
def test_entropia_normalizada_extremos(valores, esperado):
    assert p3_entropia_normalizada(pd.Series(valores)) == pytest.approx(esperado)


def test_entropia_normalizada_intermedia_y_serie_vacia():
    valor = p3_entropia_normalizada(pd.Series([1, 1, 1, 2]))
    assert 0 < valor < 1
    assert p3_entropia_normalizada(pd.Series([], dtype=int)) == 0.0


def test_agregar_entropias(viajes_sinteticos):
    features, _ = p3_construir_features(viajes_sinteticos, FIN)
    con_entropia = p3_agregar_entropias(features, viajes_sinteticos)
    assert {"entropia_destinos", "entropia_origenes"} <= set(con_entropia.columns)
    assert con_entropia["entropia_destinos"].between(0, 1).all()
    assert "entropia_destinos" not in features.columns  # no modifica la entrada
    # destinos aleatorios uniformes entre 6 estaciones -> entropía cercana a 1
    assert (con_entropia["entropia_destinos"] > 0.95).all()
