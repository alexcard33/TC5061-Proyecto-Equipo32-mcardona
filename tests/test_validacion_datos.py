"""Validación de datos con DeepChecks (``mibici.validacion_datos``).

Requiere el entorno de validación (``requirements-validacion.txt``); en el entorno
principal estas pruebas se omiten automáticamente.

1. Los chequeos DETECTAN problemas inyectados a propósito (nulos, nulos mezclados,
   tipos mezclados, duplicados, drift y fuga temporal) y PASAN con datos limpios.
2. Los chequeos de dominio cuentan exactamente los problemas inyectados.
3. La validación completa (viajes, estaciones, P1, P2, P3) corre de punta a punta,
   genera los reportes y no tiene fallas críticas.
4. Con los datos reales (si existen en ``data/processed``), los chequeos críticos de
   dominio pasan.
"""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("deepchecks", reason="Requiere el entorno de requirements-validacion.txt")

from mibici.config import ESTACIONES_PARQUET, SEED, VIAJES_PARQUET  # noqa: E402
from mibici.validacion_datos import (  # noqa: E402
    Resumen,
    chequeos_dominio_estaciones,
    chequeos_dominio_viajes,
    validar,
    validar_integridad,
    validar_train_test,
)
from tests.sinteticos import generar_viajes  # noqa: E402


def _tabla(n=1000, seed=SEED, desplazamiento=0.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(desplazamiento, 1, n)
    return pd.DataFrame(
        {
            "x": x,
            "z": rng.normal(0, 1, n),
            "cat": rng.choice(["a", "b", "c"], n),
            "y": (x + rng.normal(0, 1, n) > desplazamiento).astype(int),
        }
    )


DETECTADO = ("falla", "advertencia")  # DeepChecks marca algunos hallazgos solo como advertencia


def _integridad(df):
    resumen = Resumen()
    validar_integridad(df, "t", None, resumen, label="y", cat_features=["cat"])
    return resumen


# ---------- 1. DeepChecks detecta problemas y pasa con datos limpios ----------
def test_datos_limpios_pasan_los_chequeos_de_integridad():
    resumen = _integridad(_tabla())
    for clase in ("PercentOfNulls", "MixedNulls", "MixedDataTypes", "DataDuplicates"):
        assert resumen.estado("t", clase) == "pasa", clase
    assert resumen.fallas_criticas() == []


def test_detecta_valores_faltantes():
    df = _tabla()
    df.loc[::10, "x"] = np.nan
    assert _integridad(df).estado("t", "PercentOfNulls") == "falla"


def test_detecta_nulos_mezclados():
    df = _tabla()
    df["cat"] = df["cat"].astype(object)
    df.loc[::20, "cat"] = None
    df.loc[1::20, "cat"] = "null"
    df.loc[2::20, "cat"] = "N/A"
    assert _integridad(df).estado("t", "MixedNulls") == "falla"


def test_detecta_tipos_mezclados():
    df = _tabla()
    df["cat"] = df["cat"].astype(object)
    df.loc[::20, "cat"] = 7  # 5 % de números en una columna de texto
    resumen = _integridad(df)
    assert resumen.estado("t", "MixedDataTypes") in DETECTADO
    assert resumen.fallas_criticas()  # tipos mezclados detienen la validación


def test_detecta_duplicados():
    df = _tabla()
    df = pd.concat([df, df.iloc[:200]], ignore_index=True)  # ~17 % duplicadas
    assert _integridad(df).estado("t", "DataDuplicates") in DETECTADO


def test_detecta_drift_de_features_y_etiqueta():
    sin_drift, con_drift = Resumen(), Resumen()
    kwargs = {"label": "y", "cat_features": ["cat"]}
    validar_train_test(_tabla(seed=1), _tabla(seed=2), "t", None, sin_drift, **kwargs)
    validar_train_test(
        _tabla(seed=1), _tabla(seed=2, desplazamiento=3), "t", None, con_drift, **kwargs
    )
    assert sin_drift.estado("t", "FeatureDrift") == "pasa"
    assert con_drift.estado("t", "FeatureDrift") == "falla"
    assert con_drift.estado("t", "MultivariateDrift") == "falla"


def test_detecta_fuga_temporal_entre_train_y_test():
    def con_fechas(df, inicio):
        return df.assign(fecha=pd.date_range(inicio, periods=len(df), freq="h"))

    kwargs = {"label": "y", "cat_features": ["cat"], "datetime_name": "fecha"}
    limpio, fuga = Resumen(), Resumen()
    train = con_fechas(_tabla(seed=1), "2025-01-01")
    validar_train_test(
        train, con_fechas(_tabla(seed=2), "2025-03-01"), "t", None, limpio, **kwargs
    )
    validar_train_test(train, con_fechas(_tabla(seed=2), "2025-01-10"), "t", None, fuga, **kwargs)
    assert limpio.estado("t", "DateTrainTestLeakageOverlap") == "pasa"
    assert fuga.estado("t", "DateTrainTestLeakageOverlap") == "falla"
    assert fuga.fallas_criticas()


# ---------- 2. Chequeos de dominio ----------
def _catalogo(ids, capacidad=20):
    return pd.DataFrame(
        {
            "id": list(ids),
            "name": [f"E{i}" for i in ids],
            "latitude": 20.67,
            "longitude": -103.35,
            "dpcapacity": capacidad,
        }
    )


def test_chequeos_dominio_viajes_cuentan_problemas_inyectados():
    viajes = generar_viajes(dias=10).assign(mes_archivo="2025_01")
    duplicados = viajes.iloc[:5].assign(viaje_id=viajes["viaje_id"].max() + 1 + np.arange(5))
    raros = viajes.iloc[5:8].copy()
    raros["viaje_id"] += 100_000
    raros["fin_viaje"] = raros["inicio_viaje"] + pd.to_timedelta([-10, 30, 90_000], unit="s")
    raros.loc[raros.index[0], "origen_id"] = 999  # estación fuera del catálogo
    viajes = pd.concat([viajes, duplicados, raros], ignore_index=True)

    dominio = {c.chequeo: c for c in chequeos_dominio_viajes(viajes, _catalogo(range(1, 7)))}
    assert dominio["viajes duplicados (mismos datos, distinto id)"].valor == 5
    assert not dominio["viajes duplicados (mismos datos, distinto id)"].pasa
    assert dominio["duración negativa"].valor == 1 and not dominio["duración negativa"].pasa
    assert dominio["duración < 60 s"].valor == 2  # la negativa y la de 30 s
    assert dominio["duración > 24 h"].valor == 1
    assert dominio["estaciones fuera del catálogo"].valor == 1
    assert dominio["viaje_id duplicado"].pasa and dominio["valores nulos"].pasa


def test_chequeos_dominio_estaciones():
    estaciones = _catalogo(range(1, 6))
    estaciones.loc[0, "dpcapacity"] = 0
    estaciones.loc[1, "latitude"] = 0.0  # coordenada inválida
    dominio = {c.chequeo: c for c in chequeos_dominio_estaciones(estaciones)}
    assert dominio["capacidad <= 0"].valor == 1
    assert dominio["coordenadas fuera de Guadalajara"].valor == 1
    assert not dominio["coordenadas fuera de Guadalajara"].pasa
    assert dominio["id duplicado"].pasa


# ---------- 3. Validación completa de punta a punta ----------
def test_validacion_completa_genera_reportes_sin_fallas_criticas(tmp_path):
    viajes = generar_viajes(n_estaciones=20, dias=365, viajes_por_dia=300)
    viajes["mes_archivo"] = viajes["inicio_viaje"].dt.strftime("%Y_%m")
    resumen = validar(
        viajes, _catalogo(range(1, 21)), tmp_path, p2_estaciones=8, p2_filas=10_000,
        viajes_filas=20_000,
    )  # fmt: skip

    conjuntos = {c.conjunto for c in resumen.chequeos}
    assert conjuntos == {"viajes", "estaciones", "p1", "p2", "p3"}
    assert resumen.fallas_criticas() == []
    assert len(list(tmp_path.glob("*.html"))) == 8
    datos = json.loads((tmp_path / "resumen.json").read_text(encoding="utf-8"))
    assert datos["fallas_criticas"] == [] and len(datos["chequeos"]) == len(resumen.chequeos)
    assert "Fallas críticas: 0" in (tmp_path / "resumen.md").read_text(encoding="utf-8")


# ---------- 4. Datos reales ----------
@pytest.mark.skipif(
    not (VIAJES_PARQUET.exists() and ESTACIONES_PARQUET.exists()),
    reason="Sin datos reales: correr antes python -m mibici.dataset",
)
def test_datos_reales_pasan_chequeos_criticos_de_dominio():
    viajes = pd.read_parquet(VIAJES_PARQUET)
    estaciones = pd.read_parquet(ESTACIONES_PARQUET)
    dominio = {c.chequeo: c for c in chequeos_dominio_viajes(viajes, estaciones)}
    dominio.update({f"est_{c.chequeo}": c for c in chequeos_dominio_estaciones(estaciones)})
    for chequeo in (
        "valores nulos", "viaje_id duplicado", "duración negativa",
        "estaciones fuera del catálogo", "est_id duplicado", "est_valores nulos",
        "est_coordenadas fuera de Guadalajara",
    ):  # fmt: skip
        assert dominio[chequeo].pasa, f"{chequeo}: {dominio[chequeo].valor}"
