"""Pruebas de la ingesta (``mibici.dataset``): normalización, lectura, hashes y descarga."""

import hashlib

import pandas as pd
import pytest

from mibici import dataset
from mibici.dataset import (
    COLUMNAS_PERSONALES,
    _normalizar_nombre,
    _sha256,
    descargar,
    leer_csv,
    normalizar_viajes,
)


# ---------- _normalizar_nombre ----------
@pytest.mark.parametrize(
    "original, esperado",
    [
        ("Inicio_del_viaje", "inicio_del_viaje"),
        ("Año de nacimiento", "ano_de_nacimiento"),  # acentos y espacios
        ("  Origen_Id  ", "origen_id"),  # espacios al borde
        ("Género", "genero"),
        ("A?o_de_nacimiento", "a_o_de_nacimiento"),  # "ñ" perdida al exportar el CSV
        ("fin-del.viaje", "fin_del_viaje"),  # puntuación -> guion bajo
    ],
)
def test_normalizar_nombre(original, esperado):
    assert _normalizar_nombre(original) == esperado


def test_normalizar_nombre_es_idempotente():
    nombre = _normalizar_nombre("Año de Nacimiento")
    assert _normalizar_nombre(nombre) == nombre


# ---------- normalizar_viajes ----------
@pytest.fixture
def crudo():
    return pd.DataFrame(
        {
            "Viaje_Id": [1, 2, 3],
            "Usuario_Id": [10, 11, 12],
            "Genero": ["M", "F", "M"],
            "Año_de_nacimiento": [1990, 1985, 2000],
            "Inicio_del_viaje": ["2025-01-01 08:00:00", "2025-01-01 09:30:00", "no es fecha"],
            "Fin_del_viaje": ["2025-01-01 08:20:00", "2025-01-01 10:00:00", "2025-01-01 11:00:00"],
            "Origen_Id": [5, 6, 7],
            "Destino_Id": [6, 7, 5],
        }
    )


def test_normalizar_viajes_esquema_y_tipos(crudo):
    df = normalizar_viajes(crudo, origen="2025_01")
    assert set(df.columns) == {
        "viaje_id", "inicio_viaje", "fin_viaje", "origen_id", "destino_id", "mes_archivo",
    }  # fmt: skip
    assert pd.api.types.is_datetime64_any_dtype(df["inicio_viaje"])
    assert pd.api.types.is_datetime64_any_dtype(df["fin_viaje"])
    assert (df["mes_archivo"] == "2025_01").all()
    assert len(df) == len(crudo)  # la normalización no descarta filas


def test_normalizar_viajes_elimina_datos_personales(crudo):
    df = normalizar_viajes(crudo, origen="2025_01")
    assert not set(COLUMNAS_PERSONALES) & set(df.columns)


def test_normalizar_viajes_fecha_invalida_queda_como_nat(crudo):
    df = normalizar_viajes(crudo, origen="2025_01")
    assert df["inicio_viaje"].isna().sum() == 1
    assert pd.isna(df.loc[2, "inicio_viaje"])


def test_normalizar_viajes_sin_columnas_personales(crudo):
    """Caso límite: un archivo que ya no trae las columnas personales no debe fallar."""
    sin_personales = crudo.drop(columns=["Usuario_Id", "Genero", "Año_de_nacimiento"])
    df = normalizar_viajes(sin_personales, origen="2025_02")
    assert "viaje_id" in df.columns and "usuario_id" not in df.columns


def test_normalizar_viajes_dataframe_vacio():
    vacio = pd.DataFrame(columns=["Inicio_del_viaje", "Fin_del_viaje", "Origen_Id", "Destino_Id"])
    df = normalizar_viajes(vacio, origen="2025_03")
    assert df.empty
    assert {"inicio_viaje", "fin_viaje", "mes_archivo"} <= set(df.columns)


# ---------- leer_csv ----------
def test_leer_csv_utf8(tmp_path):
    ruta = tmp_path / "utf8.csv"
    ruta.write_text("Año,estación\n2025,Glorieta Niños Héroes\n", encoding="utf-8")
    df = leer_csv(ruta)
    assert df.loc[0, "estación"] == "Glorieta Niños Héroes"


def test_leer_csv_latin1(tmp_path):
    """Los CSV de MiBici a veces vienen en latin-1: debe leerlos sin error."""
    ruta = tmp_path / "latin1.csv"
    ruta.write_bytes("Año,estación\n2025,Glorieta Niños Héroes\n".encode("latin-1"))
    df = leer_csv(ruta)
    assert list(df.columns) == ["Año", "estación"]
    assert df.loc[0, "estación"] == "Glorieta Niños Héroes"


# ---------- _sha256 ----------
def test_sha256_coincide_con_hashlib(tmp_path):
    ruta = tmp_path / "archivo.bin"
    contenido = b"mibici" * 500_000  # > 1 MiB: obliga a leer en varios bloques
    ruta.write_bytes(contenido)
    assert _sha256(ruta) == hashlib.sha256(contenido).hexdigest()


def test_sha256_archivo_vacio(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.write_bytes(b"")
    assert _sha256(ruta) == hashlib.sha256(b"").hexdigest()


# ---------- descargar ----------
class _RespuestaFalsa:
    def __init__(self, contenido: bytes, status: int = 200):
        self.content = contenido
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise dataset.requests.HTTPError(f"HTTP {self.status}")


def test_descargar_es_idempotente(tmp_path, monkeypatch):
    """Si el archivo ya existe en data/raw no se vuelve a pedir a la red."""
    destino = tmp_path / "viajes_2025_01.csv"
    destino.write_text("ya estaba")

    def no_llamar(*args, **kwargs):
        raise AssertionError("No debía descargar un archivo existente")

    monkeypatch.setattr(dataset.requests, "get", no_llamar)
    assert descargar("https://ejemplo/x.csv", destino) == destino
    assert destino.read_text() == "ya estaba"


def test_descargar_guarda_contenido(tmp_path, monkeypatch):
    destino = tmp_path / "nuevo.csv"
    llamadas = []

    def get_falso(url, headers, timeout):
        llamadas.append((url, timeout))
        return _RespuestaFalsa(b"a,b\n1,2\n")

    monkeypatch.setattr(dataset.requests, "get", get_falso)
    descargar("https://ejemplo/nuevo.csv", destino)
    assert destino.read_bytes() == b"a,b\n1,2\n"
    assert llamadas == [("https://ejemplo/nuevo.csv", 120)]  # siempre con timeout


def test_descargar_error_http_no_deja_archivo(tmp_path, monkeypatch):
    destino = tmp_path / "falla.csv"
    monkeypatch.setattr(dataset.requests, "get", lambda *a, **k: _RespuestaFalsa(b"", 404))
    with pytest.raises(dataset.requests.HTTPError):
        descargar("https://ejemplo/falla.csv", destino)
    assert not destino.exists()
