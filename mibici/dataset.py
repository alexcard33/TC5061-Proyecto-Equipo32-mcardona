"""Ingesta reproducible de datos de MiBici — Equipo 32.

Descarga los meses definidos en config.py, normaliza columnas y
codificación, guarda Parquet y versiona los datos con un manifiesto
de hashes. Ejecutable sin pasos manuales:

    python -m mibici.dataset                    # referencia: ene-dic 2025 (entrenamiento)
    python -m mibici.dataset --periodo actual   # actual: ene-oct 2026 (monitoreo de drift)

Cada periodo escribe sus propios Parquet y su propio manifiesto, así que ingerir el
periodo actual no modifica los datos con los que se entrenaron los modelos registrados.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata

from loguru import logger
import pandas as pd
import requests
import typer

from mibici.config import (
    ESTACIONES_ACTUAL_PARQUET,
    ESTACIONES_PARQUET,
    MANIFEST_ACTUAL_PATH,
    MANIFEST_PATH,
    NOMENCLATURA_ACTUAL_URL,
    NOMENCLATURA_URL,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    VIAJES_ACTUAL_PARQUET,
    VIAJES_ACTUAL_URLS,
    VIAJES_PARQUET,
    VIAJES_URLS,
)

app = typer.Typer()

HEADERS = {"User-Agent": "Mozilla/5.0 (proyecto academico TC5061)"}

# Mapeo de nombres de columna: los archivos de MiBici cambian
# nombres y acentos entre años; todo se normaliza a este esquema.
COLUMNAS_VIAJES = {
    "viaje_id": "viaje_id",
    "usuario_id": "usuario_id",
    "genero": "genero",
    "ano_de_nacimiento": "anio_nacimiento",
    "anio_de_nacimiento": "anio_nacimiento",
    "a_o_de_nacimiento": "anio_nacimiento",
    "inicio_del_viaje": "inicio_viaje",
    "fin_del_viaje": "fin_viaje",
    "origen_id": "origen_id",
    "destino_id": "destino_id",
}


# Columnas con datos personales que se eliminan en la ingesta.
COLUMNAS_PERSONALES = ["usuario_id", "genero", "anio_nacimiento"]

# Periodos de ingesta. "referencia" es el de entrenamiento (sin cambios respecto a las
# entregas anteriores); "actual" es el periodo posterior que se monitorea.
PERIODOS = {
    "referencia": {
        "viajes_urls": VIAJES_URLS,
        "nomenclatura_url": NOMENCLATURA_URL,
        "nomenclatura_raw": "nomenclatura.csv",
        "viajes_parquet": VIAJES_PARQUET,
        "estaciones_parquet": ESTACIONES_PARQUET,
        "manifest": MANIFEST_PATH,
    },
    "actual": {
        "viajes_urls": VIAJES_ACTUAL_URLS,
        "nomenclatura_url": NOMENCLATURA_ACTUAL_URL,
        "nomenclatura_raw": "nomenclatura_actual.csv",
        "viajes_parquet": VIAJES_ACTUAL_PARQUET,
        "estaciones_parquet": ESTACIONES_ACTUAL_PARQUET,
        "manifest": MANIFEST_ACTUAL_PATH,
    },
}


def _normalizar_nombre(col: str) -> str:
    """Quita acentos, pasa a minúsculas y deja solo [a-z0-9_]."""
    col = unicodedata.normalize("NFKD", col).encode("ascii", "ignore").decode()
    col = col.strip().lower().replace(" ", "_")
    return "".join(c if c.isalnum() or c == "_" else "_" for c in col)


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(url: str, destino: Path) -> Path:
    """Descarga un archivo si no existe ya en data/raw (idempotente)."""
    if destino.exists():
        logger.info(f"Ya existe, no se descarga: {destino.name}")
        return destino
    logger.info(f"Descargando {url}")
    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    destino.write_bytes(r.content)
    return destino


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV tolerando las codificaciones usadas por MiBici."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(ruta, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"No se pudo decodificar {ruta}")


def normalizar_viajes(df: pd.DataFrame, origen: str) -> pd.DataFrame:
    """Normaliza nombres de columna, tipos de fecha y agrega el mes origen."""
    df.columns = [_normalizar_nombre(c) for c in df.columns]
    df = df.rename(columns={c: COLUMNAS_VIAJES[c] for c in df.columns if c in COLUMNAS_VIAJES})
    df["inicio_viaje"] = pd.to_datetime(df["inicio_viaje"], errors="coerce")
    df["fin_viaje"] = pd.to_datetime(df["fin_viaje"], errors="coerce")
    df["mes_archivo"] = origen
    # Minimización de datos personales (aviso de privacidad de MiBici/AMIM): el
    # pseudónimo de usuario, el género y el año de nacimiento no se usan en P1-P3,
    # así que no se guardan en los Parquet procesados.
    return df.drop(columns=[c for c in COLUMNAS_PERSONALES if c in df.columns])


def ingestar(periodo: str = "referencia") -> dict:
    """Descarga, normaliza y guarda un periodo de datos; devuelve su manifiesto."""
    cfg = PERIODOS[periodo]
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifiesto = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "periodo": periodo,
        "archivos": {},
    }

    # ---- 1. Viajes ----
    partes = []
    for mes, url in cfg["viajes_urls"].items():
        ruta = descargar(url, RAW_DATA_DIR / f"viajes_{mes}.csv")
        df = normalizar_viajes(leer_csv(ruta), origen=mes)
        manifiesto["archivos"][ruta.name] = {
            "url": url,
            "sha256": _sha256(ruta),
            "registros": len(df),
        }
        partes.append(df)
    viajes = pd.concat(partes, ignore_index=True)

    # ---- 2. Nomenclatura de estaciones ----
    nomenclatura_url = cfg["nomenclatura_url"]
    ruta_nom = descargar(nomenclatura_url, RAW_DATA_DIR / cfg["nomenclatura_raw"])
    estaciones = leer_csv(ruta_nom)
    estaciones.columns = [_normalizar_nombre(c) for c in estaciones.columns]
    manifiesto["archivos"][ruta_nom.name] = {
        "url": nomenclatura_url,
        "sha256": _sha256(ruta_nom),
        "registros": len(estaciones),
    }

    # ---- 3. Guardar en formato eficiente ----
    viajes.to_parquet(cfg["viajes_parquet"], index=False)
    estaciones.to_parquet(cfg["estaciones_parquet"], index=False)

    # ---- 4. Versión de los datos ----
    hashes = sorted(v["sha256"] for v in manifiesto["archivos"].values())
    manifiesto["data_version"] = hashlib.sha256("".join(hashes).encode()).hexdigest()[:12]
    manifiesto["total_viajes"] = len(viajes)
    cfg["manifest"].write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False))

    logger.success(
        f"Ingesta '{periodo}' completa: {len(viajes):,} viajes, {len(estaciones)} estaciones, "
        f"data_version={manifiesto['data_version']}"
    )
    return manifiesto


def leer_viajes(columnas: list[str] | None = None, incluir_actual: bool = True) -> pd.DataFrame:
    """Lee los viajes procesados de referencia (2025) y, si se pide, los del periodo actual.

    Los rezagos y medias móviles de P1 necesitan historia continua: los primeros días de
    2026 usan viajes de diciembre de 2025. Por eso el monitoreo de drift y el A/B testing
    leen ambos Parquet como una sola tabla.
    """
    partes = [pd.read_parquet(VIAJES_PARQUET, columns=columnas)]
    if incluir_actual:
        if not VIAJES_ACTUAL_PARQUET.exists():
            raise FileNotFoundError(
                f"No existe {VIAJES_ACTUAL_PARQUET.name}. "
                "Ejecuta primero: python -m mibici.dataset --periodo actual"
            )
        partes.append(pd.read_parquet(VIAJES_ACTUAL_PARQUET, columns=columnas))
    return pd.concat(partes, ignore_index=True)


@app.command()
def main(
    periodo: str = typer.Option(
        "referencia",
        help="referencia: ene-dic 2025 (entrenamiento) | actual: ene-oct 2026 (monitoreo).",
    ),
):
    if periodo not in PERIODOS:
        raise typer.BadParameter(f"Periodo desconocido: {periodo}. Opciones: {sorted(PERIODOS)}")
    ingestar(periodo)


if __name__ == "__main__":
    app()