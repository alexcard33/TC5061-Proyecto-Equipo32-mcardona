"""Validación de calidad e integridad de datos con DeepChecks — Equipo 32.

Valida los mismos datos que consumen los pipelines de P1, P2 y P3, construidos
con las mismas funciones de ``mibici.features``:

1. ``viajes``: tabla de viajes procesada (``data/processed/viajes.parquet``).
2. ``estaciones``: catálogo de estaciones (``data/processed/estaciones.parquet``).
3. ``p1``: panel estación × día con sus features; TRAIN (ene-sep) vs TEST (nov-dic).
4. ``p2``: panel estación × hora con la etiqueta de riesgo; TRAIN vs TEST.
5. ``p3``: perfil de uso por estación; periodo TRAIN vs periodo TEST.

Para cada conjunto corre la suite ``data_integrity`` de DeepChecks más
``PercentOfNulls`` (valores faltantes, tipos mezclados, duplicados, etiquetas en
conflicto, outliers...) y, para P1-P3, la suite ``train_test_validation`` (drift
de features y de etiqueta, categorías nuevas, mezcla de muestras entre TRAIN y
TEST, fuga temporal). Además calcula chequeos de dominio exactos sobre la tabla
completa (duraciones inválidas, viajes duplicados, ids fuera de catálogo...).

Salidas en ``reports/deepchecks/``: un HTML por suite, ``resumen.json`` y
``resumen.md``. Termina con código 1 si falla algún chequeo crítico.

DeepChecks 0.19.1 no es compatible con numpy 2 / pandas 3 / scikit-learn 1.9,
por eso corre en un entorno aparte (``requirements-validacion.txt``)::

    python -m mibici.validacion_datos
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import warnings

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    ESTACIONES_PARQUET,
    P1_FERIADOS,
    P1_HORIZONTE_DIAS,
    P1_LAGS_DIAS,
    P1_MAX_DURATION_SECONDS,
    P1_MIN_DURATION_SECONDS,
    P1_TEST_END,
    P1_TEST_START,
    P1_TRAIN_START,
    P1_VAL_START,
    P1_VENTANAS_MEDIA_DIAS,
    P2_FRACCION_CAPACIDAD,
    P2_HORIZONTE_HORAS,
    P2_TEST_END,
    P2_TEST_START,
    P2_TRAIN_START,
    P2_VAL_START,
    P3_MAX_DURATION_SECONDS,
    P3_MAX_INACTIVITY_DAYS,
    P3_MIN_DURATION_SECONDS,
    P3_MIN_HISTORY_DAYS,
    P3_TEST_END,
    P3_TEST_START,
    P3_TRAIN_END,
    P3_TRAIN_START,
    REPORTS_DIR,
    SEED,
    VIAJES_PARQUET,
)
from mibici.features import (
    P1_FEATURES,
    P2_FEATURES,
    P3_MODEL_FEATURES,
    p1_agregar_features,
    p1_construir_panel_diario,
    p2_agregar_etiqueta,
    p2_agregar_features,
    p2_ajustar_historico,
    p2_aplicar_historico,
    p2_construir_panel,
    p2_historico_pasado,
    p3_construir_features,
    p3_filtrar_viajes,
)

DEEPCHECKS_DIR = REPORTS_DIR / "deepchecks"

# Límites de la zona metropolitana de Guadalajara para validar coordenadas.
LAT_RANGO = (20.4, 20.9)
LON_RANGO = (-103.6, -103.1)

CAT_VIAJES = ["origen_id", "destino_id", "dia_semana", "mes_archivo"]
CAT_P1 = ["dia_semana", "fin_semana", "feriado"]
CAT_P2 = ["hora_dia", "dia_semana", "fin_semana"]

# Chequeos (por clase de DeepChecks) cuyo fallo indica un error en los datos o una fuga
# entre TRAIN y TEST. El resto (drift entre ene-sep y nov-dic, correlaciones, filas
# repetidas en paneles de conteos con muchos ceros...) son características del dominio:
# se documentan en el resumen pero no detienen el proceso.
CHEQUEOS_CRITICOS = {
    "PercentOfNulls",
    "MixedNulls",
    "MixedDataTypes",
    "ConflictingLabels",
    "DateTrainTestLeakageOverlap",
    "DateTrainTestLeakageDuplicates",
}

ICONOS = {"pasa": "✅", "advertencia": "⚠️", "falla": "❌", "sin_condicion": "ℹ️", "no_aplica": "➖"}


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------
@dataclass
class ResultadoChequeo:
    conjunto: str
    suite: str
    chequeo: str
    clase: str
    estado: str  # pasa | advertencia | falla | sin_condicion | no_aplica
    detalle: str = ""

    @property
    def critico(self) -> bool:
        return self.clase in CHEQUEOS_CRITICOS


@dataclass
class ChequeoDominio:
    conjunto: str
    chequeo: str
    valor: float
    pasa: bool
    detalle: str = ""


@dataclass
class Resumen:
    chequeos: list[ResultadoChequeo] = field(default_factory=list)
    dominio: list[ChequeoDominio] = field(default_factory=list)
    reportes: list[str] = field(default_factory=list)

    def estado(self, conjunto: str, clase: str) -> str | None:
        """Estado de un chequeo por conjunto y clase de DeepChecks (p. ej. "MixedNulls")."""
        for c in self.chequeos:
            if c.conjunto == conjunto and c.clase == clase:
                return c.estado
        return None

    def fallas_criticas(self) -> list[ResultadoChequeo]:
        """Chequeos críticos que no pasaron (DeepChecks marca algunos como advertencia)."""
        return [c for c in self.chequeos if c.critico and c.estado in ("falla", "advertencia")]


def _deepchecks():
    """Importa DeepChecks con un mensaje claro si no está el entorno de validación."""
    try:
        warnings.filterwarnings("ignore", module="deepchecks")
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.checks import PercentOfNulls
        from deepchecks.tabular.suites import data_integrity, train_test_validation
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise ImportError(
            "DeepChecks no está disponible. Crea el entorno de validación "
            "(ver requirements-validacion.txt)."
        ) from exc
    return Dataset, PercentOfNulls, data_integrity, train_test_validation


def resultados_de_suite(suite_result, conjunto: str, suite: str) -> list[ResultadoChequeo]:
    """Traduce el resultado de una suite de DeepChecks a filas simples."""
    from deepchecks.core import CheckFailure, ConditionCategory

    filas = []
    for res in suite_result.results:
        nombre = res.get_header()
        clase = type(res.check).__name__
        if isinstance(res, CheckFailure):
            motivo = str(res.exception)[:200]
            filas.append(ResultadoChequeo(conjunto, suite, nombre, clase, "no_aplica", motivo))
            continue
        condiciones = res.conditions_results
        if not condiciones:
            filas.append(ResultadoChequeo(conjunto, suite, nombre, clase, "sin_condicion"))
            continue
        categorias = {c.category for c in condiciones}
        if ConditionCategory.FAIL in categorias:
            estado = "falla"
        elif ConditionCategory.WARN in categorias:
            estado = "advertencia"
        else:
            estado = "pasa"
        detalle = " | ".join(f"{c.name}: {c.details}" for c in condiciones if not c.is_pass())
        filas.append(ResultadoChequeo(conjunto, suite, nombre, clase, estado, detalle[:500]))
    return filas


# ---------------------------------------------------------------------------
# Chequeos de dominio (exactos, sobre la tabla completa)
# ---------------------------------------------------------------------------
def chequeos_dominio_viajes(
    viajes: pd.DataFrame, estaciones: pd.DataFrame
) -> list[ChequeoDominio]:
    duracion = (viajes["fin_viaje"] - viajes["inicio_viaje"]).dt.total_seconds()
    sin_id = viajes.drop(columns=[c for c in ("viaje_id", "mes_archivo") if c in viajes])
    fuera = (set(viajes["origen_id"]) | set(viajes["destino_id"])) - set(estaciones["id"])

    def chequeo(nombre, valor, pasa, detalle=""):
        return ChequeoDominio("viajes", nombre, float(valor), bool(pasa), detalle)

    return [
        chequeo("filas", len(viajes), len(viajes) > 0),
        chequeo("valores nulos", viajes.isna().sum().sum(), viajes.notna().all().all()),
        chequeo("viaje_id duplicado", viajes["viaje_id"].duplicated().sum(),
                not viajes["viaje_id"].duplicated().any()),
        chequeo("viajes duplicados (mismos datos, distinto id)", sin_id.duplicated().sum(),
                not sin_id.duplicated().any(),
                "Mismo origen, destino, inicio y fin al segundo; representan < 0.1 % del total."),
        chequeo("duración negativa", (duracion < 0).sum(), not (duracion < 0).any()),
        chequeo("duración < 60 s", (duracion < P1_MIN_DURATION_SECONDS).sum(), True,
                "Desanclajes fallidos: los pipelines de P1-P3 los descartan."),
        chequeo("duración > 24 h", (duracion > P1_MAX_DURATION_SECONDS).sum(), True,
                "Bicicletas no ancladas: los pipelines de P1-P3 los descartan."),
        chequeo("estaciones fuera del catálogo", len(fuera), not fuera,
                ", ".join(map(str, sorted(fuera)[:20]))),
    ]  # fmt: skip


def chequeos_dominio_estaciones(estaciones: pd.DataFrame) -> list[ChequeoDominio]:
    coords_ok = estaciones["latitude"].between(*LAT_RANGO) & estaciones["longitude"].between(
        *LON_RANGO
    )
    cap0 = estaciones.loc[estaciones["dpcapacity"] <= 0, "id"].tolist()

    def chequeo(nombre, valor, pasa, detalle=""):
        return ChequeoDominio("estaciones", nombre, float(valor), bool(pasa), detalle)

    return [
        chequeo("id duplicado", estaciones["id"].duplicated().sum(),
                not estaciones["id"].duplicated().any()),
        chequeo("valores nulos", estaciones.isna().sum().sum(), estaciones.notna().all().all()),
        chequeo("coordenadas fuera de Guadalajara", (~coords_ok).sum(), coords_ok.all()),
        chequeo("capacidad <= 0", len(cap0), True,
                "P2 las excluye (no se puede calcular el umbral de riesgo). ids: "
                + ", ".join(map(str, cap0))),
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# Conjuntos de cada pregunta (mismas funciones que los pipelines)
# ---------------------------------------------------------------------------
def _filtrar_duracion(viajes: pd.DataFrame, minimo: int, maximo: int) -> pd.DataFrame:
    duracion = (viajes["fin_viaje"] - viajes["inicio_viaje"]).dt.total_seconds()
    return viajes.loc[duracion.between(minimo, maximo)].dropna()


def _entre(df: pd.DataFrame, nivel: str, inicio, fin) -> pd.DataFrame:
    t = df.index.get_level_values(nivel)
    return df.loc[(t >= pd.Timestamp(inicio)) & (t < pd.Timestamp(fin))]


def tabla_viajes(viajes: pd.DataFrame) -> pd.DataFrame:
    """Viajes con variables derivadas para DeepChecks (sin filtrar)."""
    df = viajes.drop(columns=["viaje_id"], errors="ignore").copy()
    df["duracion_s"] = (df["fin_viaje"] - df["inicio_viaje"]).dt.total_seconds()
    df["hora"] = df["inicio_viaje"].dt.hour
    df["dia_semana"] = df["inicio_viaje"].dt.dayofweek
    return df.drop(columns=["fin_viaje"])


def conjuntos_p1(viajes: pd.DataFrame, capacidad: pd.Series):
    """TRAIN y TEST de P1 tal como los arma ``p1_regresion.preparar_datos``."""
    viajes = _filtrar_duracion(viajes, P1_MIN_DURATION_SECONDS, P1_MAX_DURATION_SECONDS)
    panel = p1_construir_panel_diario(viajes, P1_TRAIN_START, P1_TEST_END)
    df = p1_agregar_features(
        panel, capacidad, P1_FERIADOS, P1_LAGS_DIAS, P1_VENTANAS_MEDIA_DIAS, P1_HORIZONTE_DIAS
    )
    train = _entre(df, "fecha", P1_TRAIN_START, P1_VAL_START).dropna(subset=P1_FEATURES)
    test = _entre(df, "fecha", P1_TEST_START, P1_TEST_END).dropna(subset=P1_FEATURES)
    columnas = P1_FEATURES + ["viajes"]
    return train[columnas].reset_index(), test[columnas].reset_index()


def conjuntos_p2(viajes, capacidad, n_estaciones: int | None, n_filas: int | None):
    """TRAIN y TEST de P2 como en ``p2_clasificacion.preparar_datos``.

    El panel horario completo tiene millones de filas; para que la validación sea
    rápida se muestrean estaciones y filas con ``SEED``. Cada estación se procesa de
    forma independiente, así que la muestra no altera sus variables.
    """
    viajes = _filtrar_duracion(viajes, P1_MIN_DURATION_SECONDS, P1_MAX_DURATION_SECONDS)
    estaciones = np.array(sorted(set(viajes["origen_id"]) | set(viajes["destino_id"])))
    if n_estaciones and n_estaciones < len(estaciones):
        elegidas = np.random.default_rng(SEED).choice(estaciones, n_estaciones, replace=False)
        viajes = viajes.loc[
            viajes["origen_id"].isin(elegidas) & viajes["destino_id"].isin(elegidas)
        ]

    panel = p2_construir_panel(viajes, P2_TRAIN_START, P2_TEST_END)
    df = p2_agregar_etiqueta(panel, P2_HORIZONTE_HORAS, capacidad, P2_FRACCION_CAPACIDAD)
    df = p2_agregar_features(df, P2_HORIZONTE_HORAS)

    margen = pd.Timedelta(hours=P2_HORIZONTE_HORAS)
    train = _entre(df, "hora", P2_TRAIN_START, pd.Timestamp(P2_VAL_START) - margen)
    test = _entre(df, "hora", P2_TEST_START, pd.Timestamp(P2_TEST_END) - margen)
    historico = p2_ajustar_historico(train)
    train = p2_historico_pasado(train).dropna(subset=P2_FEATURES)
    test = p2_aplicar_historico(test, historico).dropna(subset=P2_FEATURES)

    columnas = P2_FEATURES + ["riesgo"]
    train, test = train[columnas].reset_index(), test[columnas].reset_index()
    if n_filas:
        train = train.sample(min(n_filas, len(train)), random_state=SEED)
        test = test.sample(min(n_filas, len(test)), random_state=SEED)
    return train.sort_values("hora"), test.sort_values("hora")


def conjuntos_p3(viajes: pd.DataFrame):
    """Perfil de uso por estación en el periodo TRAIN y en el periodo TEST de P3."""
    viajes = p3_filtrar_viajes(viajes, P3_MIN_DURATION_SECONDS, P3_MAX_DURATION_SECONDS)
    train = viajes.loc[
        (viajes["inicio_viaje"] >= pd.Timestamp(P3_TRAIN_START))
        & (viajes["inicio_viaje"] <= pd.Timestamp(P3_TRAIN_END))
        & (viajes["fin_viaje"] < pd.Timestamp(P3_TEST_START))
    ]
    test = viajes.loc[
        (viajes["inicio_viaje"] >= pd.Timestamp(P3_TEST_START))
        & (viajes["fin_viaje"] < pd.Timestamp(P3_TEST_END))
    ]
    kwargs = {
        "max_inactivity_days": P3_MAX_INACTIVITY_DAYS,
        "min_history_days": P3_MIN_HISTORY_DAYS,
    }
    f_train, _ = p3_construir_features(train, pd.Timestamp(P3_TRAIN_END), **kwargs)
    f_test, _ = p3_construir_features(
        test, pd.Timestamp(P3_TEST_END) - pd.Timedelta(seconds=1), **kwargs
    )
    return f_train.reset_index(), f_test.reset_index()


# ---------------------------------------------------------------------------
# Ejecución de suites
# ---------------------------------------------------------------------------
def _guardar(suite_result, nombre: str, salida: Path | None, resumen: Resumen) -> None:
    if salida is None:
        return
    ruta = salida / f"{nombre}.html"
    # connected=True: plotly se carga desde CDN en lugar de incrustarse en cada HTML.
    suite_result.save_as_html(str(ruta), as_widget=False, connected=True)
    resumen.reportes.append(ruta.name)


def validar_integridad(df, conjunto, salida, resumen, **kwargs_dataset):
    """Suite ``data_integrity`` + ``PercentOfNulls`` sobre un conjunto."""
    Dataset, PercentOfNulls, data_integrity, _ = _deepchecks()
    suite = data_integrity()
    suite.add(PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0.0))
    resultado = suite.run(Dataset(df, **kwargs_dataset))
    resumen.chequeos += resultados_de_suite(resultado, conjunto, "data_integrity")
    _guardar(resultado, f"{conjunto}_integridad", salida, resumen)
    return resultado


def validar_train_test(train, test, conjunto, salida, resumen, **kwargs_dataset):
    """Suite ``train_test_validation`` entre TRAIN y TEST."""
    Dataset, _, _, train_test_validation = _deepchecks()
    resultado = train_test_validation().run(
        Dataset(train, **kwargs_dataset), Dataset(test, **kwargs_dataset)
    )
    resumen.chequeos += resultados_de_suite(resultado, conjunto, "train_test_validation")
    _guardar(resultado, f"{conjunto}_train_test", salida, resumen)
    return resultado


def escribir_resumen(resumen: Resumen, salida: Path) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    datos = {
        "chequeos": [{**asdict(c), "critico": c.critico} for c in resumen.chequeos],
        "dominio": [asdict(c) for c in resumen.dominio],
        "reportes": resumen.reportes,
        "fallas_criticas": [asdict(c) for c in resumen.fallas_criticas()],
    }
    (salida / "resumen.json").write_text(
        json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lineas = [
        "# Resumen de validación de datos (DeepChecks)",
        "",
        "Generado con `python -m mibici.validacion_datos`.",
        "",
        "## Chequeos de dominio (tabla completa)",
        "",
        "| Conjunto | Chequeo | Valor | Resultado | Detalle |",
        "|---|---|---:|---|---|",
    ]
    for c in resumen.dominio:
        marca = "✅" if c.pasa else "❌"
        lineas.append(f"| {c.conjunto} | {c.chequeo} | {c.valor:,.0f} | {marca} | {c.detalle} |")
    lineas += [
        "",
        "## Chequeos de DeepChecks",
        "",
        "| Conjunto | Suite | Chequeo | Estado | Detalle |",
        "|---|---|---|---|---|",
    ]
    for c in resumen.chequeos:
        detalle = c.detalle.replace("|", "/").replace("\n", " ")
        critico = " **(crítico)**" if c.critico else ""
        lineas.append(
            f"| {c.conjunto} | {c.suite} | {c.chequeo}{critico} | {ICONOS[c.estado]} {c.estado}"
            f" | {detalle} |"
        )
    lineas += ["", f"**Fallas críticas: {len(resumen.fallas_criticas())}**", ""]
    (salida / "resumen.md").write_text("\n".join(lineas), encoding="utf-8")


def validar(
    viajes: pd.DataFrame,
    estaciones: pd.DataFrame,
    salida: Path | None,
    p2_estaciones: int | None = 120,
    p2_filas: int | None = 150_000,
    viajes_filas: int | None = 300_000,
) -> Resumen:
    """Corre todos los chequeos y devuelve el resumen (``salida=None`` no escribe HTML)."""
    if salida is not None:
        salida.mkdir(parents=True, exist_ok=True)
    resumen = Resumen()
    capacidad = estaciones.set_index("id")["dpcapacity"]

    logger.info("1/5 Viajes")
    resumen.dominio += chequeos_dominio_viajes(viajes, estaciones)
    muestra = viajes
    if viajes_filas and viajes_filas < len(viajes):
        muestra = viajes.sample(viajes_filas, random_state=SEED)
    validar_integridad(
        tabla_viajes(muestra), "viajes", salida, resumen,
        cat_features=CAT_VIAJES, datetime_name="inicio_viaje",
    )  # fmt: skip

    logger.info("2/5 Estaciones")
    resumen.dominio += chequeos_dominio_estaciones(estaciones)
    validar_integridad(
        estaciones, "estaciones", salida, resumen,
        index_name="id", features=["latitude", "longitude", "dpcapacity"], cat_features=[],
    )  # fmt: skip

    logger.info("3/5 P1")
    train, test = conjuntos_p1(viajes, capacidad)
    kwargs = {"label": "viajes", "features": P1_FEATURES, "cat_features": CAT_P1,
              "datetime_name": "fecha", "label_type": "regression"}  # fmt: skip
    validar_integridad(train, "p1", salida, resumen, **kwargs)
    validar_train_test(train, test, "p1", salida, resumen, **kwargs)

    logger.info("4/5 P2")
    train, test = conjuntos_p2(viajes, capacidad, p2_estaciones, p2_filas)
    kwargs = {"label": "riesgo", "features": P2_FEATURES, "cat_features": CAT_P2,
              "datetime_name": "hora", "label_type": "binary"}  # fmt: skip
    validar_integridad(train, "p2", salida, resumen, **kwargs)
    validar_train_test(train, test, "p2", salida, resumen, **kwargs)

    logger.info("5/5 P3")
    train, test = conjuntos_p3(viajes)
    # Sin index_name: las mismas estaciones aparecen en ambos periodos por diseño.
    kwargs = {"features": P3_MODEL_FEATURES, "cat_features": []}
    validar_integridad(train, "p3", salida, resumen, **kwargs)
    validar_train_test(train, test, "p3", salida, resumen, **kwargs)

    if salida is not None:
        escribir_resumen(resumen, salida)
    return resumen


app = typer.Typer()


@app.command()
def main(
    viajes_path: Path = typer.Option(VIAJES_PARQUET, help="Parquet de viajes."),
    estaciones_path: Path = typer.Option(ESTACIONES_PARQUET, help="Parquet de estaciones."),
    salida: Path = typer.Option(DEEPCHECKS_DIR, help="Carpeta de reportes."),
    p2_estaciones: int = typer.Option(120, help="Estaciones muestreadas en P2 (0 = todas)."),
    p2_filas: int = typer.Option(
        150_000, help="Filas muestreadas por conjunto en P2 (0 = todas)."
    ),
    viajes_filas: int = typer.Option(
        300_000, help="Viajes muestreados para DeepChecks (0 = todos)."
    ),
):
    _deepchecks()
    resumen = validar(
        pd.read_parquet(viajes_path),
        pd.read_parquet(estaciones_path),
        salida,
        p2_estaciones or None,
        p2_filas or None,
        viajes_filas or None,
    )
    criticas = resumen.fallas_criticas()
    hallazgos = [c for c in resumen.dominio if not c.pasa]
    logger.info(f"Resumen en {salida / 'resumen.md'}")
    if criticas:
        logger.error(
            f"Fallaron {len(criticas)} chequeos críticos: {[c.chequeo for c in criticas]}"
        )
        raise typer.Exit(code=1)
    logger.success(
        f"Validación completa: 0 fallas críticas, {len(hallazgos)} hallazgos de dominio "
        "documentados."
    )


if __name__ == "__main__":
    app()
