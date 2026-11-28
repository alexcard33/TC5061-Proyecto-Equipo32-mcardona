"""Validación de la prueba de carga corta del pipeline (carga/validar_carga.py)."""

import importlib.util
from pathlib import Path

import pytest

RUTA = Path(__file__).resolve().parents[1] / "carga" / "validar_carga.py"
_spec = importlib.util.spec_from_file_location("validar_carga", RUTA)
validar_carga = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validar_carga)

ENCABEZADO = (
    "Type,Name,Request Count,Failure Count,Median Response Time,Average Response Time,"
    "Min Response Time,Max Response Time,Average Content Size,Requests/s,Failures/s,"
    "50%,66%,75%,80%,90%,95%,98%,99%,99.9%,99.99%,100%"
)


def _csv(tmp_path: Path, peticiones: int, fallas: int, p95: int) -> Path:
    percentiles = f"60,65,70,72,80,{p95},{p95},{p95},{p95},{p95},{p95}"
    datos = f"{peticiones},{fallas},60,65,10,300,120,9.5,0,{percentiles}"
    ruta = tmp_path / "locust_stats.csv"
    ruta.write_text(
        "\n".join([ENCABEZADO, f"POST,/predict,{datos}", f",Aggregated,{datos}"]),
        encoding="utf-8",
    )
    return ruta


@pytest.fixture(autouse=True)
def _umbrales_por_defecto(monkeypatch):
    monkeypatch.delenv("CARGA_MAX_ERRORES_PCT", raising=False)
    monkeypatch.delenv("CARGA_MAX_P95_MS", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)


def test_pasa_dentro_de_umbrales(tmp_path):
    ruta = _csv(tmp_path, peticiones=400, fallas=0, p95=90)

    assert validar_carga.main(["validar_carga.py", str(ruta)]) == 0


def test_falla_si_hay_errores(tmp_path):
    ruta = _csv(tmp_path, peticiones=400, fallas=2, p95=90)

    assert validar_carga.main(["validar_carga.py", str(ruta)]) == 1


def test_falla_si_el_p95_rebasa_el_umbral(tmp_path):
    ruta = _csv(tmp_path, peticiones=400, fallas=0, p95=900)

    assert validar_carga.main(["validar_carga.py", str(ruta)]) == 1


def test_umbrales_se_ajustan_con_variables(tmp_path, monkeypatch):
    ruta = _csv(tmp_path, peticiones=400, fallas=0, p95=900)
    monkeypatch.setenv("CARGA_MAX_P95_MS", "1000")

    assert validar_carga.main(["validar_carga.py", str(ruta)]) == 0


def test_sin_peticiones_se_considera_falla(tmp_path):
    ruta = _csv(tmp_path, peticiones=0, fallas=0, p95=0)

    assert validar_carga.main(["validar_carga.py", str(ruta)]) == 1


def test_csv_sin_fila_aggregated(tmp_path):
    ruta = tmp_path / "locust_stats.csv"
    ruta.write_text(ENCABEZADO + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Aggregated"):
        validar_carga.main(["validar_carga.py", str(ruta)])


def test_escribe_el_resumen_de_github(tmp_path, monkeypatch):
    ruta = _csv(tmp_path, peticiones=400, fallas=0, p95=90)
    resumen = tmp_path / "resumen.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(resumen))

    validar_carga.main(["validar_carga.py", str(ruta)])

    texto = resumen.read_text(encoding="utf-8")
    assert "Prueba de carga (Locust)" in texto
    assert "≤ 500 ms" in texto
