"""Pruebas de la API REST con el modelo empaquetado en models/serving/p1.

Usan el TestClient de FastAPI, así que no hace falta levantar el contenedor: la app carga
el modelo igual que dentro de la imagen (MODEL_SOURCE=local). El mismo payload de ejemplo
lo usa la prueba de humo del pipeline de CI/CD contra el sistema levantado con Compose.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="Requiere fastapi (requirements.txt)")
pytest.importorskip("httpx", reason="El TestClient de FastAPI necesita httpx")
pytest.importorskip("skops", reason="El modelo servido está serializado con skops")

from fastapi.testclient import TestClient  # noqa: E402

from api.main import app  # noqa: E402

PAYLOAD = Path(__file__).parent / "fixtures" / "predict_ejemplo.json"


@pytest.fixture(scope="module")
def cliente():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_SOURCE", "local")
        mp.delenv("MODEL_DIR", raising=False)
        # Entrar al "with" dispara el lifespan de la app, que es donde se carga el modelo.
        with TestClient(app) as c:
            yield c


@pytest.fixture
def payload() -> dict:
    return json.loads(PAYLOAD.read_text(encoding="utf-8"))


def test_health_con_modelo_cargado(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"status": "ok", "modelo_cargado": True}


def test_model_info_expone_la_firma_del_modelo(cliente):
    respuesta = cliente.get("/model-info")
    assert respuesta.status_code == 200
    info = respuesta.json()
    assert info["fuente"] == "local"
    assert len(info["columnas"]) == 12


def test_predict_regresa_una_prediccion_por_instancia(cliente, payload):
    respuesta = cliente.post("/predict", json=payload)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    n = len(payload["instancias"])
    assert cuerpo["n_instancias"] == n
    assert len(cuerpo["predicciones"]) == n
    assert all(p >= 0 for p in cuerpo["predicciones"])


def test_predict_rechaza_campos_desconocidos(cliente, payload):
    payload["instancias"][0]["campo_extra"] = 1
    assert cliente.post("/predict", json=payload).status_code == 422


def test_predict_rechaza_fin_de_semana_incoherente(cliente, payload):
    # Martes marcado como fin de semana: la validación del esquema debe rechazarlo.
    payload["instancias"][0]["dia_semana"] = 1
    payload["instancias"][0]["fin_semana"] = 1
    assert cliente.post("/predict", json=payload).status_code == 422


def test_predict_rechaza_lote_vacio(cliente):
    assert cliente.post("/predict", json={"instancias": []}).status_code == 422
