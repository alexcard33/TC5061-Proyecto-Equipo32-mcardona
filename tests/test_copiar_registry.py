"""Copia del Model Registry local al MLflow de Compose (mibici.modeling.copiar_registry).

Solo se prueban las partes que no necesitan servidor: qué tags se copian y cómo se
reconocen las versiones que ya se copiaron antes.
"""

from types import SimpleNamespace

from mibici.modeling.copiar_registry import (
    TAG_REGISTRY_ORIGEN,
    TAG_RUN_ORIGEN,
    TAG_VERSION_ORIGEN,
    tags_de_corrida,
    versiones_copiadas,
)


def test_tags_internos_de_mlflow_no_se_copian():
    tags = {
        "mlflow.runName": "p1_v2_reentrenado",
        "mlflow.log-model.history": "[...]",
        "question": "p1",
        "data_version": "7aa5e436d887",
    }

    copiados = tags_de_corrida(tags, "abc123", "sqlite:///mlflow.db")

    assert not any(k.startswith("mlflow.") for k in copiados)
    assert copiados["question"] == "p1"
    assert copiados["data_version"] == "7aa5e436d887"


def test_la_corrida_copiada_apunta_a_su_origen():
    copiados = tags_de_corrida({}, "abc123", "sqlite:///mlflow.db")

    assert copiados[TAG_RUN_ORIGEN] == "abc123"
    assert copiados[TAG_REGISTRY_ORIGEN] == "sqlite:///mlflow.db"


def test_versiones_copiadas_se_indexan_por_su_version_de_origen():
    versiones = [
        SimpleNamespace(version="1", tags={TAG_VERSION_ORIGEN: "1"}),
        SimpleNamespace(version="2", tags={TAG_VERSION_ORIGEN: "2"}),
        SimpleNamespace(version="3", tags={}),
    ]
    cliente = SimpleNamespace(search_model_versions=lambda filtro: versiones)

    copiadas = versiones_copiadas(cliente, "mibici-equipo32-p1-demanda")

    assert set(copiadas) == {"1", "2"}
    assert copiadas["2"].version == "2"


def test_destino_vacio_no_tiene_versiones_copiadas():
    cliente = SimpleNamespace(search_model_versions=lambda filtro: [])

    assert versiones_copiadas(cliente, "mibici-equipo32-p1-demanda") == {}
