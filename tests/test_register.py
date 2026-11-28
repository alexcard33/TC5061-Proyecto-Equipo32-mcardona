"""Model Registry: estado de producción, linaje y descripción de cada versión.

Usa un cliente falso, así que no necesita servidor de MLflow.
"""

from types import SimpleNamespace

from mlflow.exceptions import MlflowException

from mibici.modeling.register import TAGS_LINAJE
from mibici.modeling.registry_estado import (
    TAG_ETAPA_CORRIDA,
    estado_de,
    metrica_principal,
    sincronizar_versiones,
)

NOMBRE = "mibici-equipo32-p1-demanda"


class ClienteFalso:
    """Guarda en memoria lo mínimo del registry que usa sincronizar_versiones."""

    def __init__(self, aliases: dict, versiones: list, runs: dict):
        self.aliases = aliases
        self.versiones = {v.version: v for v in versiones}
        self.runs = runs
        self.tags_modelo = {}

    def get_registered_model(self, nombre):
        return SimpleNamespace(aliases=self.aliases)

    def search_model_versions(self, filtro):
        return list(self.versiones.values())

    def set_model_version_tag(self, nombre, version, clave, valor):
        self.versiones[str(version)].tags[clave] = valor

    def delete_model_version_tag(self, nombre, version, clave):
        del self.versiones[str(version)].tags[clave]

    def update_model_version(self, nombre, version, description):
        self.versiones[str(version)].description = description

    def get_run(self, run_id):
        if run_id not in self.runs:
            raise MlflowException(f"no existe {run_id}")
        return self.runs[run_id]

    def set_registered_model_tag(self, nombre, clave, valor):
        self.tags_modelo[clave] = valor


def _version(numero: int, run_id: str, **tags):
    base = {
        "stage_of_work": "candidate",
        "source_run_id": run_id,
        "source_run_name": f"corrida_{numero}",
        "git_commit": "01c6d6cbd0682827f83a89ce",
        "data_version": "7aa5e436d887",
    }
    base.update(tags)
    return SimpleNamespace(version=str(numero), run_id=run_id, tags=base, description="")


def _run(metricas: dict | None = None, params: dict | None = None):
    return SimpleNamespace(data=SimpleNamespace(metrics=metricas or {}, params=params or {}))


def _cliente():
    ab = {"ab_test_run_id": "ab1", "ab_test_decision": "challenger"}
    versiones = [
        _version(1, "r1", alias="challenger", **ab),
        _version(2, "r2", alias="challenger"),
        _version(3, "r3", alias="champion", **ab),
    ]
    runs = {
        "r1": _run({"test_mae": 6.4, "val_mae": 6.0}),
        "r2": _run({"test_mae": 6.9}),
        "r3": _run({"val_mae": 5.81234}),
        "ab1": _run(params={"champion_version": "1", "challenger_version": "3"}),
    }
    return ClienteFalso({"champion": 3, "challenger": "1"}, versiones, runs)


def test_estado_sale_del_alias():
    assert estado_de("champion") == "produccion"
    assert estado_de("challenger") == "en_evaluacion"
    assert estado_de(None) == "retirado"


def test_linaje_guarda_la_etapa_de_la_corrida_con_otro_nombre():
    assert TAGS_LINAJE["stage_of_work"] == TAG_ETAPA_CORRIDA
    assert "stage_of_work" not in TAGS_LINAJE.values()


def test_metrica_principal_usa_la_primera_disponible():
    assert metrica_principal("p1", {"val_mae": 5.0}) == ("val_mae", 5.0)
    assert metrica_principal("p1", {"test_mae": 4.0, "val_mae": 5.0}) == ("test_mae", 4.0)
    assert metrica_principal("p1", {"otra": 1.0}) is None


def test_cada_version_queda_con_estado_coherente_con_su_alias():
    cliente = _cliente()

    resumen = sincronizar_versiones(cliente, NOMBRE, "p1")

    estados = {v: cliente.versiones[v].tags["estado_produccion"] for v in ("1", "2", "3")}
    assert estados == {"1": "en_evaluacion", "2": "retirado", "3": "produccion"}
    assert cliente.versiones["2"].tags["alias"] == "ninguno"
    assert cliente.tags_modelo["version_produccion"] == "3"
    assert [f["estado"] for f in resumen] == ["en_evaluacion", "retirado", "produccion"]


def test_stage_of_work_se_migra_a_run_stage_of_work():
    cliente = _cliente()

    sincronizar_versiones(cliente, NOMBRE, "p1")

    for version in cliente.versiones.values():
        assert "stage_of_work" not in version.tags
        assert version.tags[TAG_ETAPA_CORRIDA] == "candidate"


def test_resultado_del_ab_usa_el_papel_que_tenia_cada_version():
    cliente = _cliente()

    sincronizar_versiones(cliente, NOMBRE, "p1")

    v1, v3 = cliente.versiones["1"].tags, cliente.versiones["3"].tags
    assert (v1["ab_test_rol"], v1["ab_test_resultado"]) == ("champion", "perdedora")
    assert (v3["ab_test_rol"], v3["ab_test_resultado"]) == ("challenger", "ganadora")
    assert "ab_test_resultado" not in cliente.versiones["2"].tags


def test_descripcion_resume_estado_linaje_y_metrica():
    cliente = _cliente()

    sincronizar_versiones(cliente, NOMBRE, "p1")

    descripcion = cliente.versiones["3"].description
    assert descripcion.startswith("Estado: produccion (@champion).")
    assert "corrida_3 (r3)" in descripcion
    assert "etapa candidate" in descripcion
    assert "val_mae = 5.8123" in descripcion
    assert "fue la ganadora" in descripcion


def test_sincronizar_es_idempotente():
    cliente = _cliente()
    sincronizar_versiones(cliente, NOMBRE, "p1")
    antes = {v: (dict(x.tags), x.description) for v, x in cliente.versiones.items()}

    sincronizar_versiones(cliente, NOMBRE, "p1")

    despues = {v: (dict(x.tags), x.description) for v, x in cliente.versiones.items()}
    assert antes == despues


def test_version_sin_corrida_no_rompe_la_sincronizacion():
    cliente = _cliente()
    del cliente.runs["r2"]

    resumen = sincronizar_versiones(cliente, NOMBRE, "p1")

    assert resumen[1]["metrica"] == "n/d"
    assert cliente.versiones["2"].tags["estado_produccion"] == "retirado"
