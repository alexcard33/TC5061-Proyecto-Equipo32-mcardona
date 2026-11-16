"""Entry point register (mibici.modeling.register): limpieza de alias que ya se movieron.

Usa un cliente falso, así que no necesita servidor de MLflow.
"""

from types import SimpleNamespace

from mibici.modeling.register import limpiar_alias_obsoletos


class ClienteFalso:
    def __init__(self, alias: dict, versiones: list):
        self.alias = alias
        self.versiones = versiones
        self.tags = []
        self.descripciones = []

    def get_registered_model(self, nombre):
        return SimpleNamespace(aliases=self.alias)

    def search_model_versions(self, filtro):
        return self.versiones

    def set_model_version_tag(self, nombre, version, clave, valor):
        self.tags.append((version, clave, valor))

    def update_model_version(self, nombre, version, description):
        self.descripciones.append((version, description))


def _version(numero: int, alias: str | None, descripcion: str = "challenger: corrida x"):
    tags = {"alias": alias} if alias else {}
    return SimpleNamespace(version=str(numero), tags=tags, description=descripcion)


def test_version_que_perdio_el_alias_deja_de_anunciarlo():
    cliente = ClienteFalso(
        {"champion": "3", "challenger": "1"},
        [_version(1, "challenger"), _version(2, "challenger"), _version(3, "champion")],
    )

    limpiar_alias_obsoletos(cliente, "mibici-equipo32-p1-demanda")

    assert cliente.tags == [("2", "alias", "ninguno")]
    assert len(cliente.descripciones) == 1
    version, descripcion = cliente.descripciones[0]
    assert version == "2"
    assert descripcion.startswith("Sin alias (antes challenger).")


def test_limpieza_es_idempotente():
    cliente = ClienteFalso({"champion": 3}, [_version(2, "ninguno"), _version(3, "champion")])

    limpiar_alias_obsoletos(cliente, "mibici-equipo32-p1-demanda")

    assert cliente.tags == []
    assert cliente.descripciones == []
