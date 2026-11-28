"""Estado de producción y descripción de cada versión del Model Registry.

El esquema de ciclo de vida es por aliases: champion es la versión en producción y
challenger la alternativa que se está evaluando. MLflow no guarda por sí mismo qué significa
cada alias, así que aquí se deja escrito en cada versión con el tag estado_produccion y con
una descripción que se arma igual para todas:

    champion   -> produccion
    challenger -> en_evaluacion
    sin alias  -> retirado

Además se corrige un tag que se prestaba a confusión. Al registrar una versión se copiaba
stage_of_work desde la corrida de origen; ese tag describe en qué etapa del experimento se
hizo la corrida (baseline, search o candidate), no el estado de la versión. Por eso el
champion aparecía como "candidate". Ahora se guarda como run_stage_of_work.

Todo lo de este módulo se puede correr las veces que haga falta: siempre deja las versiones
en el mismo estado a partir de los aliases vigentes.
"""

from loguru import logger
from mlflow.exceptions import MlflowException

from mibici.config import ALIAS_CHALLENGER, ALIAS_PRODUCCION, MLFLOW_EXPERIMENT

ESTADO_POR_ALIAS = {
    ALIAS_PRODUCCION: "produccion",
    ALIAS_CHALLENGER: "en_evaluacion",
}
ESTADO_SIN_ALIAS = "retirado"

# Métrica que se muestra en la descripción de cada versión. Se toma la primera que exista
# en la corrida de origen (el reentrenamiento de P1 no tiene partición de test, solo val).
METRICAS_DESCRIPCION = {
    "p1": ("test_mae", "val_mae"),
    "p2": ("test_pr_auc", "val_pr_auc"),
    "p3": ("test_silhouette", "train_silhouette"),
}

TAG_ETAPA_ANTERIOR = "stage_of_work"
TAG_ETAPA_CORRIDA = "run_stage_of_work"


def estado_de(alias: str | None) -> str:
    return ESTADO_POR_ALIAS.get(alias, ESTADO_SIN_ALIAS)


def alias_por_version(aliases: dict) -> dict:
    """Invierte el diccionario de aliases del registered model: versión -> alias."""
    return {str(version): alias for alias, version in (aliases or {}).items()}


def metrica_principal(question: str, metricas: dict) -> tuple[str, float] | None:
    for nombre in METRICAS_DESCRIPCION.get(question, ()):
        if nombre in metricas:
            return nombre, float(metricas[nombre])
    return None


def resultado_ab(client, version: str, tags: dict) -> tuple[str, str] | None:
    """Papel que tuvo la versión en el A/B y si ganó o perdió.

    El A/B guarda en sus parámetros qué versión era champion y cuál challenger en ese
    momento. Con eso y la decisión (ab_test_decision) se sabe cuál ganó, aunque después
    los aliases se hayan movido.
    """
    run_id = tags.get("ab_test_run_id")
    decision = tags.get("ab_test_decision")
    if not run_id or not decision:
        return None
    try:
        params = client.get_run(run_id).data.params
    except MlflowException:
        return None
    for rol in (ALIAS_PRODUCCION, ALIAS_CHALLENGER):
        if params.get(f"{rol}_version") == str(version):
            return rol, ("ganadora" if rol == decision else "perdedora")
    return None


def descripcion_version(
    alias: str | None,
    estado: str,
    tags: dict,
    metrica: tuple[str, float] | None,
    ab: tuple[str, str] | None,
) -> str:
    partes = [f"Estado: {estado}" + (f" (@{alias})." if alias else ".")]
    partes.append(
        f"Origen: corrida {tags.get('source_run_name', 'n/d')} "
        f"({tags.get('source_run_id', 'n/d')}) del experimento "
        f"{tags.get('source_experiment', MLFLOW_EXPERIMENT)}, etapa "
        f"{tags.get(TAG_ETAPA_CORRIDA, 'n/d')}."
    )
    partes.append(
        f"Commit {tags.get('git_commit', 'n/d')[:12]}; "
        f"data_version {tags.get('data_version', 'n/d')}."
    )
    if metrica:
        partes.append(f"{metrica[0]} = {metrica[1]:.4f}.")
    if ab:
        partes.append(
            f"A/B {tags.get('ab_test_run_id', '')[:12]}: participó como {ab[0]} y fue la {ab[1]}."
        )
    return " ".join(partes)


def sincronizar_versiones(client, nombre: str, question: str) -> list[dict]:
    """Deja alias, estado, etapa de la corrida y descripción coherentes en cada versión.

    Regresa un resumen por versión, que sirve para el log y para la tabla del README.
    """
    modelo = client.get_registered_model(nombre)
    alias_de = alias_por_version(modelo.aliases)
    resumen = []

    for version in sorted(
        client.search_model_versions(f"name = '{nombre}'"), key=lambda v: int(v.version)
    ):
        numero = str(version.version)
        tags = dict(version.tags or {})
        alias = alias_de.get(numero)
        estado = estado_de(alias)

        # El tag viejo pasa a run_stage_of_work; si ya se migró antes, no se toca.
        if TAG_ETAPA_ANTERIOR in tags:
            tags.setdefault(TAG_ETAPA_CORRIDA, tags[TAG_ETAPA_ANTERIOR])
            client.set_model_version_tag(
                nombre, numero, TAG_ETAPA_CORRIDA, tags[TAG_ETAPA_CORRIDA]
            )
            client.delete_model_version_tag(nombre, numero, TAG_ETAPA_ANTERIOR)
            del tags[TAG_ETAPA_ANTERIOR]

        nuevos = {"alias": alias or "ninguno", "estado_produccion": estado}
        ab = resultado_ab(client, numero, tags)
        if ab:
            nuevos["ab_test_rol"], nuevos["ab_test_resultado"] = ab
        for clave, valor in nuevos.items():
            if tags.get(clave) != valor:
                client.set_model_version_tag(nombre, numero, clave, valor)
        tags.update(nuevos)

        metrica = None
        try:
            metrica = metrica_principal(question, client.get_run(version.run_id).data.metrics)
        except MlflowException:
            logger.warning(f"{nombre} v{numero}: no se encontró la corrida {version.run_id}")

        descripcion = descripcion_version(alias, estado, tags, metrica, ab)
        if (version.description or "") != descripcion:
            client.update_model_version(nombre, numero, description=descripcion)

        resumen.append(
            {
                "modelo": nombre,
                "version": numero,
                "alias": alias or "-",
                "estado": estado,
                "corrida": tags.get("source_run_name", "n/d"),
                "data_version": tags.get("data_version", "n/d"),
                "metrica": f"{metrica[0]} = {metrica[1]:.4f}" if metrica else "n/d",
            }
        )
        logger.info(f"{nombre} v{numero}: {estado} ({alias or 'sin alias'})")

    # En el registered model queda a la vista cuál versión está en producción.
    produccion = modelo.aliases.get(ALIAS_PRODUCCION)
    if produccion is not None:
        client.set_registered_model_tag(nombre, "version_produccion", str(produccion))
    return resumen
