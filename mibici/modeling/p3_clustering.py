"""Pipeline reproducible de clustering de estaciones MiBici."""

import itertools
import json
from pathlib import Path
import subprocess

import folium
from folium.plugins import Fullscreen
import joblib
from loguru import logger
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import typer

from mibici.config import (
    ESTACIONES_PARQUET,
    MANIFEST_PATH,
    MLFLOW_TRACKING_URI,
    P3_AGGLO_PARAMS,
    P3_BASELINE_PARAMS,
    P3_GEO_PATH,
    P3_GMM_PARAMS,
    P3_KMEANS_PARAMS,
    P3_KMEANS_SEARCH_GRID,
    P3_MAP_PATH,
    P3_MAX_DURATION_SECONDS,
    P3_MAX_INACTIVITY_DAYS,
    P3_MIN_DURATION_SECONDS,
    P3_MIN_HISTORY_DAYS,
    P3_MLFLOW_BASELINE_RUN_NAME,
    P3_MLFLOW_CANDIDATE_RUN_NAME,
    P3_MLFLOW_EXPERIMENT,
    P3_MLFLOW_GMM_RUN_NAME,
    P3_MLFLOW_RUN_NAME,
    P3_MLFLOW_SEARCH_RUN_NAME,
    P3_MODEL_PATH,
    P3_OUTPUT_DIR,
    P3_PCA_PATH,
    P3_QUESTION,
    P3_REDUCED_START,
    P3_TABLE_PATH,
    P3_TEST_END,
    P3_TEST_START,
    P3_TRAIN_END,
    P3_TRAIN_START,
    PROJ_ROOT,
    SEED,
    TEAM_ID,
    VIAJES_PARQUET,
)
from mibici.features import (
    P3_MODEL_FEATURES,
    p3_agregar_entropias,
    p3_construir_features,
    p3_filtrar_viajes,
)
from mibici.modeling.mlflow_utils import (
    registrar_artefactos,
    registrar_config,
    registrar_datos,
    tags_estandar,
)

app = typer.Typer()

COLORES_PERFIL = {
    "Fin de semana y mediodia": "#e74c3c",
    "Llegadas matutinas / salidas vespertinas": "#3498db",
    "Salidas matutinas / llegadas vespertinas": "#2ecc71",
    "Alta intensidad balanceada": "#8e44ad",
}


def preparar_features(
    features: pd.DataFrame,
    scaler: StandardScaler | None = None,
):
    x = features[P3_MODEL_FEATURES].copy()

    x["salidas_dia"] = np.log1p(x["salidas_dia"])
    x["llegadas_dia"] = np.log1p(x["llegadas_dia"])

    if scaler is None:
        scaler = StandardScaler()
        valores = scaler.fit_transform(x)
    else:
        valores = scaler.transform(x)

    x_scaled = pd.DataFrame(
        valores,
        index=x.index,
        columns=x.columns,
    )

    return x_scaled, scaler


def asignar_perfiles(
    features: pd.DataFrame,
    labels: pd.Series,
) -> dict[int, str]:
    perfiles = (
        features[P3_MODEL_FEATURES]
        .assign(cluster=labels)
        .groupby("cluster")
        .mean()
    )

    disponibles = {int(x) for x in perfiles.index}

    intensidad = (
        perfiles["salidas_dia"]
        + perfiles["llegadas_dia"]
    )

    cluster_alta = int(intensidad.idxmax())
    disponibles.remove(cluster_alta)

    cluster_fin_semana = int(
        perfiles.loc[
            sorted(disponibles),
            "share_fin_semana",
        ].idxmax()
    )
    disponibles.remove(cluster_fin_semana)

    score_salidas = (
        perfiles.loc[
            sorted(disponibles),
            "salida_manana_share",
        ]
        + perfiles.loc[
            sorted(disponibles),
            "llegada_tarde_share",
        ]
    )

    cluster_salidas = int(score_salidas.idxmax())
    disponibles.remove(cluster_salidas)

    cluster_llegadas = disponibles.pop()

    return {
        cluster_alta: "Alta intensidad balanceada",
        cluster_fin_semana: "Fin de semana y mediodia",
        cluster_salidas: "Salidas matutinas / llegadas vespertinas",
        cluster_llegadas: "Llegadas matutinas / salidas vespertinas",
    }


def generar_pca(
    x_train: pd.DataFrame,
    labels: pd.Series,
    perfiles: dict[int, str],
    output_path: Path,
):
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(x_train)

    df = pd.DataFrame(
        componentes,
        index=x_train.index,
        columns=["PC1", "PC2"],
    )

    df["cluster"] = labels.astype(int)
    df["perfil"] = df["cluster"].map(perfiles)

    fig, ax = plt.subplots(figsize=(11, 7))

    for perfil, color in COLORES_PERFIL.items():
        datos = df.loc[df["perfil"] == perfil]

        ax.scatter(
            datos["PC1"],
            datos["PC2"],
            s=55,
            alpha=0.72,
            label=f"{perfil} ({len(datos)})",
            color=color,
            edgecolors="white",
            linewidths=0.4,
        )

    centroides = (
        df.groupby("perfil")[["PC1", "PC2"]]
        .mean()
    )

    for perfil, fila in centroides.iterrows():
        ax.scatter(
            fila["PC1"],
            fila["PC2"],
            s=240,
            marker="X",
            color=COLORES_PERFIL[perfil],
            edgecolors="black",
            linewidths=1.2,
        )

    ax.axhline(0, linewidth=0.6, alpha=0.4)
    ax.axvline(0, linewidth=0.6, alpha=0.4)

    ax.set_xlabel(
        f"PC1 — patrón temporal "
        f"({pca.explained_variance_ratio_[0]:.1%})"
    )

    ax.set_ylabel(
        f"PC2 — intensidad de uso "
        f"({pca.explained_variance_ratio_[1]:.1%})"
    )

    ax.set_title(
        "Perfiles de uso de estaciones MiBici\n"
        "Proyección PCA de TRAIN por grupo"
    )

    ax.legend(
        title="Perfil",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return pca


def generar_mapa(
    resultado_geo: pd.DataFrame,
    output_path: Path,
):
    mapa = folium.Map(
        location=[
            resultado_geo["latitude"].mean(),
            resultado_geo["longitude"].mean(),
        ],
        tiles=None,
        zoom_start=13,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Tiles © Esri",
        name="Esri WorldStreetMap",
        overlay=False,
        control=False,
    ).add_to(mapa)

    Fullscreen(
        position="topleft",
        title="Pantalla completa",
        title_cancel="Salir",
    ).add_to(mapa)

    capas = {}

    for perfil in COLORES_PERFIL:
        n = int(
            (resultado_geo["perfil"] == perfil).sum()
        )

        capas[perfil] = folium.FeatureGroup(
            name=f"{perfil} ({n})",
            show=True,
        )

        capas[perfil].add_to(mapa)

    estados = {
        "estable": "Estable",
        "cambio": "Cambió de perfil",
        "no_comparable": "No comparable",
    }

    for _, fila in resultado_geo.iterrows():
        perfil = fila["perfil"]
        color = COLORES_PERFIL[perfil]
        estado = estados[fila["estado_test"]]

        popup = f"""
        <div style="width:320px">
        <b>{fila['name']}</b><br><br>
        <b>Estación:</b> {int(fila['estacion'])}<br>
        <b>Perfil:</b> {perfil}<br>
        <b>Cluster:</b> {int(fila['cluster'])}<br>
        <b>Estado en TEST:</b> {estado}<br><br>
        <b>Salidas/día:</b> {fila['salidas_dia']:.1f}<br>
        <b>Llegadas/día:</b> {fila['llegadas_dia']:.1f}<br>
        <b>Balance:</b> {fila['balance_relativo']:.3f}<br>
        <b>Fin de semana:</b> {fila['share_fin_semana']:.1%}<br><br>
        <b>Entropía destinos:</b> {fila['entropia_destinos']:.3f}<br>
        <b>Entropía orígenes:</b> {fila['entropia_origenes']:.3f}
        </div>
        """

        folium.CircleMarker(
            location=[
                fila["latitude"],
                fila["longitude"],
            ],
            radius=5,
            color=color,
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.80,
            popup=folium.Popup(
                popup,
                max_width=350,
            ),
            tooltip=(
                f"{fila['name']} | "
                f"{perfil} | {estado}"
            ),
        ).add_to(capas[perfil])

    folium.LayerControl(
        collapsed=False,
        position="topright",
    ).add_to(mapa)

    mapa.fit_bounds(
        [
            [
                resultado_geo["latitude"].min(),
                resultado_geo["longitude"].min(),
            ],
            [
                resultado_geo["latitude"].max(),
                resultado_geo["longitude"].max(),
            ],
        ]
    )

    mapa.save(output_path)


MODELOS = ("baseline", "kmeans", "agglomerative", "gmm")


def preparar_datos(
    viajes_path: Path = VIAJES_PARQUET,
    estaciones_path: Path = ESTACIONES_PARQUET,
) -> dict:
    """Viajes filtrados, partición temporal y features escaladas de TRAIN, TEST e histórico."""
    P3_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    P3_PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
    P3_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    viajes = pd.read_parquet(viajes_path)
    estaciones = pd.read_parquet(estaciones_path)

    viajes["inicio_viaje"] = pd.to_datetime(
        viajes["inicio_viaje"]
    )
    viajes["fin_viaje"] = pd.to_datetime(
        viajes["fin_viaje"]
    )

    viajes = p3_filtrar_viajes(
        viajes,
        min_duration_seconds=P3_MIN_DURATION_SECONDS,
        max_duration_seconds=P3_MAX_DURATION_SECONDS,
    )

    train = viajes.loc[
        (viajes["inicio_viaje"] >= pd.Timestamp(P3_TRAIN_START))
        & (viajes["inicio_viaje"] <= pd.Timestamp(P3_TRAIN_END))
        & (viajes["fin_viaje"] < pd.Timestamp(P3_TEST_START))
    ].copy()

    test = viajes.loc[
        (viajes["inicio_viaje"] >= pd.Timestamp(P3_TEST_START))
        & (viajes["fin_viaje"] < pd.Timestamp(P3_TEST_END))
    ].copy()

    features_train, _ = (
        p3_construir_features(
            train,
            fin_periodo=pd.Timestamp(P3_TRAIN_END),
            max_inactivity_days=P3_MAX_INACTIVITY_DAYS,
            min_history_days=P3_MIN_HISTORY_DAYS,
        )
    )

    features_test, _ = (
        p3_construir_features(
            test,
            fin_periodo=pd.Timestamp(P3_TEST_END) - pd.Timedelta(seconds=1),
            max_inactivity_days=P3_MAX_INACTIVITY_DAYS,
            min_history_days=P3_MIN_HISTORY_DAYS,
        )
    )

    train_reducido = train.loc[
        train["inicio_viaje"] >= pd.Timestamp(P3_REDUCED_START)
    ].copy()

    features_reducido, _ = p3_construir_features(
        train_reducido,
        fin_periodo=pd.Timestamp(P3_TRAIN_END),
        max_inactivity_days=P3_MAX_INACTIVITY_DAYS,
        min_history_days=P3_MIN_HISTORY_DAYS,
    )

    x_train, scaler = preparar_features(
        features_train
    )

    x_test, _ = preparar_features(
        features_test,
        scaler=scaler,
    )

    x_reducido, _ = preparar_features(
        features_reducido
    )

    return {
        "viajes": viajes,
        "estaciones": estaciones,
        "train": train,
        "test": test,
        "features_train": features_train,
        "x_train": x_train,
        "x_test": x_test,
        "x_reducido": x_reducido,
        "scaler": scaler,
    }


def params_comunes(data_version: str) -> dict:
    """Parámetros de datos y preprocesamiento que comparten todos los runs de P3."""
    return {
        "features": len(P3_MODEL_FEATURES),
        "data_version": data_version,
        "max_inactivity_days": P3_MAX_INACTIVITY_DAYS,
        "min_history_days": P3_MIN_HISTORY_DAYS,
        "min_duration_seconds": P3_MIN_DURATION_SECONDS,
        "max_duration_seconds": P3_MAX_DURATION_SECONDS,
        "train_inicio": P3_TRAIN_START[:10],
        "train_fin": P3_TRAIN_END[:10],
        "transformacion_intensidad": "log1p",
        "scaler": "StandardScaler",
    }


def tags_run(familia: str, etapa: str, git_commit: str, data_version: str) -> dict:
    """Tags de equipo, commit y versión de datos más los estándar (question, family, stage)."""
    return {
        "equipo": TEAM_ID,
        "mlflow.user": TEAM_ID,
        "problema": "P3",
        "tipo": "clustering",
        "pregunta": P3_QUESTION,
        "git_commit": git_commit,
        "data_version": data_version,
        **tags_estandar("p3", familia, etapa),
    }


def predecir(modelo, x: pd.DataFrame):
    """Etiquetas para x: predict si el modelo lo tiene; si no (Agglomerative), se reagrupa x.

    ARI y NMI no dependen de la numeración de los grupos, así que siguen siendo comparables.
    """
    if hasattr(modelo, "predict"):
        return modelo.predict(x)
    return clone(modelo).fit_predict(x)


def firma_clustering(modelo, x: pd.DataFrame):
    """Firma con entradas y salida (grupo); sin predict solo se declaran las entradas."""
    ejemplo = x.head(5)
    if hasattr(modelo, "predict"):
        return infer_signature(ejemplo, modelo.predict(ejemplo))
    return infer_signature(ejemplo)


def correr_comparacion(modelo, etapa, params_log, run_name, d, git_commit, data_version) -> dict:
    """Ajusta un modelo de comparación (baseline o agglomerative) y registra sus métricas TRAIN."""
    x_train = d["x_train"]
    labels = pd.Series(modelo.fit_predict(x_train), index=x_train.index)
    metricas = {
        "train_silhouette": silhouette_score(x_train, labels),
        "train_davies_bouldin": davies_bouldin_score(x_train, labels),
        "train_calinski_harabasz": calinski_harabasz_score(x_train, labels),
    }

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({**params_log, **params_comunes(data_version)})
        mlflow.log_metrics(metricas)
        mlflow.set_tags(tags_run("kmeans", etapa, git_commit, data_version))
        registrar_config(params_log)
        registrar_datos(x_train, "p3_train_estaciones", VIAJES_PARQUET)

    logger.info(
        f"{run_name}: train_silhouette={metricas['train_silhouette']:.4f} "
        f"train_davies_bouldin={metricas['train_davies_bouldin']:.4f}"
    )
    return metricas


def correr_candidato(crear_modelo, familia, params_log, run_name, d, git_commit, data_version) -> dict:
    """Candidata de 4 grupos: perfiles, estabilidad TRAIN-TEST e histórica, mapa, PCA y modelo."""
    viajes = d["viajes"]
    estaciones = d["estaciones"]
    train = d["train"]
    test = d["test"]
    features_train = d["features_train"]
    x_train = d["x_train"]
    x_test = d["x_test"]
    x_reducido = d["x_reducido"]
    scaler = d["scaler"]

    modelo = crear_modelo()

    labels_train = pd.Series(
        modelo.fit_predict(x_train),
        index=x_train.index,
        name="cluster_train",
    )

    labels_test = pd.Series(
        predecir(modelo, x_test),
        index=x_test.index,
        name="cluster_test",
    )

    modelo_reducido = crear_modelo()

    labels_reducido = pd.Series(
        modelo_reducido.fit_predict(x_reducido),
        index=x_reducido.index,
    )

    perfiles = asignar_perfiles(
        features_train,
        labels_train,
    )

    comunes_test = (
        x_train.index
        .intersection(x_test.index)
    )

    comunes_historico = (
        x_train.index
        .intersection(x_reducido.index)
    )

    comparacion_test = pd.DataFrame(
        {
            "cluster_train": labels_train.loc[
                comunes_test
            ],
            "cluster_test": labels_test.loc[
                comunes_test
            ],
        }
    )

    comparacion_test["cambio"] = (
        comparacion_test["cluster_train"]
        != comparacion_test["cluster_test"]
    )

    entropias = p3_agregar_entropias(
        features_train,
        train,
    )

    resultado = features_train.copy()

    resultado["cluster"] = (
        labels_train.astype(int)
    )

    resultado["perfil"] = (
        resultado["cluster"].map(perfiles)
    )

    resultado["entropia_destinos"] = (
        entropias["entropia_destinos"]
    )

    resultado["entropia_origenes"] = (
        entropias["entropia_origenes"]
    )

    estado_test = pd.Series(
        "no_comparable",
        index=resultado.index,
        dtype="object",
    )

    estado_test.loc[comparacion_test.index] = np.where(
        comparacion_test["cambio"],
        "cambio",
        "estable",
    )

    resultado["estado_test"] = estado_test

    resultado_geo = (
        resultado
        .reset_index()
        .merge(
            estaciones[
                [
                    "id",
                    "name",
                    "latitude",
                    "longitude",
                ]
            ],
            left_on="estacion",
            right_on="id",
            how="left",
            validate="one_to_one",
        )
        .drop(columns="id")
    )

    metricas_cluster = (
        resultado
        .reset_index()
        .groupby(
            ["cluster", "perfil"],
            as_index=False,
        )
        .agg(
            estaciones_train=(
                "estacion",
                "size",
            ),
            salidas_dia=(
                "salidas_dia",
                "mean",
            ),
            llegadas_dia=(
                "llegadas_dia",
                "mean",
            ),
            balance_relativo=(
                "balance_relativo",
                "mean",
            ),
            share_fin_semana=(
                "share_fin_semana",
                "mean",
            ),
            entropia_destinos=(
                "entropia_destinos",
                "mean",
            ),
            entropia_origenes=(
                "entropia_origenes",
                "mean",
            ),
        )
    )

    estabilidad_cluster = (
        comparacion_test
        .groupby("cluster_train")
        .agg(
            comparables_test=(
                "cambio",
                "size",
            ),
            cambios_test=(
                "cambio",
                "sum",
            ),
        )
    )

    estabilidad_cluster["estables_test"] = (
        estabilidad_cluster["comparables_test"]
        - estabilidad_cluster["cambios_test"]
    )

    estabilidad_cluster["estabilidad_pct"] = (
        estabilidad_cluster["estables_test"]
        / estabilidad_cluster["comparables_test"]
        * 100
    )

    tabla_maestra = (
        metricas_cluster
        .merge(
            estabilidad_cluster,
            left_on="cluster",
            right_index=True,
            how="left",
        )
        .sort_values("cluster")
    )

    tabla_path = (
        P3_TABLE_PATH
    )

    geo_path = (
        P3_GEO_PATH
    )

    mapa_path = (
        P3_MAP_PATH
    )

    pca_path = (
        P3_PCA_PATH
    )

    model_path = (
        P3_MODEL_PATH
    )

    tabla_maestra.to_csv(
        tabla_path,
        index=False,
    )

    resultado_geo.to_parquet(
        geo_path,
        index=False,
    )

    pca = generar_pca(
        x_train,
        labels_train,
        perfiles,
        pca_path,
    )

    generar_mapa(
        resultado_geo,
        mapa_path,
    )

    bundle = {
        "scaler": scaler,
        "modelo": modelo,
        "features": P3_MODEL_FEATURES,
        "perfiles": perfiles,
    }

    joblib.dump(
        bundle,
        model_path,
    )

    metricas = {
        "train_silhouette": silhouette_score(
            x_train,
            labels_train,
        ),
        "test_silhouette": silhouette_score(
            x_test,
            labels_test,
        ),
        "train_davies_bouldin": davies_bouldin_score(
            x_train,
            labels_train,
        ),
        "test_davies_bouldin": davies_bouldin_score(
            x_test,
            labels_test,
        ),
        "train_calinski_harabasz": calinski_harabasz_score(
            x_train,
            labels_train,
        ),
        "test_calinski_harabasz": calinski_harabasz_score(
            x_test,
            labels_test,
        ),
        "ari_train_test": adjusted_rand_score(
            labels_train.loc[comunes_test],
            labels_test.loc[comunes_test],
        ),
        "nmi_train_test": normalized_mutual_info_score(
            labels_train.loc[comunes_test],
            labels_test.loc[comunes_test],
        ),
        "estabilidad_exacta_test": (
            comparacion_test["cluster_train"]
            == comparacion_test["cluster_test"]
        ).mean(),
        "ari_historico_reducido": adjusted_rand_score(
            labels_train.loc[comunes_historico],
            labels_reducido.loc[comunes_historico],
        ),
        "nmi_historico_reducido": normalized_mutual_info_score(
            labels_train.loc[comunes_historico],
            labels_reducido.loc[comunes_historico],
        ),
        "estaciones_train": len(x_train),
        "estaciones_test": len(x_test),
        "estaciones_comunes": len(comunes_test),
        "pca_varianza_acumulada": (
            pca.explained_variance_ratio_.sum()
        ),
    }

    if not hasattr(modelo, "predict"):
        # TEST se reagrupó: la coincidencia exacta de etiquetas no tiene sentido
        metricas.pop("estabilidad_exacta_test")

    with mlflow.start_run(
        run_name=run_name
    ) as run:
        mlflow.log_params(
            {
                **params_log,
                "features": len(P3_MODEL_FEATURES),
                "data_version": data_version,
                "max_inactivity_days": P3_MAX_INACTIVITY_DAYS,
                "min_history_days": P3_MIN_HISTORY_DAYS,
                "min_duration_seconds": P3_MIN_DURATION_SECONDS,
                "max_duration_seconds": P3_MAX_DURATION_SECONDS,
                "split": "temporal_80_20",
                "train_inicio": P3_TRAIN_START[:10],
                "train_fin": P3_TRAIN_END[:10],
                "test_inicio": P3_TEST_START[:10],
                "test_fin": str(
                    (
                        pd.Timestamp(P3_TEST_END)
                        - pd.Timedelta(seconds=1)
                    ).date()
                ),
                "transformacion_intensidad": "log1p",
                "scaler": "StandardScaler",
            }
        )

        mlflow.log_metrics(metricas)

        mlflow.set_tags(
            {
                "equipo": TEAM_ID,
                "mlflow.user": TEAM_ID,
                "problema": "P3",
                "tipo": "clustering",
                "pregunta": P3_QUESTION,
                "git_commit": git_commit,
                "data_version": data_version,
                **tags_estandar("p3", familia, "candidate"),
            }
        )

        mlflow.sklearn.log_model(
            modelo,
            name="modelo",
            signature=firma_clustering(modelo, x_train),
            input_example=x_train.head(5),
        )
        registrar_artefactos([pca_path, mapa_path, tabla_path, geo_path])
        mlflow.log_artifact(str(model_path), artifact_path="model_bundle")
        registrar_config(params_log)
        registrar_datos(x_train, "p3_train_estaciones", VIAJES_PARQUET)

        run_id = run.info.run_id

    logger.info(
        f"Viajes limpios: {len(viajes):,}"
    )

    logger.info(
        f"Train: {len(train):,} | "
        f"Test: {len(test):,}"
    )

    logger.info(
        f"Estaciones TRAIN: {len(x_train)} | "
        f"TEST: {len(x_test)}"
    )

    logger.info(
        f"Perfiles: {perfiles}"
    )

    logger.info(
        f"Silhouette TRAIN: "
        f"{metricas['train_silhouette']:.4f}"
    )

    logger.info(
        f"Silhouette TEST: "
        f"{metricas['test_silhouette']:.4f}"
    )

    logger.info(
        f"ARI TRAIN-TEST: "
        f"{metricas['ari_train_test']:.4f}"
    )

    logger.info(
        f"ARI histórico reducido: "
        f"{metricas['ari_historico_reducido']:.4f}"
    )

    logger.success(
        f"Pipeline P3 terminado | "
        f"MLflow run: {run_id}"
    )

    return metricas


def config_candidato(nombre: str, params: dict):
    """Devuelve (familia, constructor, parámetros a registrar, nombre de run por defecto)."""
    if nombre == "kmeans":
        p = {"random_state": SEED, **P3_KMEANS_PARAMS, **params}
        k = p["n_clusters"]
        familia, clase, run_default = "kmeans", KMeans, P3_MLFLOW_RUN_NAME
    elif nombre == "agglomerative":
        p = {**P3_AGGLO_PARAMS, **params}
        k = p["n_clusters"]
        familia, clase, run_default = "hierarchical", AgglomerativeClustering, P3_MLFLOW_CANDIDATE_RUN_NAME
    elif nombre == "gmm":
        p = {"random_state": SEED, **P3_GMM_PARAMS, **params}
        k = p["n_components"]
        familia, clase, run_default = "gaussian_mixture", GaussianMixture, P3_MLFLOW_GMM_RUN_NAME
    else:
        raise ValueError(
            f"Modelo de P3 desconocido: {nombre!r}. Opciones: {', '.join(MODELOS)}, search o all."
        )
    if k != len(COLORES_PERFIL):
        raise ValueError(
            f"Las candidatas de P3 nombran {len(COLORES_PERFIL)} perfiles; no se puede usar k={k}. "
            "Para explorar otros k use -P model=search."
        )
    params_log = {"modelo": clase.__name__, "k": k}
    params_log.update({c: v for c, v in p.items() if c not in ("n_clusters", "n_components")})
    return familia, (lambda: clase(**p)), params_log, run_default


def ejecutar_busqueda(run_name, d, git_commit, data_version) -> dict:
    """Búsqueda de k con K-Means: un run padre y un run hijo (nested) por valor de k.

    Se elige solo con TRAIN (mayor silhouette, desempate menor Davies-Bouldin); además se
    registra el ARI frente a la ventana histórica reducida como señal de estabilidad.
    """
    x_train, x_reducido = d["x_train"], d["x_reducido"]
    comunes = x_train.index.intersection(x_reducido.index)
    claves = list(P3_KMEANS_SEARCH_GRID)
    combinaciones = [
        dict(zip(claves, valores)) for valores in itertools.product(*P3_KMEANS_SEARCH_GRID.values())
    ]
    resultados = []
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(tags_run("kmeans", "search", git_commit, data_version))
        mlflow.log_params(
            {
                "modelo": "KMeans",
                "criterio": "max train_silhouette; desempate min train_davies_bouldin",
                "combinaciones": len(combinaciones),
                "grid": json.dumps(P3_KMEANS_SEARCH_GRID),
                **params_comunes(data_version),
            }
        )
        for i, combo in enumerate(combinaciones, start=1):
            p = {"random_state": SEED, **P3_KMEANS_PARAMS, **combo}
            labels = pd.Series(KMeans(**p).fit_predict(x_train), index=x_train.index)
            labels_red = pd.Series(KMeans(**p).fit_predict(x_reducido), index=x_reducido.index)
            metricas = {
                "train_silhouette": silhouette_score(x_train, labels),
                "train_davies_bouldin": davies_bouldin_score(x_train, labels),
                "train_calinski_harabasz": calinski_harabasz_score(x_train, labels),
                "ari_historico_reducido": adjusted_rand_score(
                    labels.loc[comunes], labels_red.loc[comunes]
                ),
            }
            params_log = {"modelo": "KMeans", "k": p["n_clusters"]}
            params_log.update({c: v for c, v in p.items() if c != "n_clusters"})
            with mlflow.start_run(run_name=f"{run_name}_k{p['n_clusters']}", nested=True) as hijo:
                mlflow.set_tags(tags_run("kmeans", "search", git_commit, data_version))
                mlflow.log_params({**params_log, **params_comunes(data_version)})
                mlflow.log_metrics(metricas)
            resultados.append({"run_id": hijo.info.run_id, **combo, **metricas})
            logger.info(
                f"{run_name}_{i:02d} {combo}: train_silhouette={metricas['train_silhouette']:.4f} "
                f"train_davies_bouldin={metricas['train_davies_bouldin']:.4f}"
            )

        tabla = (
            pd.DataFrame(resultados)
            .sort_values(["train_silhouette", "train_davies_bouldin"], ascending=[False, True])
            .reset_index(drop=True)
        )
        mejor = tabla.iloc[0]
        mlflow.log_params({f"mejor_{k}": mejor[k] for k in claves})
        mlflow.log_metrics(
            {
                "mejor_train_silhouette": mejor["train_silhouette"],
                "mejor_train_davies_bouldin": mejor["train_davies_bouldin"],
            }
        )
        mlflow.set_tag("mejor_run_hijo", mejor["run_id"])
        ruta = P3_OUTPUT_DIR / f"{run_name}_resultados.csv"
        tabla.to_csv(ruta, index=False)
        registrar_artefactos([ruta])
        registrar_config(P3_KMEANS_PARAMS, {"grid": P3_KMEANS_SEARCH_GRID})
        registrar_datos(x_train, "p3_train_estaciones", VIAJES_PARQUET)

    logger.success(
        f"Mejor k: {int(mejor['n_clusters'])} (train_silhouette={mejor['train_silhouette']:.4f})"
    )
    return {"mejor": mejor.to_dict(), "resultados": tabla}


def ejecutar(
    model: str = "all",
    params: dict | None = None,
    run_name: str | None = None,
    viajes_path: Path = VIAJES_PARQUET,
    estaciones_path: Path = ESTACIONES_PARQUET,
    experiment_name: str = P3_MLFLOW_EXPERIMENT,
) -> dict:
    """Entrena y registra un modelo de P3, la búsqueda (model="search") o todos (model="all")."""
    params = params or {}
    if model == "all" and (params or run_name):
        raise ValueError("params y run_name solo se pueden usar al entrenar un modelo concreto.")
    if model == "search" and params:
        raise ValueError("La búsqueda usa P3_KMEANS_SEARCH_GRID de config.py; no recibe params.")
    nombres = MODELOS if model == "all" else (model,)
    candidatos = {
        nombre: config_candidato(nombre, params)  # valida antes de cargar datos
        for nombre in nombres
        if nombre not in ("baseline", "search")
    }

    logger.info(f"Iniciando pipeline P3 | modelos={', '.join(nombres)} | params={params}")
    d = preparar_datos(viajes_path, estaciones_path)
    data_version = json.loads(MANIFEST_PATH.read_text())["data_version"]
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJ_ROOT, text=True
    ).strip()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    resultados = {}
    for nombre in nombres:
        if nombre == "baseline":
            p = {"random_state": SEED, **P3_BASELINE_PARAMS, **params}
            params_log = {"modelo": "KMeans", "k": p["n_clusters"]}
            params_log.update({c: v for c, v in p.items() if c != "n_clusters"})
            resultados[nombre] = correr_comparacion(
                KMeans(**p), "baseline", params_log,
                run_name or P3_MLFLOW_BASELINE_RUN_NAME, d, git_commit, data_version,
            )
        elif nombre == "search":
            resultados[nombre] = ejecutar_busqueda(
                run_name or P3_MLFLOW_SEARCH_RUN_NAME, d, git_commit, data_version
            )
        else:
            familia, crear_modelo, params_log, run_default = candidatos[nombre]
            resultados[nombre] = correr_candidato(
                crear_modelo, familia, params_log, run_name or run_default,
                d, git_commit, data_version,
            )
    logger.success("Pipeline P3 terminado")
    return resultados


@app.command()
def main(
    model: str = typer.Option("all", help="baseline, kmeans, agglomerative, gmm, search o all."),
    params_json: str = typer.Option("{}", help="Hiperparámetros en JSON (se combinan con config.py)."),
    run_name: str = typer.Option("", help="Nombre del run; vacío usa el de config.py."),
    viajes_path: Path = VIAJES_PARQUET,
    estaciones_path: Path = ESTACIONES_PARQUET,
    experiment_name: str = P3_MLFLOW_EXPERIMENT,
):
    ejecutar(
        model, json.loads(params_json), run_name or None,
        viajes_path, estaciones_path, experiment_name,
    )

if __name__ == "__main__":
    app()
