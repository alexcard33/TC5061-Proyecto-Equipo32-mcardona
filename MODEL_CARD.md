# Model Card: `mibici-equipo32-p1-demanda` (versión 3, `@champion`)

Ficha del modelo que sirve la API del proyecto MiBici. Sigue la estructura propuesta por
Mitchell et al. (2019), *Model Cards for Model Reporting*, adaptada al ciclo de MLOps del curso.
Las cifras salen de los runs de MLflow y de los reportes versionados en `reports/`; cada sección
indica de dónde.

| | |
|---|---|
| Registered model | `mibici-equipo32-p1-demanda` |
| Versión en producción | 3 (alias `champion`, `estado_produccion = produccion`) |
| Versión en evaluación | 1 (alias `challenger`, `estado_produccion = en_evaluacion`) |
| Corrida de origen | `p1_v2_reentrenado` (`e40c098a073f422db3fb7c3193db9fbe`), experimento `mibici-equipo32` |
| Commit del código | `01c6d6cbd068` |
| Versión de datos | `7aa5e436d887+8460b856c7e8` (manifiestos de 2025 y 2026) |
| Registrado | 15 de noviembre de 2026 |
| Responsables | Equipo 32, TC5061 Operaciones de aprendizaje automático, Tecnológico de Monterrey |
| Última revisión de esta ficha | 27 de noviembre de 2026 |

## 1. Detalles del modelo

- **Tarea:** regresión de conteos. Predice cuántos viajes se iniciarán en una estación de MiBici
  (bicicleta pública del Área Metropolitana de Guadalajara) el día siguiente.
- **Algoritmo:** `PoissonRegressor` de scikit-learn con `alpha = 0.001`. Los conteos (rezagos y
  medias móviles) entran como `log1p` y todas las variables se estandarizan dentro del mismo
  pipeline, así que el modelo registrado recibe los valores crudos.
- **Por qué este modelo:** en la selección original (`p1_regresion.py`) se compararon un baseline
  ingenuo semanal, este modelo lineal y un `HistGradientBoostingRegressor` con pérdida Poisson.
  El lineal tuvo el menor MAE de test (5.41 contra 5.90 del HGB, versiones 1 y 2 del registry),
  es más fácil de explicar al operador y es barato de servir.
- **Formato:** modelo `pyfunc` de MLflow serializado con `skops` (no usa `pickle`).
- **Granularidad y horizonte:** estación × día, horizonte de 1 día. La predicción del día D se
  emite al cierre del día D-1.

## 2. Uso previsto

**Usuarios:** el operador del sistema MiBici, en particular quien planea la flota y los turnos de
rebalanceo.

**Uso previsto:**

- Estimar la demanda de salidas de cada estación para el día siguiente y, sumando estaciones, la
  de una zona o de todo el sistema.
- Priorizar a qué estaciones llevar bicicletas antes de que empiece el día.
- Servir de insumo para una persona que toma la decisión. El modelo no mueve camiones ni cambia
  la operación por sí solo.

**Fuera de alcance (no usar para):**

- Pronósticos por hora o con horizonte mayor a un día. El modelo no se entrenó para eso.
- Estaciones con menos de 14 días de historia: no tienen los rezagos que necesita la entrada.
- Decisiones sobre personas usuarias, como tarifas o sanciones. El modelo no ve datos de personas
  y no debe usarse para eso.
- Días con eventos extraordinarios (cierres viales, marchas, conciertos, lluvias fuertes). El
  modelo no tiene variables de clima ni de eventos.
- Otros sistemas de bicicleta compartida sin reentrenar con sus propios datos.

## 3. Datos

**Fuente:** [datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/), viajes y
nomenclatura de estaciones. La ingesta está en `mibici/dataset.py` y cada corrida queda
identificada por el `data_version` de `data/manifest.json` y `data/manifest_actual.json`.

| Conjunto | Periodo | Tamaño | Uso |
|---|---|---|---|
| Entrenamiento | ene 2025 – jun 2026 | 191,455 filas estación-día, 388 estaciones | Ajuste de la versión 3 |
| Validación | jul 2026 | — | Selección y control de sobreajuste |
| Holdout del A/B | ago – oct 2026 | 42,520 filas | Comparación contra la versión 1 |
| Referencia de drift | ene – sep 2025 | 90,230 filas, 369 estaciones | Base del monitoreo |

**Variables de entrada** (12, construidas en `mibici/panel_p1.py` y `mibici/features.py`):

| Grupo | Variables |
|---|---|
| Calendario del día objetivo | `dia_semana`, `fin_semana`, `feriado` |
| Rezagos | `lag_1`, `lag_2`, `lag_7`, `lag_14` |
| Medias que terminan en D-1 | `media_7d`, `media_28d`, `media_mismo_dia_4sem` |
| Estación | `dias_activa`, `capacidad` |

**Prevención de fuga de información:** todas las variables usan solo datos hasta D-1.
`p1_agregar_features` lanza `LeakageError` si se pide un rezago menor que el horizonte, y las
particiones son temporales, sin mezclar fechas.

**Privacidad y minimización:** antes de guardar los Parquet, la ingesta elimina el pseudónimo de
la persona usuaria, el género y el año de nacimiento. El modelo trabaja con conteos agregados por
estación y día, así que ni la entrada ni la salida contienen datos personales. Los datos crudos
no se versionan en Git (`.gitignore`).

**Calidad:** las validaciones de DeepChecks (`reports/deepchecks/`) y la revisión del drift no
encontraron nulos ni columnas vacías. Las 155 filas duplicadas del periodo actual son días sin
viajes de estaciones distintas con las mismas variables, no errores de captura.

## 4. Evaluación

**Validación (jul 2026), run `p1_v2_reentrenado`:** MAE 5.563, RMSE 9.938, WAPE 19.3%, R² 0.901.

**A/B testing pareado sobre el holdout ago–oct 2026** (`reports/ab_test/`, run `p1_ab_test`):

| Métrica | Versión 1 | **Versión 3 (producción)** | Baseline (lag 7 días) |
|---|---|---|---|
| MAE | 6.420 | **6.071** | 7.552 |
| RMSE | 11.817 | **11.296** | 14.498 |
| WAPE | 18.8% | **17.8%** | 22.2% |
| R² | 0.911 | **0.919** | 0.866 |
| Sesgo (viajes/día) | -2.77 | **-1.22** | -1.50 |
| Error del total diario | 10.1% | **7.8%** | 10.4% |

La diferencia de MAE (versión 3 menos versión 1) es -0.349 (-5.4%), con un intervalo de confianza
al 95% de [-0.506, -0.201]. Se calculó con un bootstrap por días completos, porque los errores de
un mismo día están correlacionados. La prueba de Wilcoxon da p = 4.2×10⁻⁵ y la versión 3 gana 62
de 92 días. La regla de promoción se fijó antes de correr la prueba: el challenger solo se
promueve si todo el intervalo queda debajo de cero.

**Resultados por segmento** (MAE en el holdout):

| Segmento | Filas | Versión 1 | Versión 3 | Baseline |
|---|---|---|---|---|
| Estaciones que ya existían en 2025 | 33,948 | 7.213 | **6.788** | 8.475 |
| Estaciones nuevas de 2026 | 8,572 | 3.282 | **3.231** | 3.897 |

El MAE absoluto de las estaciones nuevas es menor porque tienen menos demanda (unos 10.7 viajes
al día en promedio contra unos 30 de las existentes). En términos relativos su error es mayor:
alrededor de 30% de su demanda. Ver la sección 7.

## 5. Monitoreo

- **Drift de datos:** Evidently compara la ventana de entrenamiento con los mismos meses del año
  siguiente (`mibici/monitoreo_drift.py`, `reports/drift/`). En la última revisión (ene–sep 2026)
  hubo drift en 9 de 13 variables. La causa principal es la expansión de la red, con 103
  estaciones nuevas. En las 369 estaciones que ya existían, ninguna variable de demanda pasa el
  umbral.
- **Desempeño:** el A/B y el MAE mensual por segmento (`p1_ab_metricas.csv`) son la referencia
  para detectar degradación.
- **Servicio:** cada push corre en CI una prueba de humo y una prueba de carga corta con Locust
  sobre el sistema levantado con Compose. El pipeline se detiene si hay errores o si el p95 pasa
  de 500 ms.
- **Política de reentrenamiento recomendada:** mensual, con ventana creciente. Se adelanta si las
  variables de demanda de las estaciones existentes tienen drift o si el MAE del mes se aleja del
  de validación. Toda versión nueva entra como `challenger` y solo pasa a `champion` con la misma
  regla del A/B.

## 6. Operación

- **Servicio:** API REST con FastAPI (`api/`) en un contenedor sin privilegios. Endpoints:
  `/health`, `/model-info` y `/predict`.
- **Validación de entrada:** Pydantic rechaza valores fuera de rango (días de la semana de 0 a 6,
  conteos entre 0 y 5,000, capacidad entre 1 y 200) y lotes de más de 1,000 estaciones.
- **Carga del modelo:** por defecto, el modelo exportado del registry viene empaquetado en la
  imagen (`MODEL_SOURCE=local`). Con `MODEL_SOURCE=registry` se descarga del MLflow de Compose
  por nombre y alias.
- **Desempeño medido** (`reports/carga/`): con 1 worker, p95 de 72 ms a 10 usuarios y de 89 ms a
  25. Con 2 workers se sostiene el SLO de p95 ≤ 200 ms hasta 100 usuarios. No hubo errores en
  ninguna corrida.
- **Costo** (`finops/costo_inferencia.py`, `reports/finops/`): 0.074 USD por millón de
  inferencias con 2 workers en EC2 c7i.large y capacidad al 100%. Con la demanda real (un
  tablero que se refresca cada 15 minutos), el costo lo domina la instancia encendida sin uso, así
  que la recomendación es apagarla fuera de horario o predecir por lotes.

## 7. Limitaciones y riesgos conocidos

- **`dias_activa` está mal definida.** Cuenta los días desde la primera salida de la estación
  dentro del panel, que empieza el 1 de enero de 2025, no desde su apertura real. En 2026 toma
  valores que el modelo nunca vio (hasta 637 contra 272 en el entrenamiento original) y el modelo
  lineal extrapola. Corregirla con la fecha de apertura de la nomenclatura es la primera mejora
  pendiente.
- **Subestima los crecimientos rápidos.** Ninguna versión anticipó el aumento de septiembre y
  octubre de 2026 (octubre tuvo 48% más viajes que el año anterior). La versión 3 lo reduce
  (sesgo de -1.22 contra -2.77), pero sigue siendo negativo.
- **Mayor error relativo en estaciones nuevas.** Las estaciones de 2026, en su mayoría en zonas de
  expansión, tienen menos historia y menos demanda, y su error relativo es mayor. Si las
  predicciones se usaran sin revisión para repartir bicicletas, esas zonas podrían recibir menos
  servicio. Por eso el modelo es un insumo para la persona que planea y su desempeño se reporta
  por segmento.
- **Estaciones fuera de la nomenclatura.** Las estaciones 240, 405 y 407 no vienen en la
  nomenclatura de octubre de 2026 y quedan con capacidad 0. La API rechaza capacidad 0, así que
  esas estaciones no reciben predicción hasta corregir el dato.
- **Feriados.** La lista `P1_FERIADOS` de `config.py` se mantiene a mano. Hay que agregar los de
  cada año nuevo; sin ellos, `feriado` vale 0 en todos los feriados.
- **Sin clima ni eventos.** Los días atípicos se van a predecir como un día normal.

## 8. Gobernanza

- **Ciclo de vida por aliases:** `champion` (producción), `challenger` (en evaluación) y sin
  alias (retirado). Cada versión lleva `estado_produccion`, su linaje (`source_run_id`,
  `git_commit`, `data_version`, `run_stage_of_work`) y, si participó en el A/B, su resultado
  (`ab_test_rol`, `ab_test_resultado`). `register` y `sync_registry` mantienen estos tags
  coherentes.
- **Reproducibilidad:** el entrenamiento corre como MLflow Project (`mlflow run . -e retrain`) con
  versiones fijas (`python_env.yaml`, `requirements.txt`), semilla fija y datos identificados por
  `data_version`. `tests/test_reproducibilidad.py` lo verifica.
- **Promoción y reversión:** promover o revertir es mover el alias `champion`, sin tocar código ni
  borrar versiones. Para volver a la versión 1:

  ```python
  from mlflow import MlflowClient
  MlflowClient("sqlite:///mlflow.db").set_registered_model_alias(
      "mibici-equipo32-p1-demanda", "champion", "1"
  )
  ```

  Después se corre `python -m mibici.modeling.sync_registry` para actualizar estados, y
  `python -m mibici.modeling.export_model` (o `MODEL_SOURCE=registry`) para que la API lo sirva.
- **Calidad y seguridad:** `pytest` (unitarias, reproducibilidad y API), DeepChecks para los
  datos, `bandit` y `pip-audit` para el código y las dependencias (`reports/security/`), búsqueda
  de secretos en el historial de Git y `ruff` en CI.
- **Trazabilidad del servicio:** `/model-info` regresa nombre, versión, alias, run de origen,
  fuente del modelo y la ubicación de esta ficha.

## 9. Otros modelos registrados

Estos modelos están en el registry con su propio linaje, pero la API no los sirve:

| Registered model | Tarea | `champion` | Métrica de test |
|---|---|---|---|
| `mibici-equipo32-p2-riesgo` | Clasificación: riesgo de quedarse sin bicicletas o sin anclajes en las próximas 2 h | v1, `p2_candidato_hgb_tuneado` | PR-AUC 0.400 |
| `mibici-equipo32-p3-grupos` | Clustering de estaciones por patrón de uso | v1, `p3_kmeans_final` | Silhouette 0.167 |

## Referencias

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E.,
  Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the
  Conference on Fairness, Accountability, and Transparency*, 220–229.
- Reportes del proyecto: `reports/ab_test/`, `reports/drift/`, `reports/carga/`,
  `reports/finops/`, `reports/security/` y `reports/deepchecks/`.
