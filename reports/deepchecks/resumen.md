# Resumen de validación de datos (DeepChecks)

Generado con `python -m mibici.validacion_datos`.

## Chequeos de dominio (tabla completa)

| Conjunto | Chequeo | Valor | Resultado | Detalle |
|---|---|---:|---|---|
| viajes | filas | 4,532,032 | ✅ |  |
| viajes | valores nulos | 0 | ✅ |  |
| viajes | viaje_id duplicado | 0 | ✅ |  |
| viajes | viajes duplicados (mismos datos, distinto id) | 1,076 | ❌ | Mismo origen, destino, inicio y fin al segundo; representan < 0.1 % del total. |
| viajes | duración negativa | 0 | ✅ |  |
| viajes | duración < 60 s | 117,226 | ✅ | Desanclajes fallidos: los pipelines de P1-P3 los descartan. |
| viajes | duración > 24 h | 6 | ✅ | Bicicletas no ancladas: los pipelines de P1-P3 los descartan. |
| viajes | estaciones fuera del catálogo | 0 | ✅ |  |
| estaciones | id duplicado | 0 | ✅ |  |
| estaciones | valores nulos | 0 | ✅ |  |
| estaciones | coordenadas fuera de Guadalajara | 0 | ✅ |  |
| estaciones | capacidad <= 0 | 4 | ✅ | P2 las excluye (no se puede calcular el umbral de riesgo). ids: 99, 109, 116, 138 |

## Chequeos de DeepChecks

| Conjunto | Suite | Chequeo | Estado | Detalle |
|---|---|---|---|---|
| viajes | data_integrity | Single Value in Column | ✅ pasa |  |
| viajes | data_integrity | Special Characters | ✅ pasa |  |
| viajes | data_integrity | Mixed Nulls **(crítico)** | ✅ pasa |  |
| viajes | data_integrity | Mixed Data Types **(crítico)** | ✅ pasa |  |
| viajes | data_integrity | String Mismatch | ✅ pasa |  |
| viajes | data_integrity | Data Duplicates | ✅ pasa |  |
| viajes | data_integrity | String Length Out Of Bounds | ✅ pasa |  |
| viajes | data_integrity | Feature-Feature Correlation | ✅ pasa |  |
| viajes | data_integrity | PercentOfNulls **(crítico)** | ✅ pasa |  |
| viajes | data_integrity | Outlier Sample Detection | ℹ️ sin_condicion |  |
| viajes | data_integrity | Conflicting Labels - Train Dataset **(crítico)** | ➖ no_aplica | Dataset does not contain a label column |
| viajes | data_integrity | Feature Label Correlation - Train Dataset | ➖ no_aplica | Dataset does not contain a label column |
| viajes | data_integrity | Identifier Label Correlation - Train Dataset | ➖ no_aplica | Dataset does not contain a label column |
| estaciones | data_integrity | Single Value in Column | ✅ pasa |  |
| estaciones | data_integrity | Special Characters | ✅ pasa |  |
| estaciones | data_integrity | Mixed Nulls **(crítico)** | ✅ pasa |  |
| estaciones | data_integrity | Mixed Data Types **(crítico)** | ✅ pasa |  |
| estaciones | data_integrity | String Mismatch | ✅ pasa |  |
| estaciones | data_integrity | Data Duplicates | ✅ pasa |  |
| estaciones | data_integrity | String Length Out Of Bounds | ✅ pasa |  |
| estaciones | data_integrity | Feature-Feature Correlation | ✅ pasa |  |
| estaciones | data_integrity | PercentOfNulls **(crítico)** | ✅ pasa |  |
| estaciones | data_integrity | Outlier Sample Detection | ℹ️ sin_condicion |  |
| estaciones | data_integrity | Conflicting Labels - Train Dataset **(crítico)** | ➖ no_aplica | Dataset does not contain a label column |
| estaciones | data_integrity | Feature Label Correlation - Train Dataset | ➖ no_aplica | Dataset does not contain a label column |
| estaciones | data_integrity | Identifier Label Correlation - Train Dataset | ➖ no_aplica | Dataset does not contain a label column |
| p1 | data_integrity | Feature-Feature Correlation | ❌ falla | Not more than 0 pairs are correlated above 0.9: Correlation is greater than 0.9 for pairs [('lag_1', 'media_7d'), ('lag_1', 'media_28d'), ('lag_2', 'media_7d'), ('lag_2', 'media_28d'), ('lag_7', 'media_7d'), ('lag_7', 'media_28d'), ('lag_7', 'media_mismo_dia_4sem'), ('lag_14', 'lag_7'), ('lag_14', 'media_7d'), ('lag_14', 'media_28d'), ('lag_14', 'media_mismo_dia_4sem'), ('media_7d', 'media_mismo_dia_4sem'), ('media_28d', 'media_7d'), ('media_28d', 'media_mismo_dia_4sem')] |
| p1 | data_integrity | Single Value in Column | ✅ pasa |  |
| p1 | data_integrity | Special Characters | ✅ pasa |  |
| p1 | data_integrity | Mixed Nulls **(crítico)** | ✅ pasa |  |
| p1 | data_integrity | Mixed Data Types **(crítico)** | ✅ pasa |  |
| p1 | data_integrity | String Mismatch | ✅ pasa |  |
| p1 | data_integrity | Data Duplicates | ✅ pasa |  |
| p1 | data_integrity | String Length Out Of Bounds | ✅ pasa |  |
| p1 | data_integrity | Feature Label Correlation | ✅ pasa |  |
| p1 | data_integrity | Identifier Label Correlation | ✅ pasa |  |
| p1 | data_integrity | PercentOfNulls **(crítico)** | ✅ pasa |  |
| p1 | data_integrity | Outlier Sample Detection | ℹ️ sin_condicion |  |
| p1 | data_integrity | Conflicting Labels - Train Dataset **(crítico)** | ➖ no_aplica | Check is irrelevant for regression tasks |
| p1 | train_test_validation | Feature Drift | ❌ falla | categorical drift score < 0.2 and numerical drift score < 0.2: Failed for 1 out of 12 columns. Found 1 numeric columns with Kolmogorov-Smirnov above threshold: {'dias_activa': '0.99'} |
| p1 | train_test_validation | Multivariate Drift | ❌ falla | Drift value is less than 0.25: Found drift value of: 1, corresponding to a domain classifier AUC of: 1 |
| p1 | train_test_validation | Datasets Size Comparison | ✅ pasa |  |
| p1 | train_test_validation | New Category Train Test | ✅ pasa |  |
| p1 | train_test_validation | String Mismatch Comparison | ✅ pasa |  |
| p1 | train_test_validation | Date Train-Test Leakage (duplicates) **(crítico)** | ✅ pasa |  |
| p1 | train_test_validation | Date Train-Test Leakage (overlap) **(crítico)** | ✅ pasa |  |
| p1 | train_test_validation | Train Test Samples Mix | ✅ pasa |  |
| p1 | train_test_validation | Feature Label Correlation Change | ✅ pasa |  |
| p1 | train_test_validation | Label Drift | ✅ pasa |  |
| p1 | train_test_validation | New Label Train Test | ➖ no_aplica | Check is irrelevant for regression tasks |
| p1 | train_test_validation | Index Train Test Leakage | ➖ no_aplica | Dataset does not contain an index |
| p2 | data_integrity | Feature-Feature Correlation | ❌ falla | Not more than 0 pairs are correlated above 0.9: Correlation is greater than 0.9 for pairs [('neto_3h', 'neto_3h_rel')] |
| p2 | data_integrity | Single Value in Column | ✅ pasa |  |
| p2 | data_integrity | Special Characters | ✅ pasa |  |
| p2 | data_integrity | Mixed Nulls **(crítico)** | ✅ pasa |  |
| p2 | data_integrity | Mixed Data Types **(crítico)** | ✅ pasa |  |
| p2 | data_integrity | String Mismatch | ✅ pasa |  |
| p2 | data_integrity | Data Duplicates | ✅ pasa |  |
| p2 | data_integrity | String Length Out Of Bounds | ✅ pasa |  |
| p2 | data_integrity | Conflicting Labels **(crítico)** | ✅ pasa |  |
| p2 | data_integrity | Feature Label Correlation | ✅ pasa |  |
| p2 | data_integrity | Identifier Label Correlation | ✅ pasa |  |
| p2 | data_integrity | PercentOfNulls **(crítico)** | ✅ pasa |  |
| p2 | data_integrity | Outlier Sample Detection | ℹ️ sin_condicion |  |
| p2 | train_test_validation | Train Test Samples Mix | ❌ falla | Percentage of test data samples that appear in train data is less or equal to 5%: Percent of test data samples that appear in train data: 11.79% |
| p2 | train_test_validation | Datasets Size Comparison | ✅ pasa |  |
| p2 | train_test_validation | New Label Train Test | ✅ pasa |  |
| p2 | train_test_validation | New Category Train Test | ✅ pasa |  |
| p2 | train_test_validation | String Mismatch Comparison | ✅ pasa |  |
| p2 | train_test_validation | Date Train-Test Leakage (duplicates) **(crítico)** | ✅ pasa |  |
| p2 | train_test_validation | Date Train-Test Leakage (overlap) **(crítico)** | ✅ pasa |  |
| p2 | train_test_validation | Feature Label Correlation Change | ✅ pasa |  |
| p2 | train_test_validation | Feature Drift | ✅ pasa |  |
| p2 | train_test_validation | Label Drift | ✅ pasa |  |
| p2 | train_test_validation | Multivariate Drift | ✅ pasa |  |
| p2 | train_test_validation | Index Train Test Leakage | ➖ no_aplica | Dataset does not contain an index |
| p3 | data_integrity | Feature-Feature Correlation | ❌ falla | Not more than 0 pairs are correlated above 0.9: Correlation is greater than 0.9 for pairs [('llegadas_dia', 'salidas_dia')] |
| p3 | data_integrity | Single Value in Column | ✅ pasa |  |
| p3 | data_integrity | Special Characters | ✅ pasa |  |
| p3 | data_integrity | Mixed Nulls **(crítico)** | ✅ pasa |  |
| p3 | data_integrity | Mixed Data Types **(crítico)** | ✅ pasa |  |
| p3 | data_integrity | String Mismatch | ✅ pasa |  |
| p3 | data_integrity | Data Duplicates | ✅ pasa |  |
| p3 | data_integrity | String Length Out Of Bounds | ✅ pasa |  |
| p3 | data_integrity | PercentOfNulls **(crítico)** | ✅ pasa |  |
| p3 | data_integrity | Outlier Sample Detection | ℹ️ sin_condicion |  |
| p3 | data_integrity | Conflicting Labels - Train Dataset **(crítico)** | ➖ no_aplica | Dataset does not contain a label column |
| p3 | data_integrity | Feature Label Correlation - Train Dataset | ➖ no_aplica | Dataset does not contain a label column |
| p3 | data_integrity | Identifier Label Correlation - Train Dataset | ➖ no_aplica | Dataset does not contain a label column |
| p3 | train_test_validation | Datasets Size Comparison | ✅ pasa |  |
| p3 | train_test_validation | New Category Train Test | ✅ pasa |  |
| p3 | train_test_validation | String Mismatch Comparison | ✅ pasa |  |
| p3 | train_test_validation | Train Test Samples Mix | ✅ pasa |  |
| p3 | train_test_validation | Feature Drift | ✅ pasa |  |
| p3 | train_test_validation | Multivariate Drift | ✅ pasa |  |
| p3 | train_test_validation | New Label Train Test | ➖ no_aplica | Dataset does not contain a label column |
| p3 | train_test_validation | Date Train Test Leakage Duplicates **(crítico)** | ➖ no_aplica | Dataset does not contain a datetime |
| p3 | train_test_validation | Date Train Test Leakage Overlap **(crítico)** | ➖ no_aplica | Dataset does not contain a datetime |
| p3 | train_test_validation | Index Train Test Leakage | ➖ no_aplica | Dataset does not contain an index |
| p3 | train_test_validation | Feature Label Correlation Change | ➖ no_aplica | Dataset does not contain a label column |
| p3 | train_test_validation | Label Drift | ➖ no_aplica | Dataset does not contain a label column |

**Fallas críticas: 0**
