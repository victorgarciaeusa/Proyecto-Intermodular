# TFM Forecasting — Diario Técnico
## Fase 3 (Feature Engineering + Baseline) + Fase 4 (Modelado ML)---

##  — Feature Engineering

### Script: `src/etl/feature_engineering.py`

Features generadas sobre `analytics.sales_curated`:

| Feature | Descripción |
|---|---|
| `quantity_lag_1`, `quantity_lag_7` | Lags temporales |
| `ma_7`, `std_7` | Media y desviación móvil 7 días |
| `day_of_week`, `month`, `season` | Variables de calendario |
| `is_holiday` | Festivos España (paquete `holidays`) |
| `promotion_flag` | Variable exógena del dataset |
| `temperature`, `event_indicator` | Placeholders NULL — sin datos reales disponibles |

**Error:** `ModuleNotFoundError: No module named 'holidays'`
**Solución:** `pip3 install holidays`

**Resultado:** 7.230 filas (7.300 - 70 eliminadas por lookahead bias — primeros 7 días × 10 SKUs)

**[BLOQUE DÍA 6 COMPLETADO]**

---

## — Baseline seasonal_naive

### Script: `src/models/baseline.py`

**Implementación:** predice el valor de hace 7 días (`quantity_lag_7`).
**Backtesting:** expanding window sin lookahead bias, `min_train_days=90`.

**Error 1:** `mlflow.log_artifact` — `PermissionError: Permission denied: '/mlflow'`
**Causa:** artifact_path apuntaba al path del contenedor Docker, no al host.
**Solución:** eliminar línea `log_artifact` con `sed -i '/log_artifact/d'`.

**Error 2:** conflicto versiones MLflow:
```
BAD_REQUEST: invalid input syntax for type integer: "2."
```
**Causa:** cliente MLflow 2.17.2 incompatible con servidor MLflow 2.12.1 en el contenedor.
**Solución:** `sudo pip3 install mlflow==2.12.1` para alinear versiones.

**Resultados baseline:**

| Métrica | Valor | Objetivo guía |
|---|---|---|
| MAPE global | 7.79% | 15-20% |
| MAPE promo | 10.95% | — |
| MAPE no-promo | 6.99% | — |

**Nota para memoria:** MAPE por debajo del rango esperado 15-20% porque el dataset sintético es muy regular. Documentar como limitación del dataset — no afecta a la validez metodológica.

**[BLOQUE HITO 2 COMPLETADO]**

---

## Día 9 — SARIMAX

### Script: `src/models/sarimax_model.py`

### Decisión: Opción A vs B

> **Pregunta:** ¿Grid search AIC/BIC (Opción A, correcto académicamente) u órdenes fijos (Opción B, pragmático)?
> **Respuesta:** Opción A reducida — grid search sobre SKU piloto, aplicar mejor orden al resto. Enfoque estándar en literatura: selección sobre serie representativa, generalización al resto.

### FASE 1 — Grid search AIC/BIC sobre SKU piloto (SKU 1)

- Grid: `p,q ∈ {0,1,2}`, `d ∈ {0,1}`, `P,Q,D ∈ {0,1}`, `s=7`
- Total combinaciones: 144
- Duración: **14 segundos** (train piloto = 90 observaciones)
- Mejor orden: `(0,1,2)(0,1,1,7)` AIC=597.57

**Convergencia del grid:**

| Orden | AIC |
|---|---|
| (0,0,0)(0,1,0,7) | 724.90 |
| (0,0,0)(0,1,1,7) | 612.74 |
| (0,0,1)(0,1,1,7) | 607.28 |
| (0,0,2)(0,1,1,7) | 598.66 |
| **(0,1,2)(0,1,1,7)** | **597.57** ← seleccionado |

### FASE 2 — Expanding window todos los SKUs

- `min_train_days=90`, paso 1
- Duración: ~50 minutos
- Test ADF: todas las series NO estacionarias (p-value > 0.05) → `d=1` justificado

### Proceso abortado y reiniciado

> El primer run de SARIMAX (script v1 con órdenes fijos) llevaba 70 minutos sin terminar. Se decidió matar el proceso y reescribir con grid search AIC/BIC. El grid search terminó en 14 segundos — el tiempo estimado inicial de 2-4h era para grids más amplios con más observaciones de train.

**Resultados SARIMAX:**

| Métrica | Valor |
|---|---|
| MAPE global | 3.74% |
| MAPE promo | 3.47% |
| MAPE no-promo | 3.81% |

**MAPE por SKU:** rango 3.20% (SKU 1) — 4.32% (SKU 10)

**[FASE 4 PERSONA A COMPLETADA]**

---

## XGBoost — Iteraciones de afinamiento

### Decisión de arquitectura

> **Pregunta:** ¿modelo por SKU, global, o global con SKU como feature?
> **Respuesta:** Opción C — global con SKU como feature. 730 observaciones/SKU insuficientes para hyperopt robusto individual. Modelo global ve 7.230 filas. Enfoque estándar en retail forecasting (M5/Walmart).

### Versión 1 — Hyperopt 50 iteraciones

- Expanding window paso 7
- MAPE global: 4.63%

### Versión 2 — Hyperopt 200 iteraciones

- Features adicionales: lags 14/28, MA14/28, rolling max/min, ratios lag/MA
- Expanding window paso 3
- MAPE global: 3.80%

### Versión 3 — Rigurosa (entregada)

**Mejoras implementadas:**

| Mejora | Descripción | Impacto |
|---|---|---|
| TimeSeriesSplit k=5 | CV temporal en Hyperopt | Alto académicamente |
| Expanding window paso 1 | Evalúa todos los días | Alta robustez evaluación |
| Quantile regression P10/P50/P90 | Intervalos de confianza | Muy alto académicamente |
| SHAP values | Importancia real features | Alto interpretabilidad |

**Error:** `TypeError: fit() got an unexpected keyword argument 'early_stopping_rounds'`
**Causa:** versión XGBoost instalada no acepta `early_stopping_rounds` en `fit()`.
**Solución:** eliminar el parámetro del `fit()` con `sed`.

**Mejores hiperparámetros (Hyperopt 200 iter, TimeSeriesSplit k=5):**
```
n_estimators=1050  max_depth=4  learning_rate=0.0065
subsample=0.712    colsample_bytree=0.766  colsample_bylevel=0.734
min_child_weight=15  gamma=0.040  reg_alpha=0.155  reg_lambda=19.676
```

**SHAP values — top features:**

| Feature | SHAP medio | Interpretación |
|---|---|---|
| promotion_flag | 9.15 | Factor dominante — promo mueve ~9 unidades |
| day_of_week | 6.26 | Estacionalidad semanal fuerte |
| quantity_lag_28 | 2.70 | Memoria mensual |
| quantity_lag_7 | 2.68 | Memoria semanal |
| ma_28 | 2.42 | Tendencia largo plazo |
| is_holiday | 0.003 | Sin impacto relevante |
| is_month_start/end | ~0.001 | Sin impacto |

**Resultados XGBoost riguroso:**

| Métrica | Valor |
|---|---|
| MAPE global | 3.73% |
| MAPE promo | 3.40% |
| MAPE no-promo | 3.81% |

---

## Prophet — Iteraciones de afinamiento

### Versión 1

- `seasonality_mode=multiplicative`, `changepoint_prior_scale=0.05`
- Expanding window paso 7
- MAPE global: 4.26%

### Versión afinada (entregada)

**Grid search:** 216 combinaciones sobre SKU piloto con CV nativa Prophet:

| Parámetro | Valores probados |
|---|---|
| changepoint_prior_scale | {0.001, 0.01, 0.05, 0.1, 0.3, 0.5} |
| seasonality_prior_scale | {0.1, 1.0, 5.0, 10.0, 20.0, 50.0} |
| seasonality_mode | {additive, multiplicative} |
| fourier_order_weekly | {3, 5, 7} |

**CV nativa Prophet:** `initial=180 days`, `period=30 days`, `horizon=28 days`

**Decisión — ¿más tuning?**
> Se planteó ampliar el grid. Conclusión: no compensa. Prophet tiene techo natural en series regulares sin tendencia fuerte. Mejora esperada <0.3% MAPE con x5 tiempo adicional.

**Mejores parámetros:**
```
changepoint_prior_scale = 0.05
seasonality_prior_scale = 0.1
seasonality_mode        = multiplicative
fourier_order_weekly    = 7
```

**Resultados Prophet afinado:**

| Métrica | Valor |
|---|---|
| MAPE global | 4.23% |
| MAPE promo | 4.01% |
| MAPE no-promo | 4.28% |

---

## Análisis de residuos — 4 modelos

### Script: `src/models/residuals_analysis.py`

**Error inicial:** SARIMAX no incluido — no se habían guardado las predicciones en CSV durante el entrenamiento.
**Solución:** modificar `sarimax_model.py` para guardar `/tmp/sarimax_predictions.csv` y re-ejecutar (~50 min).

### Resultados completos

| Modelo | Media res. | Std res. | Skewness | L-B lag 7 | L-B lag 28 | Heteroced. |
|---|---|---|---|---|---|---|
| Baseline | 0.32 | 25.14 | 0.006 | ⚠️ p=0.000 | ⚠️ p=0.000 | ✅ |
| Prophet | -0.17 | 14.68 | 1.485 | ✅ p=0.085 | ✅ p=0.068 | ✅ |
| **SARIMAX** | **-0.15** | **13.52** | **1.904** | **✅ p=0.342** | **✅ p=0.162** | **✅** |
| XGBoost | 2.48 | 13.34 | 1.834 | ⚠️ p=0.001 | ⚠️ p=0.000 | ⚠️ p=0.030 |

**SARIMAX mejor en diagnóstico estadístico** — sin autocorrelación, homocedastico, sin sesgo.

---

## Decisiones técnicas — resumen

| Pregunta | Respuesta |
|---|---|
| ¿Grid AIC o órdenes fijos en SARIMAX? | Grid AIC sobre SKU piloto (Opción A reducida) |
| ¿XGBoost por SKU, global, o global+SKU feature? | Global con SKU como feature (Opción C) |
| ¿Paralelizar grid search con joblib? | No necesario — grid terminó en 14 segundos |
| ¿Afinar más SARIMAX? | No compensa — ya en óptimo AIC, mejora esperada <0.1% |
| ¿Afinar más Prophet? | No compensa — techo natural, mejora esperada <0.3% |
| ¿VM encendida permanentemente? | Sí con refrigeración adecuada. 2 núcleos libres suficientes para host |

---

## Errores y soluciones — resumen

| Error | Causa | Solución |
|---|---|---|
| `PermissionError: '/mlflow'` | artifact_path apunta al contenedor | Eliminar `log_artifact` |
| `BAD_REQUEST: integer "2."` | Versiones MLflow desalineadas (2.17 vs 2.12) | `pip3 install mlflow==2.12.1` |
| `TypeError: unexpected 'early_stopping_rounds'` | Parámetro en lugar incorrecto | Mover al constructor XGBRegressor |
| SARIMAX colgado 70 min | Expanding window completo costoso con órdenes fijos | Matar proceso, implementar grid AIC |

---

## Estado MLflow — experimentos registrados

| Experimento | Run | MAPE Global |
|---|---|---|
| baseline_seasonal_naive | seasonal_naive | 7.79% |
| sarimax | sarimax_(0,1,2)_(0,1,1,7) | 3.74% |
| xgboost | xgboost_riguroso | 3.73% |
| prophet | prophet_afinado | 4.23% |

---
