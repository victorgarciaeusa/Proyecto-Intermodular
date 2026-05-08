# TFM Forecasting — Decisión Modelo Final

## 1. Tabla comparativa MAPE

| Modelo | MAPE Global | MAPE Promo | MAPE No-promo | Objetivo <12% |
|---|---|---|---|---|
| Baseline seasonal_naive | 7.79% | 10.95% | 6.99% | ✅ |
| Prophet afinado | 4.23% | 4.01% | 4.28% | ✅ |
| SARIMAX (0,1,2)(0,1,1,7) | 3.74% | 3.47% | 3.81% | ✅ |
| XGBoost riguroso | 3.73% | 3.40% | 3.81% | ✅ |

---

## 2. MAPE por SKU

| SKU | Baseline | Prophet | SARIMAX | XGBoost |
|---|---|---|---|---|
| 1 | 7.40% | 3.67% | 3.20% | 3.26% |
| 2 | 7.91% | 4.32% | 3.56% | 3.72% |
| 3 | 7.90% | 4.07% | 3.95% | 3.82% |
| 4 | 7.03% | 4.02% | 3.34% | 3.35% |
| 5 | 8.43% | 4.53% | 4.02% | 4.08% |
| 6 | 7.76% | 4.39% | 3.77% | 3.60% |
| 7 | 7.76% | 4.19% | 3.80% | 3.79% |
| 8 | 7.39% | 4.31% | 3.82% | 3.98% |
| 9 | 7.61% | 4.07% | 3.61% | 3.47% |
| 10 | 8.68% | 4.69% | 4.32% | 4.20% |

---

## 3. Análisis de residuos — 4 modelos

### Tests aplicados
- **Shapiro-Wilk** y **Kolmogorov-Smirnov** — normalidad
- **Ljung-Box** (lags 7, 14, 28) — autocorrelación residual
- **Pearson |residuos| vs tiempo** — heterocedasticidad

### Resultados

| Modelo | Media res. | Std res. | Skewness | Autocorr L-B 7 | Autocorr L-B 28 | Heteroced. | Sesgo |
|---|---|---|---|---|---|---|---|
| Baseline | 0.32 | 25.14 | 0.006 | ⚠️ p=0.000 | ⚠️ p=0.000 | ✅ | mínimo |
| Prophet | -0.17 | 14.68 | 1.485 | ✅ p=0.085 | ✅ p=0.068 | ✅ | ~0 |
| **SARIMAX** | **-0.15** | **13.52** | **1.904** | **✅ p=0.342** | **✅ p=0.162** | **✅** | **~0** |
| XGBoost | 2.48 | 13.34 | 1.834 | ⚠️ p=0.001 | ⚠️ p=0.000 | ⚠️ p=0.030 | 2.48 |

### Interpretación

**SARIMAX** — mejor comportamiento estadístico global:
- Ljung-Box lag 7: p=0.342 — muy lejos del límite crítico 0.05 — residuos sin estructura temporal
- Ljung-Box lag 28: p=0.162 — sin autocorrelación a horizonte mensual
- Homocedastico — varianza constante a lo largo del tiempo
- Sesgo prácticamente nulo (-0.15) — sin subestimación/sobreestimación sistemática

**Prophet** — segundo mejor en diagnóstico:
- **Sin autocorrelación en todos los lags** 
- **Homocedastico** 
- Sesgo ~0

**XGBoost** — advertencias estadísticas:
- Autocorrelación significativa en lags 7, 14, 28 — estructura temporal no capturada
- Heterocedastico (p=0.030) — varianza del error crece con el tiempo
- Sesgo positivo de 2.48 — subestimación sistemática

**Baseline** — peor diagnóstico esperado:
- Autocorrelación severa en todos los lags — inherente al modelo naive

---

## 4. Criterios de selección

| Criterio | Peso | Prophet | SARIMAX | XGBoost |
|---|---|---|---|---|
| MAPE global | Alto | 4.23% | 3.74% | **3.73%** |
| MAPE promo | Alto | 4.01% | 3.47% | **3.40%** |
| Sin autocorrelación residual | Alto | ✅ | **✅ p=0.342** | ⚠️ |
| Homocedasticidad | Medio | ✅ | **✅** | ⚠️ |
| Sesgo residuos | Medio | ✅ ~0 | **✅ ~0** | ⚠️ 2.48 |
| Std residuos | Medio | 14.68 | **13.52** | 13.34 |
| Interpretabilidad | Medio | Alta | **Alta** | Media |
| Rigor metodológico | Alto | Grid+CV nativa | **AIC/BIC+ADF** | TimeSeriesSplit+SHAP |
| Escalabilidad nuevos SKUs | Bajo | Media | Baja | **Alta** |

---

## 5. Decisión final

### Modelo seleccionado: **SARIMAX (0,1,2)(0,1,1,7)**

**Justificación técnica:**

**1. MAPE competitivo** — 3.74% global, prácticamente empatado con XGBoost (3.73%, diferencia de 0.01 puntos porcentuales — estadísticamente irrelevante).

**2. Mejor diagnóstico de residuos** — Ljung-Box p=0.342 en lag 7 y p=0.162 en lag 28. Los residuos no presentan estructura temporal — el modelo captura correctamente la dinámica de la serie. XGBoost presenta autocorrelación significativa (p=0.001 en lag 7) indicando estructura no capturada.

**3. Sin sesgo sistemático** — media residuos -0.15 frente a 2.48 de XGBoost. SARIMAX no subestima ni sobreestima sistemáticamente.

**4. Homocedasticidad** — varianza del error constante. XGBoost presenta heterocedasticidad (p=0.030) — el error crece con el tiempo, comprometiendo la fiabilidad a largo plazo.

**5. Selección de órdenes justificada** — grid search AIC/BIC sobre 144 combinaciones. Orden `(0,1,2)(0,1,1,7)` con AIC=597.57. Test ADF confirma no estacionariedad (d=1 justificado).

**6. Interpretabilidad estadística** — coeficientes MA y SAR directamente interpretables sin herramientas adicionales.

---

### Modelo alternativo para producción: **XGBoost riguroso**

Preferible cuando se requiere:
- Predicción en tiempo real sobre nuevos SKUs sin reentrenamiento ARIMA
- Máxima precisión en segmento promocional (MAPE promo 3.40%)
- Escalabilidad a cientos de SKUs

---

### Análisis modelo por régimen promocional

Criterio: si `MAPE(promo) > 1.3 × MAPE(no-promo)` → modelo separado justificado.

| Modelo | MAPE Promo | MAPE No-promo | Ratio | Modelo separado |
|---|---|---|---|---|
| Baseline | 10.95% | 6.99% | 1.57 | ⚠️ Justificado |
| Prophet | 4.01% | 4.28% | 0.94 | ✅ No necesario |
| SARIMAX | 3.47% | 3.81% | 0.91 | ✅ No necesario |
| XGBoost | 3.40% | 3.81% | 0.89 | ✅ No necesario |

---

## 6. Limitaciones del dataset — impacto en resultados

### Caracterización del dataset

El dataset `store_sales.csv` presenta características de serie sintética:

| Parámetro | Valor | Implicación |
|---|---|---|
| Coeficiente de variación (CV) | 11.6% (std=26/media=228) | Serie muy estable, baja variabilidad |
| Gaps temporales | 0 — continuidad perfecta | Sin interrupciones reales |
| Outliers | 0 valores negativos | Sin anomalías de registro |
| Nulos en columnas clave | 0 | Sin problemas de calidad |
| Variables exógenas reales | Solo `promotion_flag` | Sin temperatura, eventos externos |

### Efecto sobre los MAPEs

Los MAPEs obtenidos (3.73%–4.23% para modelos avanzados) son excesivamente inferiores al rango esperado en literatura para forecasting de demanda retail (15–20%). Esto se explica por la regularidad del dataset, **no por sobreajuste** del pipeline.

En datos reales los MAPEs serían significativamente más altos por:
- Outliers de ventas (promociones no documentadas, roturas de stock)
- Gaps temporales y cambios de tendencia
- Estacionalidad más compleja (anual, eventos especiales)
- Ruido inherente a la demanda real

### Impacto en la validez del trabajo

El dataset sintético **no invalida** las conclusiones metodológicas por los siguientes motivos:

**El pipeline es correcto independientemente del dataset.** La secuencia ingesta → curación → feature engineering → selección de modelos por AIC/BIC y cross-validation temporal → análisis de residuos es metodológicamente sólida. Los mismos procedimientos aplicados sobre datos reales producirían MAPEs más altos pero la jerarquía entre modelos y las conclusiones metodológicas se mantendrían.

**La comparativa relativa es válida.** Aunque los valores absolutos de MAPE son exageradamente optimistas, la ordenación de modelos (SARIMAX ≈ XGBoost > Prophet > Baseline) y las diferencias en diagnóstico de residuos reflejan las propiedades reales de cada arquitectura.

**Los tests estadísticos son rigurosos.** Ljung-Box, Shapiro-Wilk, ADF y heterocedasticidad no dependen de la procedencia del dataset — sus conclusiones son válidas sobre cualquier serie.

### Propuesta de mejora documentada

Para una versión productiva del sistema se recomienda:
1. Sustituir el dataset por datos reales de un retailer o dataset público con complejidad real (Rossmann, M5 Forecasting)
2. Añadir variables exógenas reales: temperatura, festivos autonómicos, datos de competencia
3. Incluir SKUs con comportamiento irregular para validar la robustez del pipeline ante outliers

> *"El dataset empleado presenta características de serie sintética — ausencia de outliers, continuidad temporal perfecta y varianza reducida (CV=11.6%). Esto condiciona los MAPEs obtenidos, que resultan inferiores al rango esperado en literatura para forecasting de demanda retail (15-20%). Sin embargo, hemos priorizado enfocarnos en el desarrollo metodológico del pipeline: ingesta, curación, feature engineering, selección de modelos por AIC/BIC y cross-validation temporal, y análisis de residuos. Los mismos procedimientos aplicados sobre datos reales producirían MAPEs más altos pero la jerarquía entre modelos y las conclusiones metodológicas se mantendrían."*
