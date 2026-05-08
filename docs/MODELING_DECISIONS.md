# MODELING_DECISIONS.md
## Registro de Decisiones Técnicas de Modelado — TFM Forecasting

**Versión**: 1.0  
**Última actualización**: 27 Abril 2026  

## 1. Definición del Problema

### 1.1 Formulación Formal

**Variable objetivo**: $y_t^{(s)}$ — demanda (cantidad vendida) del SKU $s$ en el período $t$.

**Horizonte de forecast**: $H = 4$ semanas (28 días).

**Problema**: Dado el histórico $\{y_1^{(s)}, y_2^{(s)}, \ldots, y_t^{(s)}\}$ y el vector de covariables $X_t$, estimar:

$$\hat{y}_{t+h}^{(s)} = f(y_{1:t}^{(s)}, X_{1:t+h}) \quad \forall h \in \{1, \ldots, 28\}$$

### 1.2 Decisiones de Alcance

**DECISIÓN 1.A — Granularidad temporal: diaria**

*Opciones evaluadas*: diaria, semanal, mensual.

*Justificación*: La granularidad diaria preserva señales relevantes de comportamiento (picos de fin de semana, eventos puntuales). La agregación semanal puede enmascarar patrones intra-semana útiles para el modelo boosting.

*Contraargumento*: Mayor ruido estadístico por día. Se mitiga con lags de ventana amplia (MA-7, MA-14).

*Decisión*: **Diaria para entrenamiento, semanal para reporting**.

---

**DECISIÓN 1.B — Unidad de predicción: SKU × día**

*Opciones evaluadas*: SKU individual, familia de producto, canal de venta.

*Justificación*: Predecir a nivel SKU es el caso más granular y de mayor valor operativo (gestión de stock). Los modelos agregados son más estables pero menos accionables.

*Riesgo*: SKUs con baja frecuencia de ventas tienen mayor error relativo (MAPE inestable cuando $y \approx 0$).

*Mitigación*: Excluir SKUs con <30 días de histórico. Evaluar métricas en términos absolutos (MAE, RMSE) para SKUs esporádicos.

*Decisión*: **SKU × día. SKUs esporádicos analizados por separado con MAE, no MAPE**.

---

**DECISIÓN 1.C — Período histórico: 2 años (730 días)**

*Justificación*: Dos años de histórico capturan dos ciclos de estacionalidad anual completos. Suficiente para Prophet (requiere 2+ periodos) y SARIMAX (requiere AL MENOS 2 × período estacional de observaciones). Menos de un año comprometería la estabilidad de los modelos estacionales.

*Decisión*: **Mínimo 730 días. Ideal 1095 días (3 años) si disponible**.

---

## 2. Protocolo de Validación Temporal

### 2.1 Decisión: Expanding Window

**DECISIÓN 2.A — Expanding Window en lugar de Sliding Window**

*Opciones evaluadas*:
- Rolling/sliding window: train fijo N días, test N+1 a N+T
- Expanding window: train crece, test siempre en [last_train + gap, last_train + gap + H]
- Hold-out único: una sola partición train/test

*Justificación expanding window*:
- Más representativa del escenario real: el modelo en producción se reentrena con todos los datos disponibles.
- Más estable estadísticamente: mayor volumen de train en cada fold.
- Alineada con buenas prácticas de forecasting temporal (Hyndman, 2021).

*Razón de rechazo de hold-out único*:
- Una sola partición es susceptible a sobreajuste accidental al período de test.
- No detecta degradación a lo largo del tiempo.

```python
def expanding_window_cv(df, n_splits=5, test_size_days=30, gap_days=7):
    """
    Expanding window cross-validation para series temporales.
    
    Folds (ejemplo 5 splits, test=30 días, gap=7 días):
    Fold 1: Train [t0, t300]   → Test [t308, t337]
    Fold 2: Train [t0, t330]   → Test [t338, t367]
    Fold 3: Train [t0, t360]   → Test [t368, t397]
    Fold 4: Train [t0, t390]   → Test [t398, t427]
    Fold 5: Train [t0, t420]   → Test [t428, t457]
    """
    n = len(df)
    fold_size = (n - test_size_days * n_splits - gap_days * n_splits) // n_splits
    
    splits = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1) + fold_size
        test_start = train_end + gap_days
        test_end = test_start + test_size_days
        
        train_idx = list(range(0, train_end))
        test_idx = list(range(test_start, min(test_end, n)))
        splits.append((train_idx, test_idx))
    
    return splits
```

*Decisión*: **Expanding window, 5 folds, test_size=30 días, gap=7 días**.

---

**DECISIÓN 2.B — Gap de 7 días entre train y test**

*Justificación*: Evitar que las últimas observaciones del train "contaminen" el test a través de features con lag corto (lag_1, lag_3). El gap garantiza que ningún lag calculado en el test tenga dependencia directa con datos recientes del train. El valor de 7 días corresponde al lag mínimo significativo en datos de consumo semanal.

*Decisión*: **gap_days=7. Ajustable a 14 si hay lags > lag_7 dominantes**.

---

### 2.2 Anti-Patrón: Data Leakage

Los siguientes errores son críticos y se han mitigado explícitamente:

| Tipo de Leakage | Causa | Mitigación |
|---|---|---|
| Future leakage en features | Calcular MA con datos futuros | Features calculadas estrictamente con `df.shift(1).rolling(N)` |
| Target encoding con test | Calcular estadísticos de target incluyendo test | Target encoding sólo sobre train fold |
| Split aleatorio | `train_test_split(shuffle=True)` | `TimeSeriesSplit` o split por fecha |
| Normalización global | Fit del scaler sobre todo el dataset | Fit sobre train, transform sobre train+test |

```python
# MAL: calcula MA con información futura (no shiftea antes de rolling)
df['ma7_WRONG'] = df['quantity'].rolling(7).mean()

# BIEN: shift(1) garantiza que no usa el valor actual
df['ma7_CORRECT'] = df['quantity'].shift(1).rolling(7).mean()
```

---

## 3. Selección de Métricas

### 3.1 Métricas Primarias

**DECISIÓN 3.A — MAPE como métrica principal**

*Fórmula*: $\text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right| \times 100$

*Justificación*:
- Interpretable en negocio: "error de X% sobre demanda real".
- Independiente de escala: comparable entre SKUs con volúmenes distintos.
- Estándar de industria en supply chain (APICS, Gartner).

*Limitación crítica*: Indefinida cuando $y_t = 0$ (división por cero). Sesgada hacia subestimaciones.

*Mitigación*:
- Excluir períodos con demanda cero del cálculo MAPE.
- Reportar MAE y RMSE como métricas complementarias.
- Para SKUs intermitentes: usar MASE (Mean Absolute Scaled Error).

*Decisión*: **MAPE global como KPI principal. MAE y RMSE complementarios. MASE para SKUs esporádicos**.

---

**DECISIÓN 3.B — Métricas complementarias**

| Métrica | Fórmula | Cuándo usar |
|---|---|---|
| MAE | $\frac{1}{n}\sum\|y - \hat{y}\|$ | SKUs con demanda variable, interpretable en unidades |
| RMSE | $\sqrt{\frac{1}{n}\sum(y - \hat{y})^2}$ | Penaliza errores grandes, útil para detección outliers |
| MASE | $\frac{\text{MAE}}{\text{MAE}_{\text{naive}}}$ | SKUs esporádicos, invariante a escala y frecuencia |
| Coverage | % predicciones dentro de intervalo [lower, upper] | Evaluar intervalos de confianza |

---

### 3.2 Métricas Segmentadas

**DECISIÓN 3.C — Evaluar MAPE segmentado como métricas de segundo nivel**

*Razón*: Un MAPE global bajo puede enmascarar degradación severa en segmentos críticos (ej. stock out en períodos promo).

*Segmentación definida*:
```
MAPE_global        → KPI principal
MAPE_sku_{id}      → Detectar SKUs problemáticos
MAPE_promo         → Evaluar sensibilidad a cambios de régimen
MAPE_no_promo      → Rendimiento base (sin distorsión)
MAPE_Q{1..4}       → Estacionalidad por trimestre
```

*Umbral de alerta*:
- `MAPE_sku_{id} > 20%` → SKU marcado como "difícil", análisis adicional
- `MAPE_promo / MAPE_no_promo > 1.3` → Señal de cambio de régimen → proponer modelo separado

---

## 4. Feature Engineering

### 4.1 Features Temporales

**DECISIÓN 4.A — Lags: lag(1), lag(7), lag(14), lag(28)**

*Justificación*:
- `lag_1`: demanda del día anterior (autocorrelación fuerte en consumo diario)
- `lag_7`: mismo día de la semana anterior (estacionalidad semanal)
- `lag_14`, `lag_28`: efectos de periodicidad quincenal y mensual

*Criterio de selección*: ACF (Autocorrelation Function) y PACF (Partial ACF) del entrenamiento. Sólo incluir lags con correlación significativa ($|\rho| > 0.2$, $p < 0.05$).

```python
# Lag features (con shift para evitar leakage)
df['lag_1'] = df.groupby('sku')['quantity'].shift(1)
df['lag_7'] = df.groupby('sku')['quantity'].shift(7)
df['lag_14'] = df.groupby('sku')['quantity'].shift(14)
df['lag_28'] = df.groupby('sku')['quantity'].shift(28)
```

*Decisión*: **lag_1, lag_7, lag_14, lag_28. Verificar significancia con ACF/PACF post-EDA**.

---

**DECISIÓN 4.B — Ventanas móviles: MA(7), MA(14), STD(7)**

*Justificación*: Las medias móviles suavizan el ruido y capturan tendencia local. La desviación estándar rolling captura volatilidad (útil para detectar entradas en períodos de alta variabilidad).

```python
df['ma_7']  = df.groupby('sku')['quantity'].shift(1).rolling(7).mean()
df['ma_14'] = df.groupby('sku')['quantity'].shift(1).rolling(14).mean()
df['std_7'] = df.groupby('sku')['quantity'].shift(1).rolling(7).std()
```

*Nota*: Siempre `.shift(1)` antes de `.rolling()` para evitar incluir el valor actual.

*Decisión*: **MA_7, MA_14, STD_7. Evaluar MA_28 si el período es mensual**.

---

**DECISIÓN 4.C — Features de Calendario**

| Feature | Tipo | Descripción |
|---|---|---|
| `dow` | int (0-6) | Día de la semana (0=lunes) |
| `month` | int (1-12) | Mes del año |
| `quarter` | int (1-4) | Trimestre |
| `is_weekend` | bool | Sábado o domingo |
| `is_holiday` | bool | Festivo local (fuente: API o CSV) |
| `days_to_eom` | int | Días al fin de mes |
| `week_of_year` | int (1-52) | Semana del año |

*Justificación*: Demanda retail y de consumo tiene componentes estacionales semanales, mensuales y anuales. Las variables de calendario permiten que modelos no-ARIMA (XGBoost) capturen estas señales sin parametrización explícita.

*Codificación*: Variables categóricas ordinales (dow, month) se pasan como int sin one-hot encoding al XGBoost (maneja nativamente splits no-lineales).

*Decisión*: **Incluir todas. Excluir `week_of_year` si número de datos <2 años (underfitting)**.

---

### 4.2 Features Exógenas

**DECISIÓN 4.D — Promotion Flag como feature binaria**

*Justificación*: Variable con mayor impacto esperado en cambios de régimen de demanda. Confirmado en EDA: demanda media en días de promoción es 1.3-1.8× la media normal.

*Problema identificado*: El modelo puede no generalizar bien a **nuevas promociones** (distintas mecánicas, productos distintos). El análisis segmentado (sección 7) cuantificará este impacto.

*Decisión*: **Incluir como feature. Monitorizar degradación en `MAPE_promo`**.

---

**DECISIÓN 4.E — Temperatura como feature exógena**

*Condición*: Sólo incluir si la correlación con demanda es estadísticamente significativa ($r > 0.15$, $p < 0.05$) para al menos un SKU piloto.

*Fuente*: API AEMET (Agencia Estatal de Meteorología) o CSV meteorológico.

*Decisión*: **Condicional. Incluir si supera umbral de correlación. Excluir si dataset pequeño y riesgo de overfitting**.

---

### 4.3 Selección Final de Features

```python
FEATURES_BASE = [
    # Lags
    'lag_1', 'lag_7', 'lag_14', 'lag_28',
    # Rolling
    'ma_7', 'ma_14', 'std_7',
    # Calendar
    'dow', 'month', 'quarter', 'is_weekend', 'is_holiday',
    # Exógenas
    'promotion_flag',
]

FEATURES_EXTENDED = FEATURES_BASE + [
    'lag_21',
    'ma_28',
    'days_to_eom',
    'temperature',
    'price_change',
]

TARGET = 'quantity'
```

*Proceso de selección*:
1. Entrenar XGBoost con FEATURES_EXTENDED
2. Calcular feature importance (SHAP)
3. Excluir features con SHAP value < umbral (0.01 por defecto)
4. Verificar que MAPE no empeora con subconjunto reducido

---

## 5. Selección de Modelos

### 5.1 Baseline: Seasonal Naive

**DECISIÓN 5.A — Seasonal Naive como baseline obligatorio**

*Fórmula*: $\hat{y}_{t+h} = y_{t+h-m}$ donde $m = 7$ (período estacional = semana)

*Justificación*:
- Todo modelo competitivo DEBE superar baseline.
- Referencia interpretable: "predecir lo mismo que la semana pasada".
- Si un modelo no supera seasonal_naive, hay un error metodológico (leak, overfitting o features incorrectas).

*MAPE esperado*: 15-22% en demanda diaria retail.

*Decisión*: **Obligatorio. MAPE baseline es el suelo de aceptación**.

---

### 5.2 Modelo 1: XGBoost

**DECISIÓN 5.B — XGBoost como modelo principal**

*Justificación técnica*:
- Estado del arte en competencias de forecasting tabular (M5 Competition, 2020: mayoría de top-50 usaron LightGBM/XGBoost).
- Manejo nativo de features mixtas (numéricas + booleanas + ordinales).
- Compatible con lags y ventanas rolling sin parametrización explícita de estacionalidad.
- Interpretable: SHAP values nativos.
- Rápido: entrenamiento en segundos para 100K filas.

*Limitación*: No extrapola bien fuera del rango de entrenamiento. Si la tendencia es fuerte y creciente, puede underpredict en el futuro.

*Mitigación*: Incluir features de tendencia (lag diferenciado, index temporal normalizado).

*Formato de entrada*: Regresión multistep → una fila por (sku, date). Target: quantity del día siguiente (o H días después si se usa multistep directo).

```python
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_lambda=1.0,  # L2 regularización
        random_state=42
    ))
])
```

*Decisión*: **XGBoost como modelo principal. LightGBM como alternativa si dataset >500K filas**.

---

### 5.3 Modelo 2: Prophet

**DECISIÓN 5.C — Prophet para captura de estacionalidad compleja**

*Justificación*:
- Diseñado específicamente para series con estacionalidad múltiple (diaria, semanal, anual).
- Robusto a missing values y outliers.
- Soporta regresores externos (covariables adicionales).
- Genera intervalos de incertidumbre nativamente.

*Limitación*:
- Asume tendencia piecewise lineal o logística. Puede fallar en demanda altamente irregular.
- Entrenamiento más lento que XGBoost.
- Requiere columna `ds` (datetime) y `y` (target). No acepta features tabulares directamente.

*Configuración*:
```python
from prophet import Prophet

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,  # datos diarios pero granularidad mayor
    changepoint_prior_scale=0.05,  # regularización tendencia
    seasonality_prior_scale=10,
    seasonality_mode='multiplicative'  # demanda retail suele ser multiplicativa
)
model.add_regressor('promotion_flag')
model.add_regressor('is_holiday')
```

*Decisión de `seasonality_mode`*:
- `additive`: estacionalidad suma magnitud fija → independiente del nivel de demanda.
- `multiplicative`: estacionalidad es proporcional → más realista si la demanda crece.
- **Elegir según EDA**: si la amplitud de las fluctuaciones estacionales crece con el nivel → `multiplicative`.

*Decisión*: **Prophet como modelo estacional. Comparar additive vs multiplicative en Día 9**.

---

### 5.4 Modelo 3: SARIMAX

**DECISIÓN 5.D — SARIMAX para modelado explícito de autocorrelación**

*Justificación académica*:
- Familia de modelos clásica, bien entendida teóricamente.
- Permite inferencia estadística sobre coeficientes (test de significancia).
- Covariables exógenas (X) integradas en el componente regresivo.
- Referencia obligada en memorias de TFM sobre forecasting.

*Configuración*:
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Selección de órdenes: grid search o auto_arima
model = SARIMAX(
    endog=y_train,
    exog=X_train[['promotion_flag', 'is_holiday']],
    order=(p, d, q),         # AR, diferenciación, MA
    seasonal_order=(P, D, Q, 7),  # SAR, SD, SMA, período=7
    enforce_stationarity=False,
    enforce_invertibility=False
)
```

*Selección de órdenes (p,d,q)(P,D,Q)*:
1. Test ADF para determinar `d` (diferenciación regular)
2. Test KPSS complementario
3. Grid search (p,q) ∈ {0,1,2}, (P,Q) ∈ {0,1}
4. Criterio: AIC (penaliza complejidad)

*Limitación*: Lento para múltiples SKUs. Estacionaridad a veces difícil de garantizar.

*Decisión*: **SARIMAX como modelo comparativo académico. Si dataset >10 SKUs, evaluar sólo sobre SKU piloto**.

---

### 5.5 Alternativas Descartadas

| Modelo | Razón de Descarte |
|---|---|
| LSTM/RNN | Requiere más datos (>5K observaciones por SKU), difícil de debugar, sin ganancia clara sobre XGBoost en M5 |
| Theta | No soporta covariables fácilmente |
| ETS | Similar a seasonal naive, no aporta diferenciación suficiente en comparativa |
| VAR | Asume linealidad, no maneja covariables exógenas bien |
| Ensemble XGB+Prophet | Reservado para fase post-TFM si mejora MAPE >3% |

---

## 6. Configuración de Hiperparámetros

### 6.1 XGBoost: Hyperparameter Search

**DECISIÓN 6.A — Grid search limitado vs. Bayesian optimization**

*Justificación*: Con 14 días de plazo, el grid search simple (máximo 20 combinaciones) es suficiente para validar el impacto de hiperparámetros principales. Bayesian optimization (Optuna) se deja para refinamiento post-defensa.

```python
XGBOOST_PARAM_GRID = {
    'model__max_depth': [4, 5, 6],
    'model__learning_rate': [0.05, 0.1],
    'model__n_estimators': [200, 300],
    'model__subsample': [0.8],
    'model__colsample_bytree': [0.7, 0.8],
}

# Usar TimeSeriesSplit (no KFold aleatorio)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

cv = TimeSeriesSplit(n_splits=5)
search = GridSearchCV(
    xgb_pipeline,
    XGBOOST_PARAM_GRID,
    cv=cv,
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=1
)
```

*Decisión*: **Grid search (≤24 combinaciones). Registrar cada run en MLflow para auditoría**.

---

### 6.2 Prophet: Parámetros Fijos con Justificación

| Parámetro | Valor | Justificación |
|---|---|---|
| `changepoint_prior_scale` | 0.05 | Regularización moderada; previene overfitting en cambios de tendencia |
| `seasonality_prior_scale` | 10 | Permite estacionalidad fuerte (demanda retail tiene señal semanal robusta) |
| `seasonality_mode` | Según EDA | `multiplicative` si amplitud crece; `additive` si estable |
| `changepoint_range` | 0.8 | Detectar changepoints hasta el 80% del histórico |

---

### 6.3 SARIMAX: Selección de Órdenes

```python
from statsmodels.tsa.stattools import adfuller, kpss

def select_sarimax_order(series):
    """
    Protocolo:
    1. Test ADF → determinar d
    2. Grid search (p,d,q)(P,D,Q,7)
    3. Seleccionar mín AIC
    """
    # Diferenciación
    adf_stat, adf_p, *_ = adfuller(series.dropna())
    d = 0 if adf_p < 0.05 else 1  # d=0 si ya estacionaria
    
    best_aic = float('inf')
    best_order = None
    
    for p in range(3):
        for q in range(3):
            for P in range(2):
                for Q in range(2):
                    try:
                        model = SARIMAX(series, order=(p,d,q), seasonal_order=(P,1,Q,7))
                        res = model.fit(disp=False)
                        if res.aic < best_aic:
                            best_aic = res.aic
                            best_order = ((p,d,q), (P,1,Q,7))
                    except:
                        continue
    return best_order
```

*Decisión*: **AIC como criterio de selección SARIMAX. Registrar AIC, BIC y Ljung-Box test en MLflow**.

---

## 7. Análisis Segmentado

### 7.1 Degradación en Períodos Promocionales

**DECISIÓN 7.A — Análisis obligatorio de MAPE(promo) vs MAPE(no_promo)**

*Contexto*: El dataset presenta degradación observada en períodos promocionales. El análisis segmentado permite cuantificar el impacto y proponer soluciones.

*Protocolo*:
```python
def analyze_promotional_degradation(y_true, y_pred, promo_flag):
    """
    Cuantifica el gap de error entre períodos promo y no-promo.
    
    Umbral crítico:
    ratio > 1.3 → Proponer modelo específico para promo
    ratio > 1.5 → Señal fuerte de cambio de régimen no capturado
    ratio > 2.0 → Rediseñar features de promo
    """
    mask_promo = promo_flag == 1
    
    mape_promo    = mape(y_true[mask_promo], y_pred[mask_promo])
    mape_no_promo = mape(y_true[~mask_promo], y_pred[~mask_promo])
    ratio = mape_promo / mape_no_promo
    
    print(f"MAPE (promo):     {mape_promo:.2f}%")
    print(f"MAPE (no promo):  {mape_no_promo:.2f}%")
    print(f"Ratio:            {ratio:.2f}x")
    
    if ratio > 1.3:
        print("[ALERTA] Degradación significativa en períodos promocionales.")
        print("  Recomendación: Considerar modelo separado o features de interacción.")
    
    return mape_promo, mape_no_promo, ratio
```

*Acciones según ratio*:

| Ratio | Acción |
|---|---|
| < 1.1 | Sin problema. Modelo captura bien el efecto promo. |
| 1.1 – 1.3 | Añadir features de interacción promo × lag. |
| 1.3 – 1.5 | Proponer ensemble: modelo base + modelo promo específico. |
| > 1.5 | Modelos separados obligatorios. Documentar en TFM como limitación y línea futura. |

*Decisión*: **Reportar ratio en Día 11. Documentar en capítulo de resultados. Si ratio > 1.3, incluir propuesta de modelo separado como línea futura**.

---

### 7.2 Análisis por SKU

**DECISIÓN 7.B — MAPE individual por SKU**

*Objetivo*: Identificar SKUs con MAPE > umbral (20%) que requieren tratamiento especial.

*Clasificación*:
```
SKU Tier A: MAPE < 10%  → Modelo funciona bien
SKU Tier B: MAPE 10-20% → Revisar features específicas del SKU
SKU Tier C: MAPE > 20%  → Considerar modelo específico o excluir de pipeline
```

*Visualización obligatoria*:
- Boxplot MAPE por SKU (distribución de errores)
- SHAP dependence plot: `promotion_flag` vs `lag_7` por SKU tier

---

### 7.3 Análisis con SHAP

**DECISIÓN 7.C — SHAP sobre LIME para interpretabilidad**

*Razón de SHAP sobre LIME*:
- SHAP es consistente: si el modelo cambia y la feature importa más, SHAP sube.
- LIME es local (un punto), SHAP provee visión global + local.
- SHAP está integrado nativamente en XGBoost (`tree_explainer`), sin overhead.
- Compatible con GDPR Art. 22 (explicabilidad de decisiones automáticas).

```python
import shap

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)

# Plots:
shap.summary_plot(shap_values, X_test)                    # feature importance global
shap.dependence_plot('promotion_flag', shap_values, X_test)  # efecto promo
shap.waterfall_plot(explainer(X_test[0]))                 # explicación puntual (1 pred)
```

*Decisión*: **SHAP obligatorio. 3 plots mínimos: summary, dependence(promo_flag), waterfall(1 ejemplo)**.

---

## 8. Selección de Modelo Final

### 8.1 Protocolo de Decisión

```
Candidatos: baseline, xgboost, prophet, sarimax

Paso 1: Filtrar modelos con MAPE > baseline + 2% → DESCARTAR
Paso 2: Comparar MAPE global restantes → ordenar ascendente
Paso 3: Evaluar MAPE segmentado (promo, SKU) → penalizar si ratio > 1.3
Paso 4: Considerar latencia inferencia (XGBoost ~ms, SARIMAX ~segundos)
Paso 5: Tomar decisión y documentar en tabla
```

### 8.2 Tabla de Decisión (Plantilla)

| Modelo | MAPE Global | MAPE Promo | MAPE No-Promo | Ratio | Latencia | Decisión |
|---|---|---|---|---|---|---|
| Baseline (seasonal_naive) | ___ % | ___ % | ___ % | ___ | <1ms | Referencia |
| XGBoost | ___ % | ___ % | ___ % | ___ | ~5ms | ___ |
| Prophet | ___ % | ___ % | ___ % | ___ | ~500ms | ___ |
| SARIMAX | ___ % | ___ % | ___ % | ___ | ~2s | ___ |
| **Seleccionado** | | | | | | **___ ** |

*Esta tabla se completa el Día 10 con resultados reales.*

### 8.3 Criterios de Desempate

Si dos modelos tienen MAPE similar (diferencia < 1%):

1. **Menor MAPE en períodos promo** → prioridad (es el segmento más crítico)
2. **Menor latencia de inferencia** → para pipeline batch semanal, XGBoost es preferible
3. **Mejor interpretabilidad** → SHAP más completo en XGBoost
4. **Mayor estabilidad temporal** → comparar std(MAPE) entre folds

---

## 9. Propuesta de Industrialización

### 9.1 Pipeline de Reentrenamiento

**DECISIÓN 9.A — Reentrenamiento semanal**

*Justificación*: La demanda tiene componentes que evolucionan (cambio de precios, catálogo, tendencias). Reentrenar cada semana garantiza frescura del modelo sin sobre-rotación.

*Protocolo*:
```
Cada lunes:
1. Great Expectations: validar datos nuevos (frescura, nulos, rangos)
2. Reentrenar modelo con histórico actualizado
3. Evaluar MAPE en últimas 4 semanas
4. Si MAPE_nuevo < MAPE_prod + 1% → promote a registry
5. Guardar predicciones semana actual en analytics.predictions
```

**DECISIÓN 9.B — MLflow Model Registry como control de versiones**

```python
# Registrar modelo
mlflow.sklearn.log_model(
    model,
    artifact_path="xgboost",
    registered_model_name="tfm_forecaster_xgboost"
)

# Promover a producción
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="tfm_forecaster_xgboost",
    version=2,
    stage="Production"
)
```

*Decisión*: **MLflow Registry con 3 etapas: Staging → Production → Archived**.

---

### 9.2 Monitorización de Drift

**DECISIÓN 9.C — Monitorización basada en métricas de negocio**

*Alternativas evaluadas*:
- KS-test estadístico sobre distribución features → sensible a cambios no relevantes
- MAPE rolling sobre últimas N semanas → directamente accionable
- Population Stability Index (PSI) → estándar en banca, menos habitual en forecasting

*Decisión*: **MAPE rolling (30 días) + alerta si MAPE > umbral configurado (default: 15%)**.

```python
MONITORING_CONFIG = {
    'mape_alert_threshold': 15.0,   # % MAPE máximo aceptable
    'mape_window_days': 30,          # ventana rolling
    'promo_ratio_alert': 1.3,        # alerta si degradación promo
    'min_predictions_to_evaluate': 20  # mínimo muestras para calcular MAPE
}
```

---

## 10. Hipótesis Rechazadas

Decisiones que se evaluaron y se descartaron explícitamente.

### HYP-01: Normalización de target con log(1+y)

*Hipótesis*: Normalizar la demanda con `log(1+y)` reduce el MAPE al hacer la distribución más simétrica.

*Resultado*: MAPE en escala logarítmica no es comparable con MAPE en escala original. Invertir la transformación introduce sesgo (Jensen's inequality). Se descarta para mantener métricas interpretables.

*Alternativa*: Usar RMSE (que penaliza valores grandes) si distribución de errores es muy skewed.

---

### HYP-02: One-hot encoding de day_of_week y month

*Hipótesis*: OHE captura mejor los efectos no lineales de día y mes.

*Resultado*: XGBoost maneja ordinales internamente mediante splits. OHE añade dimensionalidad innecesaria y puede causar colinealidad. En tests previos, no mejora MAPE.

*Decisión final*: Variables ordinales (int) para XGBoost. OHE sólo si se usa regresión lineal como modelo adicional.

---

### HYP-03: Diferenciación de la serie temporal antes de entrenar SARIMAX

*Hipótesis*: Diferenciar manualmente antes de pasar a SARIMAX evita problemas de convergencia.

*Resultado*: SARIMAX con parámetro `d` maneja la diferenciación internamente. Diferenciar externamente y pasar a SARIMAX provoca doble diferenciación y overfitting.

*Decisión final*: Dejar `d` y `D` como hiperparámetros de SARIMAX, determinados por test ADF.

---

### HYP-04: Target encoding de SKU

*Hipótesis*: Codificar el SKU con la media de demanda histórica mejora la generalización.

*Resultado*: Target encoding introduce leakage si no se hace correctamente por fold. Para un dataset con 1-2 SKUs en la PoC, es irrelevante. Para N SKUs en producción, requiere implementación cuidadosa con mean_encoding_cv.

*Decisión*: **Descartado en PoC. Revisitar en fase producción si N_SKU > 50**.

---

## 11. Decisiones Pendientes

Las siguientes decisiones están condicionadas a resultados del EDA y backtesting.

| ID | Decisión | Condición | Fecha resolución |
|---|---|---|---|
| DP-01 | `seasonality_mode` Prophet: additive vs multiplicative | Tras EDA (Día 5) | Día 5 |
| DP-02 | Lags adicionales: lag_21, lag_35 | Si ACF muestra autocorrelación en lag_21+ | Día 6 |
| DP-03 | Incluir temperatura como feature | Si correlación temperatura-demanda >0.15 | Día 6 |
| DP-04 | Modelo separado para períodos promo | Si ratio MAPE_promo/MAPE_no_promo >1.3 | Día 11 |
| DP-05 | Excluir SARIMAX si latencia >5s por SKU | Si grid search SARIMAX tarda >2h | Día 9 |
| DP-06 | Ensemble XGBoost+Prophet | Si ningún modelo supera 12% MAPE solo | Día 10 |

---

**Documento cerrado**: Día 14 (tras defensa)  
**Responsable de actualización**: Persona 2 (Ciencia de Datos)  
**Revisión cruzada**: Persona 1 (Ingeniería de Datos) en Día 13
