# TFM Forecasting — Diario Técnico
Hito 2 | Persona A (Javi)
**Días 6-7 · Mayo 2026**
## 6. Día 6 — Feature Engineering → analytics.features

Script: `src/etl/feature_engineering.py`

**Error:** módulo `holidays` no instalado:
```
ModuleNotFoundError: No module named 'holidays'
```
Solución: `pip3 install holidays`

Features generadas:

| Feature | Descripción |
|---|---|
| quantity_lag_1 | Lag 1 día por SKU |
| quantity_lag_7 | Lag 7 días por SKU |
| ma_7 | Media móvil 7 días (shift 1) |
| std_7 | Desviación estándar 7 días (shift 1) |
| day_of_week | 0=lunes … 6=domingo |
| month | 1–12 |
| season | winter/spring/summer/autumn |
| is_holiday | Festivos España 2022–2023 |
| promotion_flag | De sales_curated |
| temperature | NULL (sin datos externos) |
| event_indicator | False (placeholder) |

**Resultado:**
```
[FEATURES] Filas entrada: 7300
[FEATURES] Filas tras dropna lags: 7230 (eliminadas: 70)
[FEATURES] SKUs: 10
[FEATURES] Rango: 2022-01-08 -> 2023-12-31
```
70 filas eliminadas = 7 días × 10 SKUs (warm-up de lags, sin lookahead bias).

---

## 7. Día 7 — Baseline seasonal_naive + Backtesting + MLflow

Script: `src/models/baseline.py`

### 7.1 Conflicto cryptography / pyOpenSSL

**Error crítico:** incompatibilidad entre `cryptography` instalado por pip y `pyOpenSSL` del sistema Ubuntu 20.04:
```
AttributeError: module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'
```

**Causa:** mlflow instaló `cryptography==47.x` incompatible con `pyOpenSSL 19.x` del sistema.

**pip del sistema también roto:**
```
AttributeError: module 'lib' has no attribute 'X509_V_FLAG_NOTIFY_POLICY'
```

**Solución definitiva:**
```bash
curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
sudo python3 /tmp/get-pip.py --force-reinstall
sudo pip3 install cryptography==41.0.7
sudo pip3 install pyopenssl==23.2.0
```

### 7.2 Error artifact MLflow

**Error:** MLflow intenta escribir artifacts en `/mlflow` (path del contenedor Docker), sin permisos desde el host:
```
PermissionError: [Errno 13] Permission denied: '/mlflow'
```

**Solución:** eliminar `log_artifact` del script:
```bash
sed -i '/log_artifact/d' src/models/baseline.py
```

Las métricas MAPE sí se registran correctamente vía `log_metric`.

### 7.3 Metodología backtesting

- **Expanding window**: sin lookahead bias
- **Min train days**: 90 días
- **Predicción**: valor de hace 7 días (seasonal naive)
- **Métrica**: MAPE

### 7.4 Resultados

| Segmento | MAPE | Objetivo guía |
|---|---|---|
| Global | **7.79%** | 15–20% |
| Promo | 10.95% | — |
| No-promo | 6.99% | — |

| SKU | MAPE |
|---|---|
| 1 | 7.40% |
| 2 | 7.91% |
| 3 | 7.90% |
| 4 | 7.03% |
| 5 | 8.43% |
| 6 | 7.76% |
| 7 | 7.76% |
| 8 | 7.39% |
| 9 | 7.61% |
| 10 | 8.68% |

> **Nota:** MAPE 7.79% está muy por debajo del rango esperado 15–20%.
> El dataset sintético es extremadamente regular (std=26, min=160, max=340).
> Esto implica que los modelos ML deberán justificar mejora marginal sobre un baseline ya bajo.
> Documentar en memoria como limitación del dataset.

**Run registrado en MLflow:**
```
Experiment: baseline_seasonal_naive
Run: seasonal_naive
URL: http://localhost:5000/#/experiments/1
```

###  **HITO 2 COMPLETADO**

| Criterio | Resultado | Estado |
|---|---|---|
| Baseline operativo | seasonal_naive |**OK**|
| Backtesting expanding window | implementado |**OK**|
| MAPE global | 7.79% |**OK**|
| Run MLflow registrado | seasonal_naive |**OK**|

---

## 8. Estado del repositorio

```
~/forecast/
├── docker-compose.yml
├── schema.sql
├── docker/
│   └── mlflow/
│       └── Dockerfile
├── src/
│   ├── etl/
│   │   ├── ingest.py
│   │   ├── audit_raw.py
│   │   ├── curate.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   └── baseline.py
│   └── serving/
├── data/
│   └── raw/
│       └── store_sales.csv
├── notebooks/
├── tests/
└── docs/
```

## 9. Servicios activos

| Servicio | Puerto host | Imagen |
|---|---|---|
| PostgreSQL 15 | 5432 | postgres:15 |
| MLflow 2.12.1 | 5000 | custom (+ psycopg2-binary) |
| SSH | 2222 | — |

## 10. Dependencias Python instaladas en VM

```
pandas==2.0.3
sqlalchemy==2.0.49
psycopg2-binary==2.9.10
holidays
mlflow==2.17.2
scikit-learn==1.3.2
cryptography==41.0.7
pyopenssl==23.2.0
```

## 11. Próximos pasos

| Día | Persona | Tarea |
|---|---|---|
| 9 | **A (Javi)** | SARIMAX: selección órdenes ADF/AIC, entrenamiento, MLflow |
| 9 | B (Víctor) | Prophet: estacionalidad, change points, MLflow |
| 10 | Ambos | Comparativa modelos → Hito 3 |

---

## 12. Pendiente para sincronización con Víctor

Antes del Día 10 (reunión conjunta), Víctor necesita:

1. Acceso al repositorio con los scripts actuales
2. Confirmación de nombres de columnas en `analytics.features`
3. Acceso a MLflow UI (`http://<IP_VM>:5000`) para ver el run baseline
