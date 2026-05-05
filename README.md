# TFM Forecasting de Demanda

Sistema end-to-end de predicción de demanda semanal a partir de históricos de ventas y variables externas. Dado un histórico de ventas por tienda (SKU), el sistema predice las unidades vendidas en las próximas 4 semanas.

**Autores:** Javi · Víctor &nbsp;|&nbsp; **Mayo 2026**

---

## Arquitectura del proyecto

```
CSV de ventas
     │
     ▼
staging.sales_raw          ← Ingesta raw con logging
     │
     ▼
analytics.sales_curated    ← Limpieza y validación
     │
     ▼
analytics.features         ← Feature engineering (lags, calendario, exógenas)
     │
     ▼
Modelos: XGBoost · Prophet · SARIMAX · Baseline
     │
     ▼
analytics.predictions      ← Predicciones con intervalos de confianza
```

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Base de datos | PostgreSQL 15 |
| Tracking de experimentos | MLflow 2.12.1 |
| Orquestación | Prefect |
| Modelado | XGBoost, Prophet, statsmodels (SARIMAX) |
| ETL | Python, pandas, SQLAlchemy |
| Infraestructura | Docker, docker-compose |

---

## Estructura del repositorio

```
├── docker-compose.yml        # Servicios: PostgreSQL + MLflow
├── schema.sql                # Definición de tablas
├── requirements.txt          # Dependencias Python
├── docker/
│   └── mlflow/
│       └── Dockerfile
├── src/
│   ├── etl/                  # Ingesta, auditoría y curación (Javi)
│   ├── models/               # XGBoost, Prophet, SARIMAX, baseline (Víctor + Javi)
│   └── serving/              # Batch predictor semanal (Javi)
├── notebooks/
│   └── 01_eda_exploration.ipynb   # EDA completo (Víctor)
├── data/
│   └── raw/
│       └── store_sales.csv
├── tests/                    # pytest, cobertura >80% (Javi)
└── docs/                     # Memoria TFM, slides (Víctor)
```

---

## Levantar el entorno

### Requisitos previos
- Docker y docker-compose instalados
- Python 3.8+

### Arrancar los servicios

```bash
git clone https://github.com/victorgarciaeusa/Proyecto-Intermodular.git
cd Proyecto-Intermodular
docker-compose up -d --build
```

Esto levanta:
- **PostgreSQL 15** en `localhost:5432` con los esquemas `staging` y `analytics` ya creados
- **MLflow** en `http://localhost:5000`

### Instalar dependencias Python

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Ejecutar el pipeline ETL

```bash
# 1. Ingesta CSV → staging.sales_raw
python src/etl/ingest.py

# 2. Auditoría del raw
python src/etl/audit_raw.py

# 3. Curación → analytics.sales_curated
python src/etl/curate.py
```

---

## Dataset

| Parámetro | Valor |
|---|---|
| Fichero | `store_sales.csv` |
| Filas | 7.300 |
| Tiendas (SKUs) | 10 (1–10) |
| Rango temporal | 2022-01-01 → 2023-12-31 |
| Columnas | date, store, sales, promo, holiday |
| Nulos | 0 |
| Duplicados | 0 |

---

## Hitos del proyecto

| Hito | Día | Entregable | Estado |
|---|---|---|---|
| H1 | 4 | `analytics.sales_curated` — 0 nulos, >90% filas | ✅ |
| H2 | 7 | Baseline operativo — MAPE 15–20% | 🔄 |
| H3 | 10 | Comparativa modelos — MAPE < 12% | ⏳ |
| H4 | 11 | Análisis segmentado completo | ⏳ |

---

## Modelos

| Modelo | Responsable | Estado |
|---|---|---|
| Seasonal Naive (baseline) | Javi | 🔄 |
| XGBoost | Víctor | ⏳ |
| Prophet | Víctor | ⏳ |
| SARIMAX | Javi | ⏳ |

La métrica de evaluación principal es el **MAPE** (Mean Absolute Percentage Error). Objetivo final: MAPE < 12%.

---

## Credenciales por defecto (desarrollo)

```
PostgreSQL:
  host:     localhost:5432
  user:     tfm
  password: tfm1234
  db:       forecasting

MLflow:   http://localhost:5000
SSH VM:   ssh -p 2222 forecast_user@localhost
```
