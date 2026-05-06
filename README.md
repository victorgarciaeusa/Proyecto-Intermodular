# TFM Forecasting — Sistema de Predicción de Demanda

Pipeline end-to-end de forecasting semanal para 10 SKUs con horizonte de 4 semanas.

## Requisitos

- Docker + docker-compose
- Python 3.8+
- Ubuntu 20.04+

## Instalación

```bash
git clone https://github.com/victorgarciaeusa/Proyecto-Intermodular.git
cd Proyecto-Intermodular
pip3 install -r requirements.txt
```

## Levantar infraestructura

```bash
docker-compose up -d
docker-compose ps  # verificar PostgreSQL + MLflow Up
```

MLflow UI disponible en: http://localhost:5000

## Ejecutar pipeline completo

```bash
# 1. Ingesta
python3 src/etl/ingest.py

# 2. Curación
python3 src/etl/curate.py

# 3. Feature engineering
python3 src/etl/feature_engineering.py

# 4. Modelos
python3 src/models/baseline.py
python3 src/models/sarimax_model.py
python3 src/models/xgboost_model.py
python3 src/models/prophet_model.py

# 5. Análisis de residuos
python3 src/models/residuals_analysis.py

# 6. Batch predictor semanal
python3 src/serving/batch_predictor.py

# 7. MLOps flow (Prefect)
python3 src/serving/mlops_flow.py
```

## Tests

```bash
python3 -m pytest tests/ -v --cov=tests --cov-report=term-missing
```

## Resultados — MAPE comparativa

| Modelo | MAPE Global | MAPE Promo | MAPE No-promo |
|---|---|---|---|
| Baseline seasonal_naive | 7.79% | 10.95% | 6.99% |
| Prophet afinado | 4.23% | 4.01% | 4.28% |
| SARIMAX (0,1,2)(0,1,1,7) | 3.74% | 3.47% | 3.81% |
| XGBoost riguroso | 3.73% | 3.40% | 3.81% |

**Modelo final seleccionado: SARIMAX** — mejor diagnóstico de residuos (Ljung-Box p=0.342), sin sesgo sistemático, homocedastico.

## Estructura del repositorio

src/
├── etl/           # Ingesta, curación, feature engineering
├── models/        # Baseline, SARIMAX, XGBoost, Prophet, residuos
└── serving/       # Batch predictor, MLOps flow Prefect
tests/             # pytest — 22 tests, cobertura 99%
data/raw/          # CSV fuente
docs/              # Memoria TFM, diarios técnicos
docker-compose.yml # PostgreSQL + MLflow
schema.sql         # Definición tablas

## Variables de entorno
DB_URL=postgresql://tfm:tfm1234@localhost:5432/forecasting
MLFLOW_URI=http://localhost:5000
