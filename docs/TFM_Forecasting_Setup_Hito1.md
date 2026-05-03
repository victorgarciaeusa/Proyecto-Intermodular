# TFM Forecasting — Diario Técnico
## Setup Infraestructura + Hito 1
**Persona A (Javi) · Días 1–4 · Mayo 2026**

---

## Entorno

| Parámetro | Valor |
|---|---|
| Host | Ryzen 5800X / 32GB DDR4 |
| VM | VirtualBox — Ubuntu 20.04.6 LTS |
| vCPU | 6 |
| RAM VM | 16 GB |
| Disco VM | 60 GB (LVM, ~57 GB usables) |
| Red | NAT + Port Forwarding |
| SO Guest | Ubuntu 20.04.6 LTS (kernel 5.4.0-216) |

---

## Día 1 — Setup VM + Docker + Esquemas PostgreSQL

### 1.1 Instalación Ubuntu Server

Durante el instalador se detectaron dos problemas:

**Problema 1 — Bond de red roto**
El instalador creó un `bond0` disabled con `enp0s8` como esclavo. Solo `enp0s3` tenía IP (`10.0.2.15/24` NAT).
```
bond0    disabled   bond master for enp0s8
enp0s3   10.0.2.15/24   (activa)
enp0s8   enslaved to bond0
```
**Solución:** ignorar en el instalador, limpiar post-boot con netplan.

**Problema 2 — LVM sin expandir**
El instalador dejó 29GB sin asignar en el volume group:
```
ubuntu-lv   28.996G   /
espacio disponible   29.000G
```
**Solución:** editar `ubuntu-lv` en el instalador → expandir a 57.996G antes de confirmar.

**SSH:** marcar "Instalar servidor OpenSSH" + "Permitir autenticación con contraseña".

### 1.2 Instalación de dependencias en la VM

```bash
sudo apt update && sudo apt upgrade -y
```

**Error:** `docker-compose-plugin` y `python3-pip` no disponibles en Ubuntu 20.04:
```
E: No se ha podido localizar el paquete docker-compose-plugin
E: No se ha podido localizar el paquete python3-pip
```

**Solución:**
```bash
sudo apt install -y docker.io docker-compose git curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo usermod -aG docker $USER
newgrp docker
```

Versiones resultantes:
```
Docker version 26.1.3
docker-compose version 1.25.0
pip 20.0.2 (python 3.8)
```

### 1.3 Port Forwarding VirtualBox

Configurado en VirtualBox → forecast → Red → Adaptador 1 → Avanzado → Reenvío de puertos:

| Nombre | Protocolo | Puerto Host | Puerto Invitado |
|---|---|---|---|
| ssh | TCP | 2222 | 22 |
| postgres | TCP | 5432 | 5432 |
| mlflow | TCP | 5000 | 5000 |

### 1.4 Conexión SSH desde Windows

**Error:** SSH no disponible en PowerShell:
```
ssh : El término 'ssh' no se reconoce como nombre de un cmdlet
```

**Solución (PowerShell admin):**
```powershell
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
```

Conexión desde CMD (no PowerShell):
```cmd
ssh -p 2222 forecast_user@localhost
```

### 1.5 Estructura del repositorio

```bash
mkdir -p ~/forecast/{src/etl,src/models,src/serving,notebooks,tests,docs,data/raw,mlflow,docker/mlflow}
cd ~/forecast
touch docker-compose.yml schema.sql
```

**Error:** caracteres especiales al copiar desde el chat — directorios creados con guión (`docs-`, `mlflow-`, etc.):
```bash
mv docs- docs && mv mlflow- mlflow && mv notebooks- notebooks && mv tests- tests
```

### 1.6 docker-compose.yml

**Error:** fichero corrupto al pegar desde el chat — `build: ./docker/mlflow` se concatenó con `version:` en línea 1:
```
build: ./docker/mlflowversion: "3.3"
```

**Solución:** reescribir con `cat > file << 'EOF'` en lugar de `nano`.

Fichero final (`~/forecast/docker-compose.yml`):
```yaml
version: "3.3"

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: tfm
      POSTGRES_PASSWORD: tfm1234
      POSTGRES_DB: forecasting
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./schema.sql:/docker-entrypoint-initdb.d/schema.sql
    restart: unless-stopped

  mlflow:
    build: ./docker/mlflow
    ports:
      - "5000:5000"
    command: >
      mlflow server
      --host 0.0.0.0
      --backend-store-uri postgresql://tfm:tfm1234@postgres/forecasting
      --default-artifact-root /mlflow/artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      - postgres
    restart: unless-stopped

volumes:
  pgdata:
  mlflow_artifacts:
```

**Dockerfile MLflow** (`~/forecast/docker/mlflow/Dockerfile`):
```dockerfile
FROM ghcr.io/mlflow/mlflow:v2.12.1
RUN pip install psycopg2-binary
```

**Motivo:** la imagen oficial `ghcr.io/mlflow/mlflow:v2.12.1` no incluye `psycopg2`. MLflow entraba en bucle `Restarting` con:
```
ModuleNotFoundError: No module named 'psycopg2'
```

### 1.7 schema.sql

```sql
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE staging.sales_raw (
    load_id        SERIAL,
    load_date      TIMESTAMP DEFAULT NOW(),
    source         VARCHAR(100),
    sku            VARCHAR(50),
    sale_date      DATE,
    quantity       NUMERIC,
    price          NUMERIC,
    promotion_flag BOOLEAN,
    store_id       VARCHAR(50),
    raw_json       JSONB
);

CREATE TABLE analytics.sales_curated (
    sku            VARCHAR(50)  NOT NULL,
    sale_date      DATE         NOT NULL,
    quantity       NUMERIC      NOT NULL,
    price          NUMERIC,
    promotion_flag BOOLEAN      DEFAULT FALSE,
    store_id       VARCHAR(50),
    PRIMARY KEY (sku, sale_date)
);

CREATE TABLE analytics.features (
    sku                VARCHAR(50)  NOT NULL,
    sale_date          DATE         NOT NULL,
    quantity           NUMERIC      NOT NULL,
    quantity_lag_1     NUMERIC,
    quantity_lag_7     NUMERIC,
    ma_7               NUMERIC,
    std_7              NUMERIC,
    day_of_week        INTEGER,
    month              INTEGER,
    is_holiday         BOOLEAN      DEFAULT FALSE,
    season             VARCHAR(10),
    promotion_flag     BOOLEAN      DEFAULT FALSE,
    PRIMARY KEY (sku, sale_date)
);

CREATE TABLE analytics.predictions (
    prediction_id   SERIAL,
    sku             VARCHAR(50)  NOT NULL,
    prediction_date DATE         NOT NULL,
    horizon_week    INTEGER      NOT NULL,
    predicted_qty   NUMERIC      NOT NULL,
    ci_lower        NUMERIC,
    ci_upper        NUMERIC,
    model_name      VARCHAR(50),
    run_timestamp   TIMESTAMP    DEFAULT NOW(),
    PRIMARY KEY (sku, prediction_date, horizon_week, model_name)
);
```

### 1.8 Levantar contenedores

```bash
docker-compose up -d --build
docker-compose ps
```

Resultado:
```
forecast_mlflow_1     Up   0.0.0.0:5000->5000/tcp
forecast_postgres_1   Up   0.0.0.0:5432->5432/tcp
```

Verificación esquemas:
```bash
docker exec -it forecast_postgres_1 psql -U tfm -d forecasting -c "\dn"
```
```
 analytics | tfm
 staging   | tfm
```

Verificación tablas:
```
staging.sales_raw
analytics.features
analytics.predictions
analytics.sales_curated
```

---

## Día 2 — Ingesta CSV → staging.sales_raw

### 2.1 Dataset

**Fichero:** `store_sales.csv`

| Parámetro | Valor |
|---|---|
| Filas | 7.300 |
| Columnas | date, store, sales, promo, holiday |
| Stores (SKUs) | 10 (1–10) |
| Rango temporal | 2022-01-01 → 2023-12-31 |
| Nulos en columnas clave | 0 |
| Duplicados (date, store) | 0 |

Transferencia desde Windows a la VM:
```cmd
scp -P 2222 "C:\Users\francisco 3.2\Documents\TFM_forecasting\store_sales.csv" forecast_user@localhost:~/forecast/data/raw/
```

**Error previo:** ejecutar `scp` desde dentro de la VM en lugar del host Windows:
```
ssh: Could not resolve hostname c: Temporary failure in name resolution
```

### 2.2 Script ingest.py

Mapeo de columnas CSV → schema:

| CSV | schema | Tipo |
|---|---|---|
| store | sku | VARCHAR |
| date | sale_date | DATE |
| sales | quantity | NUMERIC |
| promo | promotion_flag | BOOLEAN |
| holiday | (descartada en raw) | — |

**Error 1:** `json.dumps(r.to_dict(default=str))` — `to_dict()` no acepta `default`:
```
TypeError: to_dict() got an unexpected keyword argument 'default'
```
**Solución:** `json.dumps(r.to_dict(), default=str)`

**Error 2:** columna `date` no existe en `staging.sales_raw` (el schema usa `sale_date`):
```
psycopg2.errors.UndefinedColumn: column "date" of relation "sales_raw" does not exist
```
**Solución:** añadir rename `"date": "sale_date"` antes del insert.

**Error 3:** `promotion_flag` es BOOLEAN en PostgreSQL pero llega como integer (0/1):
```
psycopg2.errors.DatatypeMismatch: column "promotion_flag" is of type boolean but expression is of type integer
```
**Solución:**
```python
df["promotion_flag"] = df["promotion_flag"].astype(bool)
```

Script final (`src/etl/ingest.py`):
```python
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import json, logging

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def load_csv_to_staging(csv_path: str, source: str = "store_sales_csv"):
    engine = create_engine(DB_URL)
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "store": "sku", "date": "sale_date",
        "sales": "quantity", "promo": "promotion_flag"
    })
    df["sku"] = df["sku"].astype(str)
    df["promotion_flag"] = df["promotion_flag"].astype(bool)
    df["load_date"] = datetime.now()
    df["source"] = source
    df["raw_json"] = df.apply(lambda r: json.dumps(r.to_dict(), default=str), axis=1)
    cols = ["sale_date", "sku", "quantity", "promotion_flag", "load_date", "source", "raw_json"]
    rows_before = len(df)
    df[cols].to_sql("sales_raw", engine, schema="staging", if_exists="append", index=False)
    logging.info(f"[INGESTA] {rows_before} filas cargadas desde {csv_path}")

if __name__ == "__main__":
    load_csv_to_staging("data/raw/store_sales.csv")
```

Resultado:
```
[INGESTA] 7300 filas cargadas desde data/raw/store_sales.csv
[INGESTA] SKUs unicos: 10
[INGESTA] Rango fechas: 2022-01-01 -> 2023-12-31
```

---

## Día 3 — Auditoría del raw

Script `src/etl/audit_raw.py`. Resultado:

| Check | Valor | Estado |
|---|---|---|
| Shape | (7300, 10) | ✅ |
| Nulos en sku | 0 | ✅ |
| Nulos en sale_date | 0 | ✅ |
| Nulos en quantity | 0 | ✅ |
| Nulos en price | 7300 | ℹ️ columna no usada |
| Nulos en store_id | 7300 | ℹ️ columna no usada |
| Duplicados (sku, sale_date) | 0 | ✅ |
| Negativos en quantity | 0 | ✅ |
| Max gap temporal todos los SKUs | 1 día | ✅ |
| promotion_flag=True | 1476 (20%) | ✅ |

**Error:** `DatetimeIndex` no tiene método `.diff()`:
```
AttributeError: 'DatetimeIndex' object has no attribute 'diff'
```
**Solución:** convertir a Series antes del diff:
```python
dates = grp["sale_date"].sort_values().reset_index(drop=True)
max_gap = dates.diff().dt.days.dropna().max()
```

---

## Día 4 — Curación → analytics.sales_curated

Script `src/etl/curate.py`.

Reglas aplicadas:
1. Eliminar nulos en `(sku, sale_date, quantity)`
2. Eliminar duplicados `(sku, sale_date)` — keep first
3. Cast `sale_date` → `date`, `quantity` → numeric, `promotion_flag` → bool
4. Assert retención ≥ 90%
5. Write a `analytics.sales_curated`

**Error:** `engine.execute()` deprecado en SQLAlchemy 2.x:
```
AttributeError: 'Engine' object has no attribute 'execute'
```
**Solución:** verificación final ejecutada directamente en PostgreSQL:
```sql
SELECT COUNT(*) AS total,
    SUM(CASE WHEN sku IS NULL THEN 1 ELSE 0 END) AS nulos_sku,
    SUM(CASE WHEN sale_date IS NULL THEN 1 ELSE 0 END) AS nulos_fecha,
    SUM(CASE WHEN quantity IS NULL THEN 1 ELSE 0 END) AS nulos_qty
FROM analytics.sales_curated;
```

Resultado:
```
 total | nulos_sku | nulos_fecha | nulos_qty
-------+-----------+-------------+-----------
  7300 |         0 |           0 |         0
```

---

## ✅ HITO 1 COMPLETADO

| Criterio | Resultado | Go/No-Go |
|---|---|---|
| sales_curated poblada | 7300 filas | ✅ GO |
| nulos_sku | 0 | ✅ GO |
| nulos_fecha | 0 | ✅ GO |
| nulos_qty | 0 | ✅ GO |
| duplicados (sku, sale_date) | 0 | ✅ GO |
| retención | 100% (>90%) | ✅ GO |

---

## Estructura del repositorio al cierre del Hito 1

```
~/forecast/
├── docker-compose.yml
├── schema.sql
├── docker/
│   └── mlflow/
│       └── Dockerfile
├── src/
│   └── etl/
│       ├── ingest.py
│       ├── audit_raw.py
│       └── curate.py
├── data/
│   └── raw/
│       └── store_sales.csv
├── src/models/
├── src/serving/
├── notebooks/
├── tests/
└── docs/
```

## Servicios activos

| Servicio | Puerto host | Estado |
|---|---|---|
| PostgreSQL 15 | 5432 | Up |
| MLflow 2.12.1 | 5000 | Up |
| SSH | 2222 | Up |

---

*TFM Forecasting · Persona A · Diario técnico v1.0 · Mayo 2026*
