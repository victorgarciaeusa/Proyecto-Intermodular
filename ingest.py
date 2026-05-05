import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import json
import logging

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def load_csv_to_staging(csv_path: str, source: str = "store_sales_csv"):
    engine = create_engine(DB_URL)
    df = pd.read_csv(csv_path)

    # Renombrar columnas al contrato del schema
    df = df.rename(columns={
        "store": "sku",
        "sales": "quantity",
        "promo": "promotion_flag",
        "holiday": "is_holiday"
    })

    df["sku"] = df["sku"].astype(str)
    df["load_date"] = datetime.now()
    df["source"] = source
    df["raw_json"] = df.apply(lambda r: json.dumps(r.to_dict(default=str)), axis=1)

    rows_before = len(df)
    df.to_sql("sales_raw", engine, schema="staging", if_exists="append", index=False)
    logging.info(f"[INGESTA] {rows_before} filas cargadas desde {csv_path}")
    logging.info(f"[INGESTA] SKUs únicos: {df['sku'].nunique()}")
    logging.info(f"[INGESTA] Rango fechas: {df['date'].min()} -> {df['date'].max()}")

if __name__ == "__main__":
    load_csv_to_staging("data/raw/store_sales.csv")
