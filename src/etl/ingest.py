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

    df = df.rename(columns={
        "store": "sku",
        "date": "sale_date",
        "sales": "quantity",
        "promo": "promotion_flag"
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
    logging.info(f"[INGESTA] SKUs unicos: {df['sku'].nunique()}")
    logging.info(f"[INGESTA] Rango fechas: {df['sale_date'].min()} -> {df['sale_date'].max()}")

if __name__ == "__main__":
    load_csv_to_staging("data/raw/store_sales.csv")
