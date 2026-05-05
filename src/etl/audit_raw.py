import pandas as pd
from sqlalchemy import create_engine
import logging

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def audit_raw():
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM staging.sales_raw", engine)

    print("\n=== AUDIT staging.sales_raw ===")
    print(f"Shape: {df.shape}")
    print(f"\n--- Nulos por columna ---")
    print(df.isnull().sum())

    print(f"\n--- Duplicados (sku, sale_date): {df.duplicated(subset=['sku','sale_date']).sum()} ---")

    df["sale_date"] = pd.to_datetime(df["sale_date"])
    print(f"\n--- Rango fechas: {df['sale_date'].min()} -> {df['sale_date'].max()} ---")
    print(f"\n--- SKUs únicos: {sorted(df['sku'].unique())} ---")
    print(f"\n--- quantity stats ---")
    print(df["quantity"].describe())
    print(f"\n--- Negativos en quantity: {(df['quantity'] < 0).sum()} ---")

    print("\n--- Gaps temporales por SKU ---")
    for sku, grp in df.groupby("sku"):
        dates = grp["sale_date"].sort_values().reset_index(drop=True)
        max_gap = dates.diff().dt.days.dropna().max()
        status = "WARN" if max_gap > 3 else "OK"
        print(f"  SKU {sku}: max_gap={int(max_gap)} dias [{status}]")

    print("\n--- promotion_flag ---")
    print(df["promotion_flag"].value_counts())
    print("\n=== FIN AUDIT ===\n")

if __name__ == "__main__":
    audit_raw()
