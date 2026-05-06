import pandas as pd
from sqlalchemy import create_engine
import logging

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def curate():
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM staging.sales_raw", engine)
    rows_original = len(df)

    # 1. Eliminar nulos en columnas clave
    df = df.dropna(subset=["sku", "sale_date", "quantity"])

    # 2. Eliminar duplicados
    df = df.drop_duplicates(subset=["sku", "sale_date"], keep="first")

    # 3. Forzar tipos
    df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.date
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["quantity"])
    df["promotion_flag"] = df["promotion_flag"].fillna(False).astype(bool)

    # 4. Validar retención mínima 90%
    rows_final = len(df)
    retention = rows_final / rows_original
    logging.info(f"[CURACIÓN] Retención: {retention:.1%} ({rows_final}/{rows_original} filas)")
    assert retention >= 0.90, f"FAIL H1: retención {retention:.1%} < 90%"

    # 5. Escribir a analytics.sales_curated
    cols = ["sku", "sale_date", "quantity", "promotion_flag"]
    df[cols].to_sql("sales_curated", engine, schema="analytics", if_exists="replace", index=False)
    logging.info("[CURACIÓN] analytics.sales_curated poblada ✅")

    # 6. Verificación final
    result = engine.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN sku IS NULL THEN 1 ELSE 0 END) AS nulos_sku,
            SUM(CASE WHEN sale_date IS NULL THEN 1 ELSE 0 END) AS nulos_fecha,
            SUM(CASE WHEN quantity IS NULL THEN 1 ELSE 0 END) AS nulos_qty
        FROM analytics.sales_curated
    """
    ).fetchone()
    logging.info(
        f"[VERIFY] total={result[0]} nulos_sku={result[1]} nulos_fecha={result[2]} nulos_qty={result[3]}"
    )


if __name__ == "__main__":
    curate()
