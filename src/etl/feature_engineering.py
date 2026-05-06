import pandas as pd
from sqlalchemy import create_engine
import logging

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


def build_features():
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM analytics.sales_curated ORDER BY sku, sale_date", engine)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values(["sku", "sale_date"]).reset_index(drop=True)

    logging.info(f"[FEATURES] Filas entrada: {len(df)}")

    df["quantity_lag_1"] = df.groupby("sku")["quantity"].shift(1)
    df["quantity_lag_7"] = df.groupby("sku")["quantity"].shift(7)
    df["ma_7"] = df.groupby("sku")["quantity"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["std_7"] = df.groupby("sku")["quantity"].transform(lambda x: x.shift(1).rolling(7).std())

    df["day_of_week"] = df["sale_date"].dt.dayofweek
    df["month"] = df["sale_date"].dt.month
    df["season"] = df["sale_date"].dt.month.apply(get_season)

    import holidays

    es_holidays = holidays.Spain(years=[2022, 2023])
    df["is_holiday"] = df["sale_date"].dt.date.apply(lambda d: d in es_holidays)

    df["temperature"] = None
    df["event_indicator"] = False

    rows_before = len(df)
    df = df.dropna(subset=["quantity_lag_7", "ma_7"])
    logging.info(
        f"[FEATURES] Filas tras dropna lags: {len(df)} (eliminadas: {rows_before - len(df)})"
    )

    cols = [
        "sku",
        "sale_date",
        "quantity",
        "quantity_lag_1",
        "quantity_lag_7",
        "ma_7",
        "std_7",
        "day_of_week",
        "month",
        "season",
        "is_holiday",
        "promotion_flag",
        "temperature",
        "event_indicator",
    ]
    df[cols].to_sql("features", engine, schema="analytics", if_exists="replace", index=False)
    logging.info(f"[FEATURES] analytics.features poblada con {len(df)} filas")
    logging.info(f"[FEATURES] SKUs: {df['sku'].nunique()}")
    logging.info(
        f"[FEATURES] Rango: {df['sale_date'].min().date()} -> {df['sale_date'].max().date()}"
    )


if __name__ == "__main__":
    build_features()
