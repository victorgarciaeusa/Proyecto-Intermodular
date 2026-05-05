import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import mlflow
import logging

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"
MLFLOW_URI = "http://localhost:5000"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def expanding_window_backtest(df: pd.DataFrame, min_train_days: int = 90):
    results = []
    df = df.sort_values("sale_date").reset_index(drop=True)
    dates = df["sale_date"].unique()

    for i, cutoff in enumerate(dates[min_train_days:], start=min_train_days):
        test_idx = i + 1
        if test_idx >= len(dates):
            break
        test_date = dates[test_idx]
        test_row = df[df["sale_date"] == test_date]
        if test_row.empty:
            continue

        lag7_date = pd.Timestamp(test_date) - pd.Timedelta(days=7)
        train = df[df["sale_date"] <= cutoff]
        lag7_rows = train[train["sale_date"] == lag7_date]
        if lag7_rows.empty:
            continue

        y_pred = lag7_rows["quantity"].values[0]
        y_true = test_row["quantity"].values[0]

        results.append({
            "sale_date": test_date,
            "sku": test_row["sku"].values[0],
            "y_true": y_true,
            "y_pred": y_pred,
            "promotion_flag": test_row["promotion_flag"].values[0]
        })

    return pd.DataFrame(results)

def run_baseline():
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM analytics.features ORDER BY sku, sale_date", engine)
    df["sale_date"] = pd.to_datetime(df["sale_date"])

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("baseline_seasonal_naive")

    all_results = []
    for sku, grp in df.groupby("sku"):
        res = expanding_window_backtest(grp.copy())
        all_results.append(res)

    results = pd.concat(all_results).reset_index(drop=True)

    mape_global  = mape(results["y_true"], results["y_pred"])
    mape_promo   = mape(results[results["promotion_flag"]==True]["y_true"],
                        results[results["promotion_flag"]==True]["y_pred"])
    mape_nopromo = mape(results[results["promotion_flag"]==False]["y_true"],
                        results[results["promotion_flag"]==False]["y_pred"])
    mape_sku = results.groupby("sku").apply(
        lambda g: mape(g["y_true"], g["y_pred"])
    ).reset_index(name="mape")

    logging.info(f"[BASELINE] MAPE global:   {mape_global:.2f}%")
    logging.info(f"[BASELINE] MAPE promo:    {mape_promo:.2f}%")
    logging.info(f"[BASELINE] MAPE no-promo: {mape_nopromo:.2f}%")
    logging.info(f"\n[BASELINE] MAPE por SKU:\n{mape_sku.to_string(index=False)}")

    with mlflow.start_run(run_name="seasonal_naive"):
        mlflow.log_param("model", "seasonal_naive")
        mlflow.log_param("horizon_days", 7)
        mlflow.log_param("min_train_days", 90)
        mlflow.log_metric("mape_global", round(mape_global, 4))
        mlflow.log_metric("mape_promo", round(mape_promo, 4))
        mlflow.log_metric("mape_nopromo", round(mape_nopromo, 4))
        for _, row in mape_sku.iterrows():
            mlflow.log_metric(f"mape_sku_{row['sku']}", round(row["mape"], 4))
        results.to_csv("/tmp/baseline_predictions.csv", index=False)
        results.to_csv("/tmp/baseline_predictions.csv", index=False)
        logging.info("[BASELINE] Run registrado en MLflow ✅")

if __name__ == "__main__":
    run_baseline()
