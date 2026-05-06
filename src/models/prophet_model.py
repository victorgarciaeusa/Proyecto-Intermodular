import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import mlflow
import logging
import itertools
import warnings
warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"
MLFLOW_URI = "http://localhost:5000"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def grid_search_prophet(train_df):
    """Grid search sobre SKU piloto para seleccionar mejores hiperparámetros."""
    param_grid = {
        "changepoint_prior_scale":  [0.001, 0.01, 0.05, 0.1, 0.3, 0.5],
        "seasonality_prior_scale":  [0.1, 1.0, 5.0, 10.0, 20.0, 50.0],
        "seasonality_mode":         ["additive", "multiplicative"],
        "fourier_order_weekly":     [3, 5, 7],
    }

    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))
    total  = len(combos)
    logging.info(f"[GRID PROPHET] Probando {total} combinaciones sobre SKU piloto...")

    best_mape  = np.inf
    best_params = None
    results = []

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        try:
            m = Prophet(
                changepoint_prior_scale = params["changepoint_prior_scale"],
                seasonality_prior_scale = params["seasonality_prior_scale"],
                seasonality_mode        = params["seasonality_mode"],
                yearly_seasonality      = True,
                weekly_seasonality      = False,  # la añadimos manual con fourier_order
                daily_seasonality       = False,
                holidays_prior_scale    = 10.0,
                interval_width          = 0.80
            )
            m.add_seasonality(
                name="weekly",
                period=7,
                fourier_order=params["fourier_order_weekly"]
            )
            m.add_regressor("promotion_flag", standardize=False)
            m.add_country_holidays(country_name="ES")
            m.fit(train_df[["ds", "y", "promotion_flag"]])

            # CV nativa Prophet sobre el train del piloto
            df_cv = cross_validation(
                m,
                initial="180 days",
                period="30 days",
                horizon="28 days",
                disable_tqdm=True
            )
            df_p  = performance_metrics(df_cv, rolling_window=1)
            m_val = df_p["mape"].mean() * 100

            results.append({**params, "mape_cv": m_val})

            if m_val < best_mape:
                best_mape   = m_val
                best_params = params.copy()
                logging.info(f"[GRID PROPHET] Nuevo mejor MAPE CV: {m_val:.4f}% params={params}")

        except Exception as e:
            logging.warning(f"[GRID PROPHET] combo {i}: {e}")
            continue

    logging.info(f"[GRID PROPHET] Mejor: {best_params} MAPE CV={best_mape:.4f}%")
    pd.DataFrame(results).sort_values("mape_cv").to_csv("/tmp/prophet_grid_results.csv", index=False)
    return best_params, best_mape

def run_prophet():
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM analytics.sales_curated ORDER BY sku, sale_date", engine)
    df["sale_date"]      = pd.to_datetime(df["sale_date"])
    df["promotion_flag"] = df["promotion_flag"].astype(int)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("prophet")

    # --- FASE 1: Grid search sobre SKU piloto ---
    PILOT_SKU = "1"
    logging.info(f"[PROPHET] FASE 1: Grid search sobre SKU piloto {PILOT_SKU}")
    pilot = df[df["sku"] == PILOT_SKU].sort_values("sale_date").reset_index(drop=True)
    pilot_p = pilot.rename(columns={"sale_date": "ds", "quantity": "y"})

    best_params, best_cv_mape = grid_search_prophet(pilot_p)

    # --- FASE 2: Expanding window paso 3 todos los SKUs ---
    logging.info("[PROPHET] FASE 2: Expanding window paso 3 todos los SKUs...")
    MIN_TRAIN   = 90
    all_results = []
    mape_per_sku = {}

    for sku in sorted(df["sku"].unique()):
        logging.info(f"[PROPHET] SKU {sku}...")
        grp   = df[df["sku"] == sku].sort_values("sale_date").reset_index(drop=True)
        grp_p = grp.rename(columns={"sale_date": "ds", "quantity": "y"})
        sku_results = []

        for i in range(MIN_TRAIN, len(grp_p) - 1, 3):  # paso 3
            train = grp_p.iloc[:i]
            test  = grp_p.iloc[i:i+1]
            try:
                m = Prophet(
                    changepoint_prior_scale = best_params["changepoint_prior_scale"],
                    seasonality_prior_scale = best_params["seasonality_prior_scale"],
                    seasonality_mode        = best_params["seasonality_mode"],
                    yearly_seasonality      = True,
                    weekly_seasonality      = False,
                    daily_seasonality       = False,
                    holidays_prior_scale    = 10.0,
                    interval_width          = 0.80
                )
                m.add_seasonality(
                    name="weekly",
                    period=7,
                    fourier_order=best_params["fourier_order_weekly"]
                )
                m.add_regressor("promotion_flag", standardize=False)
                m.add_country_holidays(country_name="ES")
                m.fit(train[["ds", "y", "promotion_flag"]])

                future   = test[["ds", "promotion_flag"]].copy()
                forecast = m.predict(future)

                sku_results.append({
                    "sale_date":      test["ds"].values[0],
                    "sku":            sku,
                    "y_true":         test["y"].values[0],
                    "y_pred":         forecast["yhat"].values[0],
                    "y_pred_lower":   forecast["yhat_lower"].values[0],
                    "y_pred_upper":   forecast["yhat_upper"].values[0],
                    "promotion_flag": test["promotion_flag"].values[0]
                })
            except Exception as e:
                logging.warning(f"[PROPHET] SKU {sku} idx {i}: {e}")
                continue

        if sku_results:
            sku_df   = pd.DataFrame(sku_results)
            sku_mape = mape(sku_df["y_true"], sku_df["y_pred"])
            mape_per_sku[sku] = sku_mape
            logging.info(f"[PROPHET] SKU {sku} MAPE: {sku_mape:.2f}%")
            all_results.append(sku_df)

    results      = pd.concat(all_results).reset_index(drop=True)
    mape_global  = mape(results["y_true"], results["y_pred"])
    mape_promo   = mape(results[results["promotion_flag"]==1]["y_true"],
                        results[results["promotion_flag"]==1]["y_pred"])
    mape_nopromo = mape(results[results["promotion_flag"]==0]["y_true"],
                        results[results["promotion_flag"]==0]["y_pred"])
    mape_sku = pd.DataFrame(
        list(mape_per_sku.items()), columns=["sku","mape"]
    ).sort_values("sku")

    logging.info(f"[PROPHET] MAPE global:   {mape_global:.2f}%")
    logging.info(f"[PROPHET] MAPE promo:    {mape_promo:.2f}%")
    logging.info(f"[PROPHET] MAPE no-promo: {mape_nopromo:.2f}%")
    logging.info(f"\n[PROPHET] MAPE por SKU:\n{mape_sku.to_string(index=False)}")

    results.to_csv("/tmp/prophet_predictions.csv", index=False)

    with mlflow.start_run(run_name="prophet_afinado"):
        mlflow.log_params(best_params)
        mlflow.log_param("model",            "Prophet")
        mlflow.log_param("grid_combos",      216)
        mlflow.log_param("cv_initial",       "180 days")
        mlflow.log_param("cv_horizon",       "28 days")
        mlflow.log_param("regressors",       "promotion_flag")
        mlflow.log_param("country_holidays", "ES")
        mlflow.log_param("expanding_step",   3)
        mlflow.log_metric("mape_cv_pilot",   round(best_cv_mape, 4))
        mlflow.log_metric("mape_global",     round(mape_global,  4))
        mlflow.log_metric("mape_promo",      round(mape_promo,   4))
        mlflow.log_metric("mape_nopromo",    round(mape_nopromo, 4))
        for _, row in mape_sku.iterrows():
            mlflow.log_metric(f"mape_sku_{row['sku']}", round(row["mape"], 4))
        logging.info("[PROPHET] Run registrado en MLflow ✅")

if __name__ == "__main__":
    run_prophet()
