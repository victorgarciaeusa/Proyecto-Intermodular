import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import mlflow
import itertools
import logging
import warnings

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"
MLFLOW_URI = "http://localhost:5000"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def adf_test(series, sku):
    result = adfuller(series.dropna())
    pvalue = result[1]
    status = "estacionaria" if pvalue < 0.05 else "NO estacionaria"
    logging.info(f"[ADF] SKU {sku}: p-value={pvalue:.4f} -> {status}")
    return pvalue < 0.05


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def grid_search_aic(train_series, exog_train, s=7):
    """Grid search AIC/BIC sobre SKU piloto para seleccionar órdenes óptimos."""
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    P_values = [0, 1]
    D_values = [0, 1]
    Q_values = [0, 1]

    best_aic = np.inf
    best_order = None
    best_seasonal = None
    results_grid = []

    total = (
        len(p_values)
        * len(d_values)
        * len(q_values)
        * len(P_values)
        * len(D_values)
        * len(Q_values)
    )
    logging.info(f"[GRID] Probando {total} combinaciones (p,d,q)(P,D,Q,{s})...")

    for p, d, q, P, D, Q in itertools.product(
        p_values, d_values, q_values, P_values, D_values, Q_values
    ):
        try:
            model = SARIMAX(
                train_series,
                exog=exog_train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=50)
            aic = fit.aic
            bic = fit.bic
            results_grid.append(
                {"order": (p, d, q), "seasonal_order": (P, D, Q, s), "aic": aic, "bic": bic}
            )
            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                best_seasonal = (P, D, Q, s)
                logging.info(f"[GRID] Nuevo mejor: ({p},{d},{q})({P},{D},{Q},{s}) AIC={aic:.2f}")
        except Exception as e:
            continue

    logging.info(f"[GRID] Mejor orden: {best_order}{best_seasonal} AIC={best_aic:.2f}")
    return best_order, best_seasonal, pd.DataFrame(results_grid).sort_values("aic")


def expanding_window_sarimax(grp, order, seasonal_order, exog_cols, min_train=90):
    results = []
    for i in range(min_train, len(grp) - 1):
        train = grp.iloc[:i]
        test = grp.iloc[i : i + 1]
        try:
            model = SARIMAX(
                train["quantity"],
                exog=train[exog_cols].astype(float),
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fit = model.fit(disp=False, maxiter=50)
            forecast = fit.forecast(steps=1, exog=test[exog_cols].astype(float))
            results.append(
                {
                    "sale_date": test["sale_date"].values[0],
                    "sku": test["sku"].values[0],
                    "y_true": test["quantity"].values[0],
                    "y_pred": forecast.values[0],
                    "promotion_flag": test["promotion_flag"].values[0],
                }
            )
        except:
            continue
    return pd.DataFrame(results)


def run_sarimax():
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM analytics.features ORDER BY sku, sale_date", engine)
    df["sale_date"] = pd.to_datetime(df["sale_date"])

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("sarimax")

    exog_cols = ["promotion_flag"]
    PILOT_SKU = "1"
    S = 7  # estacionalidad semanal

    # --- FASE 1: Grid search AIC/BIC sobre SKU piloto ---
    logging.info(f"[SARIMAX] FASE 1: Grid search sobre SKU piloto {PILOT_SKU}")
    pilot = df[df["sku"] == PILOT_SKU].sort_values("sale_date").reset_index(drop=True)
    adf_test(pilot["quantity"], PILOT_SKU)

    train_pilot = pilot.iloc[:90]
    best_order, best_seasonal, grid_results = grid_search_aic(
        train_pilot["quantity"], train_pilot[exog_cols].astype(float), s=S
    )

    grid_results.to_csv("/tmp/sarimax_grid_results.csv", index=False)
    logging.info(f"[SARIMAX] Órdenes seleccionados: {best_order}{best_seasonal}")

    # --- FASE 2: Expanding window sobre todos los SKUs ---
    logging.info("[SARIMAX] FASE 2: Backtesting expanding window todos los SKUs")
    all_results = []
    mape_per_sku = {}

    for sku in sorted(df["sku"].unique()):
        logging.info(f"[SARIMAX] SKU {sku}...")
        grp = df[df["sku"] == sku].sort_values("sale_date").reset_index(drop=True)
        adf_test(grp["quantity"], sku)
        sku_results = expanding_window_sarimax(grp, best_order, best_seasonal, exog_cols)
        if not sku_results.empty:
            sku_mape = mape(sku_results["y_true"], sku_results["y_pred"])
            mape_per_sku[sku] = sku_mape
            logging.info(f"[SARIMAX] SKU {sku} MAPE: {sku_mape:.2f}%")
            all_results.append(sku_results)

    results = pd.concat(all_results).reset_index(drop=True)
    mape_global = mape(results["y_true"], results["y_pred"])
    mape_promo = mape(
        results[results["promotion_flag"] == True]["y_true"],
        results[results["promotion_flag"] == True]["y_pred"],
    )
    mape_nopromo = mape(
        results[results["promotion_flag"] == False]["y_true"],
        results[results["promotion_flag"] == False]["y_pred"],
    )

    logging.info(f"[SARIMAX] MAPE global:   {mape_global:.2f}%")
    logging.info(f"[SARIMAX] MAPE promo:    {mape_promo:.2f}%")
    logging.info(f"[SARIMAX] MAPE no-promo: {mape_nopromo:.2f}%")

    with mlflow.start_run(run_name=f"sarimax_{best_order}_{best_seasonal}"):
        mlflow.log_param("model", "SARIMAX")
        mlflow.log_param("order", str(best_order))
        mlflow.log_param("seasonal_order", str(best_seasonal))
        mlflow.log_param("order_selection", "AIC_grid_search")
        mlflow.log_param("pilot_sku", PILOT_SKU)
        mlflow.log_param("exog", "promotion_flag")
        mlflow.log_param("seasonality_s", S)
        mlflow.log_metric("mape_global", round(mape_global, 4))
        mlflow.log_metric("mape_promo", round(mape_promo, 4))
        mlflow.log_metric("mape_nopromo", round(mape_nopromo, 4))
        for sku, m in mape_per_sku.items():
            mlflow.log_metric(f"mape_sku_{sku}", round(m, 4))
        results.to_csv("/tmp/sarimax_predictions.csv", index=False)
        logging.info("[SARIMAX] Predicciones guardadas")
        logging.info("[SARIMAX] Run registrado en MLflow ✅")


if __name__ == "__main__":
    run_sarimax()
    # Guardar predicciones para análisis de residuos
