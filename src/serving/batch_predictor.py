import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import json
import logging
import warnings
import time
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"

# Parámetros modelos
SARIMAX_ORDER = (0, 1, 2)
SARIMAX_SEASONAL_ORDER = (0, 1, 1, 7)
XGB_PARAMS = {
    "n_estimators": 1050,
    "max_depth": 4,
    "learning_rate": 0.006508,
    "subsample": 0.7124,
    "colsample_bytree": 0.7665,
    "colsample_bylevel": 0.7336,
    "min_child_weight": 15,
    "gamma": 0.0401,
    "reg_alpha": 0.1552,
    "reg_lambda": 19.676,
    "random_state": RANDOM_SEED,
    "tree_method": "hist",
    "verbosity": 0,
}
ENSEMBLE_WEIGHTS = {"sarimax": 0.55, "xgboost": 0.45}  # ponderación por MAPE inverso
HORIZON_WEEKS = 4


# --- Logging estructurado JSON ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }
        return json.dumps(log)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("batch_predictor")
logger.setLevel(logging.INFO)
logger.addHandler(handler)


# --- Data loading ---
def get_curated_data(engine):
    df = pd.read_sql(
        """
        SELECT sku, sale_date, quantity, promotion_flag
        FROM analytics.sales_curated
        ORDER BY sku, sale_date
    """,
        engine,
    )
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["promotion_flag"] = df["promotion_flag"].astype(int)
    return df


def get_features_data(engine):
    df = pd.read_sql("SELECT * FROM analytics.features ORDER BY sku, sale_date", engine)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    return df


def generate_future_dates(last_date, horizon_days):
    return pd.DataFrame(
        {
            "sale_date": [last_date + timedelta(days=i + 1) for i in range(horizon_days)],
            "promotion_flag": 0,
        }
    )


# --- SARIMAX predictor ---
def predict_sarimax(sku, grp, horizon_days):
    t0 = time.time()
    try:
        model = SARIMAX(
            grp["quantity"],
            exog=grp[["promotion_flag"]].astype(float),
            order=SARIMAX_ORDER,
            seasonal_order=SARIMAX_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=100)
        future = generate_future_dates(grp["sale_date"].max(), horizon_days)
        forecast = fit.get_forecast(
            steps=horizon_days, exog=future[["promotion_flag"]].astype(float)
        )
        mean_pred = forecast.predicted_mean.values
        ci = forecast.conf_int(alpha=0.20)
        elapsed = round(time.time() - t0, 2)
        return mean_pred, ci.iloc[:, 0].values, ci.iloc[:, 1].values, elapsed, None
    except Exception as e:
        return None, None, None, round(time.time() - t0, 2), str(e)


# --- XGBoost predictor ---
def add_xgb_features(df):
    df = df.copy().sort_values(["sku", "sale_date"]).reset_index(drop=True)
    for lag in [1, 7, 14, 21, 28]:
        df[f"quantity_lag_{lag}"] = df.groupby("sku")["quantity"].shift(lag)
    for w in [7, 14, 21, 28]:
        df[f"ma_{w}"] = df.groupby("sku")["quantity"].transform(
            lambda x: x.shift(1).rolling(w).mean()
        )
        df[f"std_{w}"] = df.groupby("sku")["quantity"].transform(
            lambda x: x.shift(1).rolling(w).std()
        )
    df["rolling_max_7"] = df.groupby("sku")["quantity"].transform(
        lambda x: x.shift(1).rolling(7).max()
    )
    df["rolling_min_7"] = df.groupby("sku")["quantity"].transform(
        lambda x: x.shift(1).rolling(7).min()
    )
    df["rolling_range_7"] = df["rolling_max_7"] - df["rolling_min_7"]
    df["lag7_vs_ma28"] = df["quantity_lag_7"] / (df["ma_28"] + 1e-9)
    df["lag1_vs_ma7"] = df["quantity_lag_1"] / (df["ma_7"] + 1e-9)
    df["day_of_week"] = df["sale_date"].dt.dayofweek
    df["day_of_month"] = df["sale_date"].dt.day
    df["week_of_year"] = df["sale_date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["sale_date"].dt.month
    df["quarter"] = df["sale_date"].dt.quarter
    df["is_month_end"] = df["sale_date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["sale_date"].dt.is_month_start.astype(int)
    le = LabelEncoder()
    df["sku_encoded"] = le.fit_transform(df["sku"])
    season_map = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
    df["season_encoded"] = df["sale_date"].dt.month.map(
        lambda m: (
            season_map["winter"]
            if m in [12, 1, 2]
            else (
                season_map["spring"]
                if m in [3, 4, 5]
                else season_map["summer"] if m in [6, 7, 8] else season_map["autumn"]
            )
        )
    )
    import holidays

    es_holidays = holidays.Spain(years=[2022, 2023, 2024])
    df["is_holiday"] = df["sale_date"].dt.date.apply(lambda d: int(d in es_holidays))
    df["promotion_flag"] = df["promotion_flag"].astype(int)
    return df


XGB_FEATURE_COLS = [
    "quantity_lag_1",
    "quantity_lag_7",
    "quantity_lag_14",
    "quantity_lag_21",
    "quantity_lag_28",
    "ma_7",
    "ma_14",
    "ma_21",
    "ma_28",
    "std_7",
    "std_14",
    "std_21",
    "std_28",
    "rolling_max_7",
    "rolling_min_7",
    "rolling_range_7",
    "lag7_vs_ma28",
    "lag1_vs_ma7",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "season_encoded",
    "is_weekend",
    "is_holiday",
    "is_month_end",
    "is_month_start",
    "promotion_flag",
    "sku_encoded",
]


def predict_xgboost(df_train, last_date, horizon_days):
    t0 = time.time()
    try:
        df_feat = add_xgb_features(df_train)
        df_feat = df_feat.dropna(subset=XGB_FEATURE_COLS)

        xgb = XGBRegressor(**XGB_PARAMS)
        xgb.fit(df_feat[XGB_FEATURE_COLS], df_feat["quantity"])

        # Predicción iterativa — usa predicciones anteriores como lags
        history = df_train.copy()
        preds = []
        for i in range(horizon_days):
            future_date = last_date + timedelta(days=i + 1)
            future_row = pd.DataFrame(
                [
                    {
                        "sku": history["sku"].iloc[0],
                        "sale_date": future_date,
                        "quantity": np.nan,
                        "promotion_flag": 0,
                    }
                ]
            )
            combined = pd.concat([history, future_row], ignore_index=True)
            combined = add_xgb_features(combined)
            last_row = combined.iloc[[-1]][XGB_FEATURE_COLS]
            if last_row.isnull().any().any():
                pred = (
                    history["quantity"].iloc[-7]
                    if len(history) >= 7
                    else history["quantity"].mean()
                )
            else:
                pred = float(xgb.predict(last_row)[0])
            preds.append(pred)
            new_row = future_row.copy()
            new_row["quantity"] = pred
            history = pd.concat([history, new_row], ignore_index=True)

        elapsed = round(time.time() - t0, 2)
        return np.array(preds), elapsed, None
    except Exception as e:
        return None, round(time.time() - t0, 2), str(e)


# --- Validación de predicciones ---
def validate_predictions(preds, hist_mean, hist_std, sku):
    alerts = []
    for i, p in enumerate(preds):
        if p < 0:
            alerts.append(f"SKU {sku} día {i+1}: predicción negativa ({p:.2f})")
        if p > hist_mean + 3 * hist_std:
            alerts.append(
                f"SKU {sku} día {i+1}: predicción outlier ({p:.2f} > mean+3std={hist_mean+3*hist_std:.2f})"
            )
        if p < hist_mean - 3 * hist_std:
            alerts.append(
                f"SKU {sku} día {i+1}: predicción outlier ({p:.2f} < mean-3std={hist_mean-3*hist_std:.2f})"
            )
    return alerts


# --- Monitorización ---
def detect_data_drift(engine, df):
    cutoff = df["sale_date"].max() - timedelta(days=7)
    recent = df[df["sale_date"] >= cutoff]
    historic = df[df["sale_date"] < cutoff]

    if recent.empty or historic.empty:
        logger.warning("Drift: datos insuficientes para comparar")
        return

    drift_qty = abs(recent["quantity"].mean() - historic["quantity"].mean()) / (
        historic["quantity"].mean() + 1e-9
    )
    drift_promo = abs(recent["promotion_flag"].mean() - historic["promotion_flag"].mean())

    logger.info(f"Data drift quantity: {drift_qty:.2%} (threshold 20%)")
    logger.info(f"Data drift promo_rate: {drift_promo:.2%} (threshold 10%)")

    if drift_qty > 0.20:
        logger.warning("DRIFT DETECTADO en quantity — reentrenamiento recomendado")
    if drift_promo > 0.10:
        logger.warning("DRIFT DETECTADO en promotion_flag — revisar campaña activa")


def compute_mape_rolling(engine, window_days=30):
    query = f"""
        SELECT p.sku, p.prediction_date, p.predicted_qty, c.quantity as actual,
               p.model_name
        FROM analytics.predictions p
        JOIN analytics.sales_curated c
          ON p.sku = c.sku AND p.prediction_date = c.sale_date
        WHERE p.horizon_week = 1
        ORDER BY p.prediction_date DESC
        LIMIT 300
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            logger.info("MAPE rolling: sin datos históricos de predicciones")
            return
        for model_name, grp in df.groupby("model_name"):
            mask = grp["actual"] != 0
            if mask.sum() == 0:
                continue
            mape = (
                np.mean(
                    np.abs((grp[mask]["actual"] - grp[mask]["predicted_qty"]) / grp[mask]["actual"])
                )
                * 100
            )
            status = "OK" if mape <= 7.79 * 1.2 else "WARN — reentrenamiento recomendado"
            logger.info(f"MAPE rolling {model_name}: {mape:.2f}% vs baseline 7.79% [{status}]")
    except Exception as e:
        logger.warning(f"MAPE rolling error: {e}")


# --- Runner principal ---
def run_batch():
    run_start = time.time()
    logger.info("=== BATCH PREDICTOR INICIADO ===")
    engine = create_engine(DB_URL)

    df_curated = get_curated_data(engine)
    logger.info(f"Datos cargados: {len(df_curated)} filas, {df_curated['sku'].nunique()} SKUs")

    horizon_days = HORIZON_WEEKS * 7
    all_predictions = []
    run_metrics = {
        "skus_ok": 0,
        "skus_error": 0,
        "total_predictions": 0,
        "alerts": [],
        "sku_timing": {},
    }

    for sku, grp in df_curated.groupby("sku"):
        grp = grp.sort_values("sale_date").reset_index(drop=True)
        last_date = grp["sale_date"].max()
        hist_mean = grp["quantity"].mean()
        hist_std = grp["quantity"].std()
        future = generate_future_dates(last_date, horizon_days)

        logger.info(f"Prediciendo SKU {sku} ({len(grp)} obs -> {horizon_days} dias)")

        # SARIMAX
        sarimax_preds, ci_lower, ci_upper, t_sarimax, err_sarimax = predict_sarimax(
            sku, grp, horizon_days
        )

        # XGBoost
        xgb_preds, t_xgb, err_xgb = predict_xgboost(grp.copy(), last_date, horizon_days)

        run_metrics["sku_timing"][sku] = {"sarimax_s": t_sarimax, "xgboost_s": t_xgb}

        if err_sarimax:
            logger.error(f"SARIMAX SKU {sku}: {err_sarimax}")
        if err_xgb:
            logger.error(f"XGBoost SKU {sku}: {err_xgb}")

        if sarimax_preds is None and xgb_preds is None:
            run_metrics["skus_error"] += 1
            continue

        # Ensemble — media ponderada
        if sarimax_preds is not None and xgb_preds is not None:
            ensemble_preds = (
                ENSEMBLE_WEIGHTS["sarimax"] * sarimax_preds
                + ENSEMBLE_WEIGHTS["xgboost"] * xgb_preds
            )
        elif sarimax_preds is not None:
            ensemble_preds = sarimax_preds
        else:
            ensemble_preds = xgb_preds

        # Validación
        alerts = validate_predictions(ensemble_preds, hist_mean, hist_std, sku)
        run_metrics["alerts"].extend(alerts)
        for a in alerts:
            logger.warning(a)

        # Construir registros
        for i in range(horizon_days):
            base = {
                "sku": sku,
                "prediction_date": future["sale_date"].iloc[i].date(),
                "horizon_week": (i // 7) + 1,
                "run_timestamp": datetime.utcnow(),
            }
            # SARIMAX
            if sarimax_preds is not None:
                all_predictions.append(
                    {
                        **base,
                        "predicted_qty": round(float(sarimax_preds[i]), 4),
                        "ci_lower": round(float(ci_lower[i]), 4) if ci_lower is not None else None,
                        "ci_upper": round(float(ci_upper[i]), 4) if ci_upper is not None else None,
                        "model_name": "SARIMAX",
                    }
                )
            # XGBoost
            if xgb_preds is not None:
                all_predictions.append(
                    {
                        **base,
                        "predicted_qty": round(float(xgb_preds[i]), 4),
                        "ci_lower": None,
                        "ci_upper": None,
                        "model_name": "XGBoost",
                    }
                )
            # Ensemble
            all_predictions.append(
                {
                    **base,
                    "predicted_qty": round(float(ensemble_preds[i]), 4),
                    "ci_lower": round(float(ci_lower[i]), 4) if ci_lower is not None else None,
                    "ci_upper": round(float(ci_upper[i]), 4) if ci_upper is not None else None,
                    "model_name": "Ensemble_SARIMAX_XGB",
                }
            )

        run_metrics["skus_ok"] += 1

    # Guardar en BD
    if all_predictions:
        df_preds = pd.DataFrame(all_predictions)
        df_preds.to_sql("predictions", engine, schema="analytics", if_exists="replace", index=False)
        run_metrics["total_predictions"] = len(df_preds)
        logger.info(f"Guardadas {len(df_preds)} predicciones en analytics.predictions")

    # Monitorización
    detect_data_drift(engine, df_curated)
    compute_mape_rolling(engine)

    # Resumen final JSON
    run_metrics["elapsed_total_s"] = round(time.time() - run_start, 2)
    run_metrics["status"] = "OK" if run_metrics["skus_error"] == 0 else "PARTIAL"
    logger.info(f"=== BATCH COMPLETADO === {json.dumps(run_metrics)}")

    # Exportar resumen
    with open("/tmp/batch_run_summary.json", "w") as f:
        json.dump(run_metrics, f, indent=2, default=str)


if __name__ == "__main__":
    run_batch()
