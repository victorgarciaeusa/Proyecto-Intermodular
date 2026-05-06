import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import logging
import warnings

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def analyze_residuals(name, y_true, y_pred):
    residuals = np.array(y_true) - np.array(y_pred)

    print(f"\n{'='*50}")
    print(f"MODELO: {name}")
    print(f"{'='*50}")

    # --- Estadísticas básicas ---
    print(f"\n--- Estadísticas residuos ---")
    print(f"  Media:       {residuals.mean():.4f}  (ideal: 0)")
    print(f"  Std:         {residuals.std():.4f}")
    print(f"  Min:         {residuals.min():.4f}")
    print(f"  Max:         {residuals.max():.4f}")
    print(f"  Skewness:    {stats.skew(residuals):.4f}  (ideal: 0)")
    print(f"  Kurtosis:    {stats.kurtosis(residuals):.4f}  (ideal: 0)")

    # --- Normalidad: Shapiro-Wilk ---
    if len(residuals) <= 5000:
        stat_sw, p_sw = stats.shapiro(residuals[:5000])
    else:
        stat_sw, p_sw = stats.shapiro(residuals[:5000])
    normal = "✅ Normal" if p_sw > 0.05 else "⚠️  No normal"
    print(f"\n--- Test Shapiro-Wilk (normalidad) ---")
    print(f"  Statistic: {stat_sw:.4f}  p-value: {p_sw:.6f}  → {normal}")

    # --- Normalidad: KS test ---
    stat_ks, p_ks = stats.kstest(residuals, "norm", args=(residuals.mean(), residuals.std()))
    normal_ks = "✅ Normal" if p_ks > 0.05 else "⚠️  No normal"
    print(f"\n--- Test Kolmogorov-Smirnov (normalidad) ---")
    print(f"  Statistic: {stat_ks:.4f}  p-value: {p_ks:.6f}  → {normal_ks}")

    # --- Autocorrelación: Ljung-Box ---
    lb = acorr_ljungbox(residuals, lags=[7, 14, 28], return_df=True)
    print(f"\n--- Test Ljung-Box (autocorrelación) ---")
    for lag, row in lb.iterrows():
        auto = "✅ Sin autocorr" if row["lb_pvalue"] > 0.05 else "⚠️  Autocorrelación"
        print(f"  Lag {lag:2d}: stat={row['lb_stat']:.4f}  p={row['lb_pvalue']:.6f}  → {auto}")

    # --- Heterocedasticidad: correlación |residuos| vs índice temporal ---
    abs_res = np.abs(residuals)
    tiempo = np.arange(len(abs_res))
    corr, p_hc = stats.pearsonr(tiempo, abs_res)
    hetero = "✅ Homocedastico" if p_hc > 0.05 else "⚠️  Heterocedastico"
    print(f"\n--- Heterocedasticidad (corr |residuos| vs tiempo) ---")
    print(f"  Pearson r: {corr:.4f}  p-value: {p_hc:.6f}  → {hetero}")

    return {
        "model": name,
        "mean_residual": round(residuals.mean(), 4),
        "std_residual": round(residuals.std(), 4),
        "skewness": round(stats.skew(residuals), 4),
        "kurtosis": round(stats.kurtosis(residuals), 4),
        "shapiro_p": round(p_sw, 6),
        "ks_p": round(p_ks, 6),
        "ljungbox_7_p": round(lb.loc[7, "lb_pvalue"], 6),
        "ljungbox_14_p": round(lb.loc[14, "lb_pvalue"], 6),
        "ljungbox_28_p": round(lb.loc[28, "lb_pvalue"], 6),
        "hetero_p": round(p_hc, 6),
    }


def run_analysis():
    # Carga predicciones guardadas
    models = {}

    # Prophet
    try:
        df_prophet = pd.read_csv("/tmp/prophet_predictions.csv")
        models["Prophet"] = (df_prophet["y_true"].values, df_prophet["y_pred"].values)
        logging.info("[RESIDUOS] Prophet predictions cargadas")
    except:
        logging.warning("[RESIDUOS] No se encontraron predicciones de Prophet")

    # Baseline — recalcula desde features
    try:
        engine = create_engine(DB_URL)
        df = pd.read_sql("SELECT * FROM analytics.features ORDER BY sku, sale_date", engine)
        df["sale_date"] = pd.to_datetime(df["sale_date"])
        df = df.dropna(subset=["quantity_lag_7"])
        models["Baseline"] = (df["quantity"].values, df["quantity_lag_7"].values)
        logging.info("[RESIDUOS] Baseline recalculado")
    except Exception as e:
        logging.warning(f"[RESIDUOS] Baseline error: {e}")

    # XGBoost — desde predicciones del expanding window
    try:
        from xgboost import XGBRegressor
        from sklearn.preprocessing import LabelEncoder

        engine = create_engine(DB_URL)
        df = pd.read_sql("SELECT * FROM analytics.features ORDER BY sku, sale_date", engine)
        df["sale_date"] = pd.to_datetime(df["sale_date"])

        # Añadir features igual que en xgboost_model.py
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
        df["day_of_month"] = df["sale_date"].dt.day
        df["week_of_year"] = df["sale_date"].dt.isocalendar().week.astype(int)
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["quarter"] = df["sale_date"].dt.quarter
        df["is_month_end"] = df["sale_date"].dt.is_month_end.astype(int)
        df["is_month_start"] = df["sale_date"].dt.is_month_start.astype(int)
        le = LabelEncoder()
        df["sku_encoded"] = le.fit_transform(df["sku"])
        season_map = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
        df["season_encoded"] = df["season"].map(season_map)
        df["promotion_flag"] = df["promotion_flag"].astype(int)
        df["is_holiday"] = df["is_holiday"].astype(int)

        FEATURE_COLS = [
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
        df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)

        dates = sorted(df["sale_date"].unique())
        cutoff = dates[int(len(dates) * 0.70)]
        train_xgb = df[df["sale_date"] <= cutoff]
        test_xgb = df[df["sale_date"] > cutoff]

        best_params = {
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
        }
        xgb = XGBRegressor(**best_params, random_state=RANDOM_SEED, tree_method="hist", verbosity=0)
        xgb.fit(train_xgb[FEATURE_COLS], train_xgb["quantity"])
        preds_xgb = xgb.predict(test_xgb[FEATURE_COLS])
        models["XGBoost"] = (test_xgb["quantity"].values, preds_xgb)
        logging.info("[RESIDUOS] XGBoost predicciones generadas")
    except Exception as e:
        logging.warning(f"[RESIDUOS] XGBoost error: {e}")

    # SARIMAX
    try:
        df_sarimax = pd.read_csv("/tmp/sarimax_predictions.csv")
        df_sarimax["promotion_flag"] = df_sarimax["promotion_flag"].map(
            {True: 1, False: 0, "True": 1, "False": 0}
        )
        models["SARIMAX"] = (df_sarimax["y_true"].values, df_sarimax["y_pred"].values)
        logging.info("[RESIDUOS] SARIMAX predictions cargadas")
    except Exception as e:
        logging.warning(f"[RESIDUOS] SARIMAX error: {e}")

    # --- Análisis ---
    summary = []
    for name, (y_true, y_pred) in models.items():
        result = analyze_residuals(name, y_true, y_pred)
        summary.append(result)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("/tmp/residuals_summary.csv", index=False)
    print(f"\n{'='*50}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*50}")
    print(summary_df.to_string(index=False))
    logging.info("[RESIDUOS] Análisis completado ✅")


if __name__ == "__main__":
    run_analysis()
