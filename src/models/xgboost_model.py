import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import mlflow
import logging
import warnings
import shap
from xgboost import XGBRegressor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"
MLFLOW_URI = "http://localhost:5000"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def add_features(df):
    df = df.copy()
    df = df.sort_values(["sku", "sale_date"]).reset_index(drop=True)

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

    return df


def get_feature_cols():
    return [
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


def run_xgboost():
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM analytics.features ORDER BY sku, sale_date", engine)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = add_features(df)
    FEATURE_COLS = get_feature_cols()
    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    logging.info(f"[XGB] Filas tras feature engineering: {len(df)}")

    TARGET = "quantity"
    MIN_TRAIN = 90

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("xgboost")

    # --- FASE 1: Hyperopt con TimeSeriesSplit k=5 ---
    logging.info("[XGB] FASE 1: Hyperopt con TimeSeriesSplit k=5 (200 iter)...")
    tscv = TimeSeriesSplit(n_splits=5)

    space = {
        "n_estimators": hp.quniform("n_estimators", 200, 1200, 50),
        "max_depth": hp.quniform("max_depth", 3, 9, 1),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.005), np.log(0.3)),
        "subsample": hp.uniform("subsample", 0.5, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": hp.uniform("colsample_bylevel", 0.5, 1.0),
        "min_child_weight": hp.quniform("min_child_weight", 1, 15, 1),
        "gamma": hp.uniform("gamma", 0, 1.0),
        "reg_alpha": hp.loguniform("reg_alpha", np.log(1e-4), np.log(20)),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-4), np.log(20)),
    }

    X_all = df[FEATURE_COLS].values
    y_all = df[TARGET].values
    trials = Trials()
    best_val = [np.inf]

    def objective(params):
        params["n_estimators"] = int(params["n_estimators"])
        params["max_depth"] = int(params["max_depth"])
        params["min_child_weight"] = int(params["min_child_weight"])

        fold_mapes = []
        for train_idx, val_idx in tscv.split(X_all):
            X_tr, X_va = X_all[train_idx], X_all[val_idx]
            y_tr, y_va = y_all[train_idx], y_all[val_idx]
            model = XGBRegressor(
                **params, random_state=RANDOM_SEED, tree_method="hist", verbosity=0
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            preds = model.predict(X_va)
            fold_mapes.append(mape(y_va, preds))

        m = np.mean(fold_mapes)
        if m < best_val[0]:
            best_val[0] = m
            logging.info(f"[HYPEROPT] Nuevo mejor MAPE CV: {m:.4f}%")
        return {"loss": m, "status": STATUS_OK}

    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=200,
        trials=trials,
        rstate=np.random.default_rng(RANDOM_SEED),
    )

    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(best["max_depth"]),
        "learning_rate": best["learning_rate"],
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "colsample_bylevel": best["colsample_bylevel"],
        "min_child_weight": int(best["min_child_weight"]),
        "gamma": best["gamma"],
        "reg_alpha": best["reg_alpha"],
        "reg_lambda": best["reg_lambda"],
    }
    logging.info(f"[XGB] Mejores params: {best_params}")

    # --- FASE 2: Expanding window paso 1 ---
    logging.info("[XGB] FASE 2: Expanding window paso 1 (riguroso)...")
    results = []
    date_list = sorted(df["sale_date"].unique())

    for i in range(MIN_TRAIN, len(date_list) - 1):  # paso 1
        cutoff = date_list[i]
        test_date = date_list[i + 1]
        train = df[df["sale_date"] <= cutoff]
        test = df[df["sale_date"] == test_date]
        if test.empty:
            continue

        model = XGBRegressor(
            **best_params, random_state=RANDOM_SEED, tree_method="hist", verbosity=0
        )
        model.fit(train[FEATURE_COLS], train[TARGET])
        preds = model.predict(test[FEATURE_COLS])

        for i2, (idx, row) in enumerate(test.iterrows()):
            results.append(
                {
                    "sale_date": row["sale_date"],
                    "sku": row["sku"],
                    "y_true": row[TARGET],
                    "y_pred": preds[i2],
                    "promotion_flag": row["promotion_flag"],
                }
            )

    results_df = pd.DataFrame(results)
    mape_global = mape(results_df["y_true"], results_df["y_pred"])
    mape_promo = mape(
        results_df[results_df["promotion_flag"] == 1]["y_true"],
        results_df[results_df["promotion_flag"] == 1]["y_pred"],
    )
    mape_nopromo = mape(
        results_df[results_df["promotion_flag"] == 0]["y_true"],
        results_df[results_df["promotion_flag"] == 0]["y_pred"],
    )
    mape_sku = (
        results_df.groupby("sku")
        .apply(lambda g: mape(g["y_true"], g["y_pred"]))
        .reset_index(name="mape")
    )

    logging.info(f"[XGB] MAPE global:   {mape_global:.2f}%")
    logging.info(f"[XGB] MAPE promo:    {mape_promo:.2f}%")
    logging.info(f"[XGB] MAPE no-promo: {mape_nopromo:.2f}%")
    logging.info(f"\n[XGB] MAPE por SKU:\n{mape_sku.to_string(index=False)}")

    # --- FASE 3: Modelo final + SHAP + Quantile regression ---
    logging.info("[XGB] FASE 3: Modelo final + SHAP + intervalos de confianza...")

    final_model = XGBRegressor(
        **best_params, random_state=RANDOM_SEED, tree_method="hist", verbosity=0
    )
    final_model.fit(df[FEATURE_COLS], df[TARGET])

    # Quantile regression P10/P90
    model_p10 = XGBRegressor(
        **best_params,
        random_state=RANDOM_SEED,
        tree_method="hist",
        verbosity=0,
        objective="reg:quantileerror",
        quantile_alpha=0.1,
    )
    model_p90 = XGBRegressor(
        **best_params,
        random_state=RANDOM_SEED,
        tree_method="hist",
        verbosity=0,
        objective="reg:quantileerror",
        quantile_alpha=0.9,
    )
    model_p10.fit(df[FEATURE_COLS], df[TARGET])
    model_p90.fit(df[FEATURE_COLS], df[TARGET])

    sample_pred = df[FEATURE_COLS].tail(10)
    p50 = final_model.predict(sample_pred)
    p10 = model_p10.predict(sample_pred)
    p90 = model_p90.predict(sample_pred)
    ci_df = pd.DataFrame({"P10": p10, "P50": p50, "P90": p90})
    ci_df.to_csv("/tmp/xgb_confidence_intervals.csv", index=False)
    logging.info(f"[XGB] Intervalos de confianza (últimas 10 filas):\n{ci_df.to_string()}")

    # Feature importance
    importance = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv("/tmp/xgb_feature_importance.csv", index=False)
    logging.info(f"\n[XGB] Feature importance:\n{importance.to_string(index=False)}")

    # SHAP
    logging.info("[XGB] Calculando SHAP values...")
    sample_shap = df[FEATURE_COLS].sample(500, random_state=RANDOM_SEED)
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(sample_shap)
    shap_df = pd.DataFrame(
        np.abs(shap_values).mean(axis=0), index=FEATURE_COLS, columns=["mean_abs_shap"]
    ).sort_values("mean_abs_shap", ascending=False)
    shap_df.to_csv("/tmp/xgb_shap_values.csv")
    logging.info(f"\n[XGB] SHAP mean abs:\n{shap_df.to_string()}")

    # --- MLflow ---
    with mlflow.start_run(run_name="xgboost_riguroso"):
        mlflow.log_params(best_params)
        mlflow.log_param("model", "XGBoost")
        mlflow.log_param("hyperopt_iters", 200)
        mlflow.log_param("cv_strategy", "TimeSeriesSplit_k5")
        mlflow.log_param("expanding_step", 1)
        mlflow.log_param("quantile_regression", True)
        mlflow.log_param("approach", "global_sku_as_feature")
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_metric("mape_global", round(mape_global, 4))
        mlflow.log_metric("mape_promo", round(mape_promo, 4))
        mlflow.log_metric("mape_nopromo", round(mape_nopromo, 4))
        for _, row in mape_sku.iterrows():
            mlflow.log_metric(f"mape_sku_{row['sku']}", round(row["mape"], 4))
        logging.info("[XGB] Run registrado en MLflow ✅")


if __name__ == "__main__":
    run_xgboost()
