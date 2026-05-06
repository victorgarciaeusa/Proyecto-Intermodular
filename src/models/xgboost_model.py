import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
import mlflow
import mlflow.xgboost
import warnings
warnings.filterwarnings("ignore")

# Configuracion de conexion
DB_URL = "postgresql+psycopg2://tfm:tfm1234@127.0.0.1:5432/forecasting"
MLFLOW_URI = "http://127.0.0.1:5000"

# Variables del modelo
FEATURES = ["quantity_lag_1", "quantity_lag_7", "ma_7", "std_7",
            "day_of_week", "month", "is_holiday", "promotion_flag"]
TARGET = "quantity"


def load_features():
    """Carga la tabla analytics.features desde PostgreSQL."""
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM analytics.features", engine)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df = df.sort_values(["sku", "sale_date"]).reset_index(drop=True)
    df = df.dropna(subset=FEATURES)
    return df


def expanding_window_mape(sku_df):
    """
    Calcula el MAPE mediante backtesting con expanding window.
    Se entrena con un minimo del 60% de los datos y se predice
    de forma iterativa para evitar lookahead bias.
    """
    n = len(sku_df)
    min_train = int(n * 0.6)
    errores = []
    modelo = None

    for i in range(min_train, n):
        train = sku_df.iloc[:i]
        test = sku_df.iloc[i:i+1]

        modelo = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        modelo.fit(train[FEATURES], train[TARGET])

        pred = modelo.predict(test[FEATURES])
        error = mean_absolute_percentage_error(test[TARGET], pred) * 100
        errores.append(error)

    return np.mean(errores), modelo


def train_xgboost():
    """
    Entrena el modelo XGBoost sobre analytics.features usando
    expanding window por SKU y registra los resultados en MLflow.
    """
    print("Cargando datos desde analytics.features...")
    df = load_features()
    print(f"Datos cargados: {len(df)} filas, {df['sku'].nunique()} SKUs")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("XGBoost_Forecasting")

    mapes = []
    modelo_final = None

    for sku in sorted(df["sku"].unique()):
        sku_df = df[df["sku"] == sku].reset_index(drop=True)
        mape, modelo = expanding_window_mape(sku_df)
        mapes.append(mape)
        modelo_final = modelo
        print(f"SKU {sku} - MAPE: {mape:.2f}%")

    global_mape = np.mean(mapes)
    print(f"\nMAPE Global XGBoost: {global_mape:.2f}%")

    with mlflow.start_run(run_name="XGBoost_v1"):
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("features", str(FEATURES))
        mlflow.log_metric("MAPE_global", global_mape)
        mlflow.xgboost.log_model(modelo_final, "xgboost_model")
        print("Resultado registrado en MLflow correctamente.")


if __name__ == "__main__":
    train_xgboost()