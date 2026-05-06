# Celda 1 - Instalacion de dependencias
#!pip install prophet mlflow

# Celda 2 - Imports
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
import mlflow
import warnings
warnings.filterwarnings("ignore")

# Celda 3 - Carga de datos
# Subir el fichero features.csv manualmente en Colab (boton de subir archivo)
from google.colab import files
uploaded = files.upload()

df = pd.read_csv("features.csv")
df["sale_date"] = pd.to_datetime(df["sale_date"])
df = df.sort_values(["sku", "sale_date"]).reset_index(drop=True)

print(f"Datos cargados: {len(df)} filas, {df['sku'].nunique()} SKUs")
print(df.head())

# Celda 4 - Funcion de backtesting con expanding window
def expanding_window_prophet(sku_df):
    """
    Calcula el MAPE mediante backtesting con expanding window.
    Se entrena con un minimo del 60% de los datos y se predice
    de forma iterativa para evitar lookahead bias.
    """
    n = len(sku_df)
    min_train = int(n * 0.6)
    errores = []
    modelo_final = None

    for i in range(min_train, n):
        train = sku_df.iloc[:i][["sale_date", "quantity", "promotion_flag"]].copy()
        test = sku_df.iloc[i:i+1][["sale_date", "quantity", "promotion_flag"]].copy()

        # Prophet requiere columnas ds e y
        train = train.rename(columns={"sale_date": "ds", "quantity": "y"})
        test = test.rename(columns={"sale_date": "ds", "quantity": "y"})

        modelo = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )

        # Añadir promotion_flag como regresor externo
        modelo.add_regressor("promotion_flag")
        modelo.fit(train)

        future = test[["ds", "promotion_flag"]]
        forecast = modelo.predict(future)

        pred = forecast["yhat"].values
        real = test["y"].values
        error = mean_absolute_percentage_error(real, pred) * 100
        errores.append(error)
        modelo_final = modelo

    return np.mean(errores), modelo_final


# Celda 5 - Entrenamiento por SKU
mapes = []
modelo_final = None

for sku in sorted(df["sku"].unique()):
    sku_df = df[df["sku"] == sku].reset_index(drop=True)
    mape, modelo = expanding_window_prophet(sku_df)
    mapes.append(mape)
    modelo_final = modelo
    print(f"SKU {sku} - MAPE: {mape:.2f}%")

global_mape = np.mean(mapes)
print(f"\nMAPE Global Prophet: {global_mape:.2f}%")


# Celda 6 - Registro en MLflow
MLFLOW_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Prophet_Forecasting")

with mlflow.start_run(run_name="Prophet_v1"):
    mlflow.log_param("yearly_seasonality", True)
    mlflow.log_param("weekly_seasonality", True)
    mlflow.log_param("changepoint_prior_scale", 0.05)
    mlflow.log_param("regressors", "promotion_flag")
    mlflow.log_metric("MAPE_global", global_mape)
    print("Resultado registrado en MLflow correctamente.")