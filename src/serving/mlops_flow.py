import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from prefect import flow, task, get_run_logger

from datetime import datetime, timedelta
import json
import subprocess
import warnings

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
DB_URL = "postgresql://tfm:tfm1234@localhost:5432/forecasting"
MAPE_BASELINE = 7.79
MAPE_REENTRAIN_RATIO = 1.20  # reentrenar si MAPE rolling > 120% baseline
DRIFT_QTY_THRESHOLD = 0.20
DRIFT_PROMO_THRESHOLD = 0.10


# --- TASKS ---


@task(name="cargar_datos", retries=2, retry_delay_seconds=30)
def load_data():
    logger = get_run_logger()
    engine = create_engine(DB_URL)
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
    logger.info(f"Datos cargados: {len(df)} filas, {df['sku'].nunique()} SKUs")
    return df


@task(name="detectar_drift", retries=1)
def detect_drift(df: pd.DataFrame) -> dict:
    logger = get_run_logger()
    cutoff = df["sale_date"].max() - timedelta(days=7)
    recent = df[df["sale_date"] >= cutoff]
    historic = df[df["sale_date"] < cutoff]

    drift_qty = abs(recent["quantity"].mean() - historic["quantity"].mean()) / (
        historic["quantity"].mean() + 1e-9
    )
    drift_promo = abs(recent["promotion_flag"].mean() - historic["promotion_flag"].mean())

    result = {
        "drift_qty": round(float(drift_qty), 4),
        "drift_promo": round(float(drift_promo), 4),
        "drift_qty_alert": drift_qty > DRIFT_QTY_THRESHOLD,
        "drift_promo_alert": drift_promo > DRIFT_PROMO_THRESHOLD,
        "timestamp": datetime.utcnow().isoformat(),
    }

    logger.info(f"Drift quantity: {drift_qty:.2%} (threshold {DRIFT_QTY_THRESHOLD:.0%})")
    logger.info(f"Drift promo:    {drift_promo:.2%} (threshold {DRIFT_PROMO_THRESHOLD:.0%})")

    if result["drift_qty_alert"]:
        logger.warning("⚠️  DRIFT DETECTADO en quantity — reentrenamiento recomendado")
    if result["drift_promo_alert"]:
        logger.warning("⚠️  DRIFT DETECTADO en promotion_flag")

    return result


@task(name="calcular_mape_rolling", retries=1)
def compute_mape_rolling() -> dict:
    logger = get_run_logger()
    engine = create_engine(DB_URL)
    query = """
        SELECT p.sku, p.prediction_date, p.predicted_qty,
               c.quantity as actual, p.model_name
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
            logger.info("MAPE rolling: sin datos históricos")
            return {"status": "no_data"}

        result = {}
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
            needs_retrain = mape > MAPE_BASELINE * MAPE_REENTRAIN_RATIO
            result[model_name] = {"mape_rolling": round(mape, 4), "needs_retrain": needs_retrain}
            status = "⚠️  REENTRENAMIENTO" if needs_retrain else "✅ OK"
            logger.info(f"MAPE rolling {model_name}: {mape:.2f}% [{status}]")

        return result
    except Exception as e:
        logger.warning(f"MAPE rolling error: {e}")
        return {"status": "error", "detail": str(e)}


@task(name="ejecutar_batch_predictor", retries=1, retry_delay_seconds=60)
def run_batch_predictor() -> bool:
    logger = get_run_logger()
    try:
        result = subprocess.run(
            ["python3", "src/serving/batch_predictor.py"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            logger.info("Batch predictor ejecutado correctamente")
            return True
        else:
            logger.error(f"Batch predictor error: {result.stderr[-500:]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error("Batch predictor timeout (300s)")
        return False


@task(name="reentrenar_modelos")
def retrain_models(mape_result: dict, drift_result: dict) -> bool:
    logger = get_run_logger()

    needs_retrain = (
        drift_result.get("drift_qty_alert", False)
        or drift_result.get("drift_promo_alert", False)
        or any(v.get("needs_retrain", False) for v in mape_result.values() if isinstance(v, dict))
    )

    if needs_retrain:
        logger.warning("⚠️  Reentrenamiento necesario — lanzando pipelines de modelos")
        # En producción: lanzar subflows de SARIMAX y XGBoost
        # subprocess.run(["python3", "src/models/sarimax_model.py"])
        # subprocess.run(["python3", "src/models/xgboost_model.py"])
        logger.info("Reentrenamiento completado (simulado en desarrollo)")
        return True
    else:
        logger.info("✅ Modelos en buen estado — no se requiere reentrenamiento")
        return False


@task(name="generar_reporte")
def generate_report(drift_result: dict, mape_result: dict, batch_ok: bool, retrained: bool) -> str:
    logger = get_run_logger()
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "batch_status": "OK" if batch_ok else "ERROR",
        "drift": drift_result,
        "mape_rolling": mape_result,
        "retrained": retrained,
        "overall": "OK" if batch_ok and not drift_result.get("drift_qty_alert") else "WARN",
    }
    report_path = f"/tmp/mlops_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Reporte generado: {report_path}")
    logger.info(f"Estado global: {report['overall']}")
    return report_path


# --- FLOW PRINCIPAL ---


@flow(
    name="forecast_weekly_pipeline",
    description="Pipeline semanal de forecasting: predicción, monitorización y reentrenamiento automático",
)
def weekly_forecast_pipeline():
    logger = get_run_logger()
    logger.info("=== PIPELINE SEMANAL INICIADO ===")

    # 1. Cargar datos
    df = load_data()

    # 2. Detectar drift
    drift_result = detect_drift(df)

    # 3. MAPE rolling
    mape_result = compute_mape_rolling()

    # 4. Ejecutar batch predictor
    batch_ok = run_batch_predictor()

    # 5. Reentrenamiento si necesario
    retrained = retrain_models(mape_result, drift_result)

    # 6. Reporte final
    report_path = generate_report(drift_result, mape_result, batch_ok, retrained)

    logger.info(f"=== PIPELINE COMPLETADO === reporte: {report_path}")
    return report_path


if __name__ == "__main__":
    # Ejecución manual para prueba
    weekly_forecast_pipeline()
