"""
Generates a small set of PNG charts summarizing fleet health -- a
lightweight stand-in for the PowerBI dashboard described in the README.
Run this after model_train.py so the feature-importance chart has a model
to read from.
"""
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sqlalchemy

DB_URL = "sqlite:///iot_warehouse.db"
OUTPUT_DIR = "dashboard"
MODEL_PATH = "model_failure_predictor.joblib"
FEATURE_COLS = [
    "temperature", "vibration", "pressure", "rpm",
    "rolling_avg_temp", "rolling_avg_vibration",
    "rolling_std_vibration", "vibration_anomaly_flag",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    engine = sqlalchemy.create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM gold_engine_metrics", con=engine)

    # 1. Sensor trend for one representative engine
    sample_id = df["engine_id"].iloc[0]
    sample = df[df["engine_id"] == sample_id]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sample["cycle"], sample["temperature"], label="Temperature")
    ax.plot(sample["cycle"], sample["rolling_avg_vibration"] * 100, label="Vibration x100 (rolling avg)")
    ax.set_title(f"Sensor trend -- engine {sample_id}")
    ax.set_xlabel("Operating cycle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/engine_{sample_id}_sensor_trend.png", dpi=150)
    plt.close(fig)

    # 2. Failure risk distribution across the fleet
    fig, ax = plt.subplots(figsize=(6, 4))
    df["failure_within_30_cycles"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color=["#4C72B0", "#C44E52"]
    )
    ax.set_xticklabels(["Healthy", "Failure-soon"], rotation=0)
    ax.set_title("Fleet-wide failure window distribution")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/failure_distribution.png", dpi=150)
    plt.close(fig)

    # 3. Feature importance, if a trained model is available
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        importances.plot(kind="barh", ax=ax, color="#55A868")
        ax.set_title("Model feature importance")
        fig.tight_layout()
        fig.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
        plt.close(fig)

    print(f"Dashboard charts written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
