"""
Trains a classifier to predict whether an engine will fail within the next
30 operating cycles, using engineered gold-layer features.

Important detail: the train/test split is done by engine_id (a *group*
split), not by row. If you split rows randomly, the model can see other
cycles from the same engine during training and effectively "memorize"
that engine -- giving an unrealistically high test score. Splitting by
engine means the model is judged on engines it has genuinely never seen,
which is the realistic version of this problem.
"""
import joblib
import pandas as pd
import sqlalchemy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score

DB_URL = "sqlite:///iot_warehouse.db"
FEATURE_COLS = [
    "temperature", "vibration", "pressure", "rpm",
    "rolling_avg_temp", "rolling_avg_vibration",
    "rolling_std_vibration", "vibration_anomaly_flag",
]
TARGET_COL = "failure_within_30_cycles"
MODEL_PATH = "model_failure_predictor.joblib"


def train_model(db_url: str = DB_URL):
    engine = sqlalchemy.create_engine(db_url)
    df = pd.read_sql("SELECT * FROM gold_engine_metrics", con=engine)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    groups = df["engine_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("Held-out engines:", sorted(df.loc[test_idx, "engine_id"].unique().tolist()))
    print(classification_report(y_test, preds, target_names=["healthy", "failure_soon"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances.to_string())

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    return model, importances


if __name__ == "__main__":
    train_model()
