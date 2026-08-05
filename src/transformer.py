"""
Silver/Gold layer: clean bronze data and engineer features, then write the
analytics-ready gold table.

Key detail this pipeline gets right that a naive version often misses:
every rolling calculation is computed PER ENGINE (grouped by engine_id).
If you roll a window across the raw, un-grouped table, an engine's first
few readings get blended with the previous engine's last few readings --
a subtle bug that quietly corrupts every downstream feature.
"""
import pandas as pd
import sqlalchemy

DB_URL = "sqlite:///iot_warehouse.db"
ROLLING_WINDOW = 5
ZSCORE_THRESHOLD = 2.5
SENSOR_COLS = ["temperature", "vibration", "pressure", "rpm"]


def transform_data(db_url: str = DB_URL) -> pd.DataFrame:
    engine = sqlalchemy.create_engine(db_url)
    df = pd.read_sql("SELECT * FROM bronze_sensor_logs", con=engine)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    # Fill small sensor gaps within each engine's own series (forward-fill,
    # then back-fill any that are still missing at the very start) instead
    # of dropping rows outright -- dropping mid-series would break the
    # rolling window around the missing point.
    df[SENSOR_COLS] = df.groupby("engine_id")[SENSOR_COLS].transform(lambda s: s.ffill().bfill())
    df = df.dropna(subset=SENSOR_COLS)

    grouped = df.groupby("engine_id")

    df["rolling_avg_temp"] = grouped["temperature"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["rolling_avg_vibration"] = grouped["vibration"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).mean()
    )
    df["rolling_std_vibration"] = grouped["vibration"].transform(
        lambda s: s.rolling(ROLLING_WINDOW, min_periods=1).std()
    ).fillna(0)

    def zscore_flag(s: pd.Series) -> pd.Series:
        """Flag readings that are unusual relative to THIS engine's own history."""
        std = s.std()
        z = (s - s.mean()) / (std if std > 0 else 1)
        return (z.abs() > ZSCORE_THRESHOLD).astype(int)

    df["vibration_anomaly_flag"] = grouped["vibration"].transform(zscore_flag)

    df.to_sql("gold_engine_metrics", con=engine, if_exists="replace", index=False)
    print(
        f"Transformed {len(df)} rows -> gold_engine_metrics "
        f"({int(df['vibration_anomaly_flag'].sum())} anomaly readings flagged)."
    )
    return df


if __name__ == "__main__":
    transform_data()
