"""
Bronze layer: load raw sensor telemetry into the warehouse exactly as it
arrived. No cleaning, no transformation -- that happens downstream in
transformer.py. Keeping bronze untouched means you can always replay the
pipeline from scratch if a later step has a bug.
"""
import pandas as pd
import sqlalchemy

DB_URL = "sqlite:///iot_warehouse.db"
RAW_CSV_PATH = "data/raw_sensor_telemetry.csv"


def ingest_raw_data(csv_path: str = RAW_CSV_PATH, db_url: str = DB_URL) -> int:
    engine = sqlalchemy.create_engine(db_url)
    df = pd.read_csv(csv_path)

    df.to_sql("bronze_sensor_logs", con=engine, if_exists="replace", index=False)
    print(f"Ingested {len(df)} raw rows into bronze_sensor_logs.")
    return len(df)


if __name__ == "__main__":
    ingest_raw_data()
