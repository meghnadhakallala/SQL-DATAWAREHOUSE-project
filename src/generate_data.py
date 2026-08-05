"""
Generates a synthetic multi-engine IoT sensor telemetry dataset that mimics
degradation patterns similar to real turbofan sensor data (loosely inspired
by the structure of NASA's C-MAPSS dataset). Produces
data/raw_sensor_telemetry.csv, which is what src/ingestion.py expects.

In a real project, swap this for an actual telemetry source (a live feed,
an exported log, or a public dataset like C-MAPSS) -- the rest of the
pipeline doesn't care where the CSV came from.
"""
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

np.random.seed(42)

N_ENGINES = 12
OUTPUT_PATH = "data/raw_sensor_telemetry.csv"


def generate_engine_series(engine_id: int, n_cycles: int, start_time: datetime) -> pd.DataFrame:
    """Simulate one engine's life: sensors drift as the engine degrades."""
    cycles = np.arange(1, n_cycles + 1)
    degradation = cycles / n_cycles  # 0 = healthy -> 1 = end of life

    temperature = 550 + 45 * degradation + np.random.normal(0, 3, n_cycles)
    vibration = 0.15 + 1.3 * degradation ** 2 + np.random.normal(0, 0.04, n_cycles)
    pressure = 120 - 16 * degradation + np.random.normal(0, 2, n_cycles)
    rpm = 3000 - 160 * degradation + np.random.normal(0, 15, n_cycles)

    timestamps = [start_time + timedelta(minutes=10 * i) for i in range(n_cycles)]
    remaining_cycles = n_cycles - cycles

    return pd.DataFrame({
        "engine_id": engine_id,
        "cycle": cycles,
        "timestamp": timestamps,
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "rpm": rpm,
        "remaining_cycles": remaining_cycles,
    })


def main():
    os.makedirs("data", exist_ok=True)
    start_time = datetime(2026, 1, 1)

    frames = [
        generate_engine_series(engine_id, np.random.randint(130, 230), start_time)
        for engine_id in range(1, N_ENGINES + 1)
    ]
    df = pd.concat(frames, ignore_index=True)

    # Ground-truth label: will this engine fail within the next 30 cycles?
    df["failure_within_30_cycles"] = (df["remaining_cycles"] <= 30).astype(int)
    df = df.drop(columns=["remaining_cycles"])

    # Simulate real-world sensor dropout (missing readings)
    temp_missing = df.sample(frac=0.02, random_state=1).index
    vib_missing = df.sample(frac=0.015, random_state=2).index
    df.loc[temp_missing, "temperature"] = np.nan
    df.loc[vib_missing, "vibration"] = np.nan

    # Simulate a couple of out-of-order timestamps (justifies cleaning step)
    if len(df) > 51:
        df.loc[[50, 51], "timestamp"] = df.loc[[51, 50], "timestamp"].values

    # Simulate occasional sensor glitches / brief vibration spikes -- the
    # kind of transient event an anomaly detector should actually catch,
    # separate from the slow degradation trend baked in above.
    rng = np.random.default_rng(3)
    for engine_id in df["engine_id"].unique():
        engine_rows = df.index[df["engine_id"] == engine_id]
        n_spikes = rng.integers(1, 3)
        spike_idx = rng.choice(engine_rows, size=min(n_spikes, len(engine_rows)), replace=False)
        df.loc[spike_idx, "vibration"] *= rng.uniform(3.0, 4.5, size=len(spike_idx))

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} rows across {df['engine_id'].nunique()} engines -> {OUTPUT_PATH}")
    print(f"Failure-window positive rate: {df['failure_within_30_cycles'].mean():.2%}")


if __name__ == "__main__":
    main()
