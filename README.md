# IoT Telemetry & Predictive Maintenance Pipeline

An end-to-end data engineering + ML pipeline that ingests, cleans, and analyzes
multi-engine IoT sensor telemetry (temperature, vibration, pressure, RPM) to
predict equipment failure before it happens. Built with a medallion
(bronze → silver/gold) architecture.

## Why this project

Jet engine and power systems companies run on exactly this loop: sensors
stream continuous telemetry, that telemetry has to be cleaned and organized
reliably, and the payoff is catching failures before they happen instead of
after. This project is a small, runnable version of that pipeline, built to
demonstrate the data engineering fundamentals (ingestion, ETL, schema design)
alongside the ML layer that sits on top of them.

## Architecture

```
IoT sensors  -->  Bronze layer  -->  Silver/Gold layer  -->  ML model  -->  Dashboard
(raw stream)     (raw storage)      (clean + features)     (failure risk)  (health report)
```

1. **Bronze layer** (`src/ingestion.py`) — loads raw sensor logs into SQLite exactly as they arrived. No cleaning here on purpose: if a downstream step has a bug, you can always replay from bronze without re-collecting data.
2. **Silver/Gold layer** (`src/transformer.py`) — fills small sensor gaps, computes rolling averages **per engine** (grouped, so one engine's history never bleeds into another's), and flags vibration anomalies via a per-engine z-score.
3. **ML model** (`src/model_train.py`) — a `RandomForestClassifier` predicting whether an engine will fail within the next 30 cycles, evaluated on engines it has never seen (a *grouped* train/test split — see Design Decisions below).
4. **Dashboard** (`src/visualize.py`) — generates PNG charts (sensor trends, fleet-wide risk distribution, feature importance) as a lightweight stand-in for a PowerBI dashboard.

## Tech stack

* **Languages:** Python, SQL
* **Libraries:** pandas, NumPy, scikit-learn, SQLAlchemy, matplotlib, joblib
* **Database:** SQLite (swap the connection string for PostgreSQL/SQL Server in production — see `sql/schema.sql`)

## Dataset

`src/generate_data.py` generates a synthetic 12-engine fleet with realistic
degradation curves (sensors drift as an engine approaches failure) plus
injected sensor dropout, out-of-order timestamps, and a few transient
vibration spikes — so the cleaning and anomaly-detection steps have
something real to do.

**To make this project noticeably stronger:** swap the synthetic generator
for [NASA's C-MAPSS Turbofan Engine Degradation Simulation dataset](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/),
real simulated jet engine sensor data built for predicting remaining useful
life. It's about as directly relevant to a jet engine company as a dataset
gets, and using it (with credit) is a strong signal in a portfolio piece.

## Setup & run

```bash
pip install -r requirements.txt

python src/generate_data.py    # creates data/raw_sensor_telemetry.csv
python src/ingestion.py        # raw CSV -> bronze_sensor_logs
python src/transformer.py      # bronze -> gold_engine_metrics
python src/model_train.py      # trains + evaluates the failure model
python src/visualize.py        # writes charts to dashboard/
```

Reference schema (for migrating off SQLite) lives in `sql/schema.sql`.

## Results

On held-out engines (never seen during training):

| Metric | Value |
|---|---|
| Accuracy | 0.96 |
| Precision (failure-soon) | 1.00 |
| Recall (failure-soon) | 0.81 |
| ROC-AUC | 0.997 |

The rolling vibration average and raw vibration are the strongest predictors
— consistent with how the synthetic degradation was generated, which is a
useful sanity check that the pipeline is learning the right signal rather
than an artifact.

## Design decisions

- **Rolling features are computed per engine, not globally.** A naive
  `df['vibration'].rolling(5).mean()` on the full table blends one engine's
  last few readings into the next engine's first few — a subtle bug that
  quietly corrupts every downstream feature. Grouping by `engine_id` before
  rolling fixes this.
- **Train/test split is by engine, not by row.** A random row-level split
  lets the model see other cycles from the same engine it's tested on,
  inflating the score. `GroupShuffleSplit` on `engine_id` gives an honest
  read on how the model performs on equipment it's never encountered.
- **Missing sensor values are forward/back-filled per engine before
  dropping**, not dropped outright — dropping a row mid-series breaks the
  rolling window around the gap.
- **Bronze stays untouched.** Every transformation lives in `transformer.py`,
  so the pipeline can always be replayed from raw data.

## What I'd add next

- Data quality tests (pytest) on the transform logic
- A Dockerfile so the whole pipeline runs with one command
- Structured logging instead of print statements
- A model monitoring step to detect prediction drift over time in production
- Swap the SQLite connection string for a real warehouse (PostgreSQL/SQL Server) using `sql/schema.sql` as the starting DDL
