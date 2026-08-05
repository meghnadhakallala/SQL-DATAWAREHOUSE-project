-- ============================================================
-- IoT Predictive Maintenance Pipeline -- Warehouse Schema
-- Written in portable ANSI SQL (validated against SQLite; the
-- same statements run on PostgreSQL and SQL Server with no
-- changes). The Python pipeline creates these tables itself via
-- pandas.to_sql() -- this file exists as reference documentation
-- of the schema, and as a starting point if you migrate off
-- SQLite to a real warehouse.
-- ============================================================

-- BRONZE: raw sensor readings, loaded as-is, no transformation
CREATE TABLE IF NOT EXISTS bronze_sensor_logs (
    engine_id                 INTEGER   NOT NULL,
    cycle                     INTEGER   NOT NULL,
    timestamp                 TIMESTAMP NOT NULL,
    temperature               REAL,
    vibration                 REAL,
    pressure                  REAL,
    rpm                       REAL,
    failure_within_30_cycles  INTEGER
);

-- GOLD: cleaned, feature-engineered, analytics-ready table
CREATE TABLE IF NOT EXISTS gold_engine_metrics (
    engine_id                 INTEGER   NOT NULL,
    cycle                     INTEGER   NOT NULL,
    timestamp                 TIMESTAMP NOT NULL,
    temperature               REAL      NOT NULL,
    vibration                 REAL      NOT NULL,
    pressure                  REAL      NOT NULL,
    rpm                       REAL      NOT NULL,
    rolling_avg_temp          REAL,
    rolling_avg_vibration     REAL,
    rolling_std_vibration     REAL,
    vibration_anomaly_flag    INTEGER,
    failure_within_30_cycles  INTEGER,
    PRIMARY KEY (engine_id, cycle)
);

-- Example gold-layer analytical view: current fleet health snapshot.
-- This is the kind of query a PowerBI / Tableau dashboard would sit on top of.
CREATE VIEW IF NOT EXISTS fleet_health_snapshot AS
SELECT
    engine_id,
    MAX(cycle)                    AS latest_cycle,
    AVG(rolling_avg_vibration)    AS avg_vibration_recent,
    MAX(vibration_anomaly_flag)   AS has_recent_anomaly,
    MAX(failure_within_30_cycles) AS at_risk_flag
FROM gold_engine_metrics
GROUP BY engine_id;
