-- Minimal schema (PRD section 13 / Technology Stack).
-- Optional for the hackathon MVP -- FastAPI can run with no DB configured;
-- this exists so report history can be added without redesigning storage.

CREATE TABLE IF NOT EXISTS reports (
    id                  SERIAL PRIMARY KEY,
    registration_number TEXT NOT NULL,
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    location_name       TEXT,
    incident_timestamp  TIMESTAMPTZ NOT NULL,
    smoke_confidence    REAL NOT NULL,
    vehicle_confidence  REAL,
    image_path          TEXT,
    report_text         TEXT NOT NULL,
    shared_to_x         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports (created_at DESC);
