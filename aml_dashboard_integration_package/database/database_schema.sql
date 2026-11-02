CREATE SCHEMA IF NOT EXISTS aml;

CREATE TABLE IF NOT EXISTS aml.active_transactions (
    transaction_id text PRIMARY KEY,
    source_transaction_reference text,
    idempotency_key text,
    source_system_code text,
    account_id text,
    entity_id text,
    counterparty_id text,
    merchant_id text,
    device_id text,
    direction text,
    amount_original text,
    currency_original text,
    amount_idr text,
    fx_rate_reference text,
    value_datetime_utc text,
    business_date text,
    channel text,
    transaction_type text,
    counterparty_country text,
    ip_address text,
    status_from_source text,
    batch_id text,
    late_arrival_flag text,
    lag_days text,
    created_at text,
    synthetic_pattern text,
    scenario_label text
);

CREATE TABLE IF NOT EXISTS aml.model_predictions (
    transaction_id text PRIMARY KEY REFERENCES aml.active_transactions(transaction_id) ON DELETE CASCADE,
    model_version text NOT NULL,
    risk_score double precision NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    prediction text NOT NULL CHECK (prediction IN ('NORMAL', 'SUSPICIOUS')),
    threshold double precision NOT NULL CHECK (threshold >= 0 AND threshold <= 1),
    missing_feature_count integer NOT NULL CHECK (missing_feature_count >= 0),
    predicted_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS model_predictions_risk_idx
    ON aml.model_predictions (risk_score DESC, transaction_id);

CREATE TABLE IF NOT EXISTS aml.dataset_state (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    upload_id text NOT NULL,
    filename text NOT NULL,
    transaction_count bigint NOT NULL CHECK (transaction_count >= 0),
    threshold double precision NOT NULL CHECK (threshold >= 0 AND threshold <= 1),
    model_version text NOT NULL,
    uploaded_at timestamptz NOT NULL DEFAULT now()
);
