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


-- =====================================================================
-- FCIP INDONESIA — SUPABASE AUTHENTICATION & RBAC EXTENSION
-- =====================================================================

CREATE TABLE IF NOT EXISTS aml.user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL UNIQUE,
    full_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'analyst' CHECK (role IN ('admin', 'analyst', 'investigator', 'supervisor', 'auditor')),
    requested_role TEXT CHECK (requested_role IN ('admin', 'analyst', 'investigator', 'supervisor', 'auditor')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'suspended', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    approved_by UUID,
    approved_at TIMESTAMPTZ,
    last_login_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_user_profiles_email ON aml.user_profiles(email);
CREATE INDEX IF NOT EXISTS idx_user_profiles_role ON aml.user_profiles(role);
CREATE INDEX IF NOT EXISTS idx_user_profiles_status ON aml.user_profiles(status);

CREATE TABLE IF NOT EXISTS aml.audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_id TEXT,
    actor_email TEXT,
    actor_name TEXT,
    actor_role TEXT,
    action TEXT NOT NULL,
    target_type TEXT,
    target_id TEXT,
    details JSONB DEFAULT '{}'::jsonb,
    ip_address TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON aml.audit_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_actor ON aml.audit_logs(actor_email);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON aml.audit_logs(action);

INSERT INTO aml.user_profiles (id, email, full_name, role, status, approved_at)
VALUES
    ('00000000-0000-0000-0000-000000000001', 'admin@fcip.id', 'Administrator FCIP', 'admin', 'active', now()),
    ('00000000-0000-0000-0000-000000000002', 'analyst@fcip.id', 'Budi Santoso', 'analyst', 'active', now()),
    ('00000000-0000-0000-0000-000000000003', 'investigator@fcip.id', 'Andi Saputra', 'investigator', 'active', now()),
    ('00000000-0000-0000-0000-000000000004', 'supervisor@fcip.id', 'Ratna Dewi', 'supervisor', 'active', now()),
    ('00000000-0000-0000-0000-000000000005', 'auditor@fcip.id', 'Faisal Rahman', 'auditor', 'active', now())
ON CONFLICT (email) DO UPDATE SET
    role = EXCLUDED.role,
    status = EXCLUDED.status,
    full_name = EXCLUDED.full_name;
