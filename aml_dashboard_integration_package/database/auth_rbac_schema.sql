-- =====================================================================
-- FCIP INDONESIA — AML & FRAUD DETECTION
-- SUPABASE AUTHENTICATION & ROLE-BASED ACCESS CONTROL (RBAC) SCHEMA
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS aml;

-- 1. USER PROFILES TABLE
-- Linked to Supabase auth.users(id) when deployed on Supabase.
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

-- 2. AUDIT LOGS TABLE
-- Persistent compliance audit trail for all sensitive operations.
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

-- 3. CASE INVESTIGATION RECORDS TABLE (Optional database persistence for cases)
CREATE TABLE IF NOT EXISTS aml.investigation_cases (
    case_id TEXT PRIMARY KEY,
    subject TEXT NOT NULL,
    exposure_idr NUMERIC NOT NULL DEFAULT 0,
    investigation_type TEXT NOT NULL,
    transactions_count INT DEFAULT 0,
    entities_count INT DEFAULT 1,
    risk_level TEXT NOT NULL CHECK (risk_level IN ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')),
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN', 'IN PROGRESS', 'ESCALATED', 'PENDING REVIEW', 'CLOSED')),
    assignee_id TEXT,
    assignee_name TEXT,
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at TIMESTAMPTZ
);

-- 4. AUTOMATIC USER SYNC TRIGGER (For Supabase Auth)
-- Automatically inserts an aml.user_profiles row with status 'pending'
-- whenever a new user signs up through Supabase auth.users.
CREATE OR REPLACE FUNCTION aml.handle_new_auth_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO aml.user_profiles (id, email, full_name, requested_role, status, role)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', split_part(NEW.email, '@', 1)),
        COALESCE(NEW.raw_user_meta_data->>'requested_role', 'analyst'),
        'pending',
        -- Default assigned role is 'analyst' once approved; admin will explicitly assign real role
        'analyst'
    )
    ON CONFLICT (id) DO UPDATE SET
        email = EXCLUDED.email,
        full_name = EXCLUDED.full_name,
        updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Bind trigger to auth.users if auth schema exists
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'auth' AND table_name = 'users'
    ) THEN
        DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
        CREATE TRIGGER on_auth_user_created
            AFTER INSERT ON auth.users
            FOR EACH ROW EXECUTE FUNCTION aml.handle_new_auth_user();
    END IF;
END $$;

-- 5. ROW LEVEL SECURITY (RLS) POLICIES
ALTER TABLE aml.user_profiles ENABLE ROW LEVEL SECURITY;
ALTER TABLE aml.audit_logs ENABLE ROW LEVEL SECURITY;

-- User Profiles RLS:
-- Authenticated users can view their own profile:
CREATE POLICY "Users can read own profile"
    ON aml.user_profiles
    FOR SELECT
    TO authenticated
    USING (auth.uid() = id);

-- Admins, Supervisors, and Auditors can view all user profiles:
CREATE POLICY "Privileged roles can read all profiles"
    ON aml.user_profiles
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM aml.user_profiles u
            WHERE u.id = auth.uid()
              AND u.status = 'active'
              AND u.role IN ('admin', 'supervisor', 'auditor')
        )
    );

-- Only Active Admins can update roles and approval status:
CREATE POLICY "Admins can update user profiles"
    ON aml.user_profiles
    FOR UPDATE
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM aml.user_profiles u
            WHERE u.id = auth.uid()
              AND u.status = 'active'
              AND u.role = 'admin'
        )
    );

-- Service role has full access:
CREATE POLICY "Service role full access on user_profiles"
    ON aml.user_profiles
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Audit Logs RLS:
-- Authenticated users can insert audit logs:
CREATE POLICY "Authenticated users can insert audit logs"
    ON aml.audit_logs
    FOR INSERT
    TO authenticated
    WITH CHECK (true);

-- Admins, Supervisors, and Auditors can read audit logs:
CREATE POLICY "Privileged roles can read audit logs"
    ON aml.audit_logs
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM aml.user_profiles u
            WHERE u.id = auth.uid()
              AND u.status = 'active'
              AND u.role IN ('admin', 'supervisor', 'auditor')
        )
    );

-- Service role full access on audit logs:
CREATE POLICY "Service role full access on audit_logs"
    ON aml.audit_logs
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Active Transactions / Model Predictions RLS:
-- Enable RLS for model predictions:
ALTER TABLE aml.model_predictions ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Active users can view model predictions"
    ON aml.model_predictions
    FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM aml.user_profiles u
            WHERE u.id = auth.uid()
              AND u.status = 'active'
        )
    );

CREATE POLICY "Service role full access on model_predictions"
    ON aml.model_predictions
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- 6. DEFAULT SEED DATA (For Initial Setup & Local Testing)
-- Creates default active accounts for each of the 5 roles so testing is immediately possible:
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
