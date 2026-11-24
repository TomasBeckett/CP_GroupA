"""Supabase Authentication & Role-Based Access Control (RBAC) Service.

Provides:
- User profile management (PostgreSQL with in-memory fallback)
- Supabase Auth integration (sign up, sign in, password recovery, JWT verification)
- Role and status checks (admin, analyst, investigator, supervisor, auditor)
- Audit log persistence
- Flask authentication decorators
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import time
from functools import wraps
from typing import Any, Callable
from uuid import uuid4

import jwt
import requests
from flask import jsonify, request

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET", "").strip()
LOCAL_JWT_SECRET = os.environ.get("AML_JWT_SECRET", "fcip-aml-jwt-secret-key-2026-secure-production-ready")

ALLOWED_ROLES = {"admin", "analyst", "investigator", "supervisor", "auditor"}
ALLOWED_STATUSES = {"pending", "active", "suspended", "rejected"}

# Default mock users matching the 5 required roles so demo and testing work immediately
MOCK_USERS: dict[str, dict[str, Any]] = {
    "admin@fcip.id": {
        "id": "00000000-0000-0000-0000-000000000001",
        "email": "admin@fcip.id",
        "password_hash": hashlib.sha256("Password123!".encode()).hexdigest(),
        "full_name": "Administrator FCIP",
        "role": "admin",
        "requested_role": "admin",
        "status": "active",
        "created_at": "2026-01-01T00:00:00Z",
        "approved_at": "2026-01-01T00:00:00Z",
        "approved_by": None,
        "last_login_at": None,
    },
    "analyst@fcip.id": {
        "id": "00000000-0000-0000-0000-000000000002",
        "email": "analyst@fcip.id",
        "password_hash": hashlib.sha256("Password123!".encode()).hexdigest(),
        "full_name": "Budi Santoso",
        "role": "analyst",
        "requested_role": "analyst",
        "status": "active",
        "created_at": "2026-01-01T00:00:00Z",
        "approved_at": "2026-01-01T00:00:00Z",
        "approved_by": "00000000-0000-0000-0000-000000000001",
        "last_login_at": None,
    },
    "investigator@fcip.id": {
        "id": "00000000-0000-0000-0000-000000000003",
        "email": "investigator@fcip.id",
        "password_hash": hashlib.sha256("Password123!".encode()).hexdigest(),
        "full_name": "Andi Saputra",
        "role": "investigator",
        "requested_role": "investigator",
        "status": "active",
        "created_at": "2026-01-01T00:00:00Z",
        "approved_at": "2026-01-01T00:00:00Z",
        "approved_by": "00000000-0000-0000-0000-000000000001",
        "last_login_at": None,
    },
    "supervisor@fcip.id": {
        "id": "00000000-0000-0000-0000-000000000004",
        "email": "supervisor@fcip.id",
        "password_hash": hashlib.sha256("Password123!".encode()).hexdigest(),
        "full_name": "Ratna Dewi",
        "role": "supervisor",
        "requested_role": "supervisor",
        "status": "active",
        "created_at": "2026-01-01T00:00:00Z",
        "approved_at": "2026-01-01T00:00:00Z",
        "approved_by": "00000000-0000-0000-0000-000000000001",
        "last_login_at": None,
    },
    "auditor@fcip.id": {
        "id": "00000000-0000-0000-0000-000000000005",
        "email": "auditor@fcip.id",
        "password_hash": hashlib.sha256("Password123!".encode()).hexdigest(),
        "full_name": "Faisal Rahman",
        "role": "auditor",
        "requested_role": "auditor",
        "status": "active",
        "created_at": "2026-01-01T00:00:00Z",
        "approved_at": "2026-01-01T00:00:00Z",
        "approved_by": "00000000-0000-0000-0000-000000000001",
        "last_login_at": None,
    },
}

MOCK_AUDIT_LOGS: list[dict[str, Any]] = [
    {
        "id": str(uuid4()),
        "actor_email": "system@fcip.id",
        "actor_name": "System Initializer",
        "actor_role": "admin",
        "action": "SYSTEM_INIT",
        "target_type": "system",
        "target_id": "core",
        "details": {"message": "Sistem FCIP Indonesia diinisialisasi dengan konfigurasi RBAC 5 role."},
        "ip_address": "127.0.0.1",
        "created_at": "2026-01-01T00:00:00Z",
    }
]


def is_supabase_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)


def get_auth_config() -> dict[str, Any]:
    return {
        "ok": True,
        "supabase_url": SUPABASE_URL,
        "supabase_anon_key": SUPABASE_ANON_KEY,
        "auth_mode": "supabase" if is_supabase_configured() else "local_demo",
    }


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def generate_jwt(user: dict[str, Any], expires_in_seconds: int = 86400) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user["id"]),
        "email": user["email"],
        "full_name": user.get("full_name", ""),
        "role": user.get("role", "analyst"),
        "status": user.get("status", "pending"),
        "iat": now,
        "exp": now + expires_in_seconds,
    }
    secret = SUPABASE_JWT_SECRET or LOCAL_JWT_SECRET
    return jwt.encode(payload, secret, algorithm="HS256")


def verify_jwt(token: str) -> dict[str, Any]:
    """Verify token using Supabase Auth or local JWT secret."""
    # First, try decoding with local / Supabase secret
    secrets_to_try = [s for s in (SUPABASE_JWT_SECRET, LOCAL_JWT_SECRET) if s]
    for secret in secrets_to_try:
        try:
            return jwt.decode(token, secret, algorithms=["HS256"])
        except (jwt.InvalidTokenError, jwt.ExpiredSignatureError):
            continue

    # If Supabase URL & Anon Key are set, verify with Supabase Auth endpoint
    if is_supabase_configured():
        try:
            resp = requests.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": SUPABASE_ANON_KEY,
                },
                timeout=4,
            )
            if resp.status_code == 200:
                user_data = resp.json()
                meta = user_data.get("user_metadata", {})
                return {
                    "sub": user_data.get("id"),
                    "email": user_data.get("email"),
                    "full_name": meta.get("full_name", ""),
                    "role": meta.get("role", "analyst"),
                    "status": meta.get("status", "active"),
                }
        except Exception:
            pass

    raise jwt.InvalidTokenError("Token tidak valid atau telah kedaluwarsa.")


def get_user_profile(user_id_or_email: str, db_connection_func: Callable | None = None) -> dict[str, Any] | None:
    """Fetch user profile from PostgreSQL if available, else from in-memory store."""
    if db_connection_func:
        try:
            with db_connection_func() as conn:
                row = conn.execute(
                    """
                    SELECT id::text, email, full_name, role, requested_role, status,
                           created_at::text, approved_at::text, approved_by::text, last_login_at::text
                    FROM aml.user_profiles
                    WHERE email = %s OR id::text = %s
                    LIMIT 1
                    """,
                    (user_id_or_email, user_id_or_email),
                ).fetchone()
                if row:
                    return {
                        "id": row[0],
                        "email": row[1],
                        "full_name": row[2],
                        "role": row[3],
                        "requested_role": row[4],
                        "status": row[5],
                        "created_at": row[6],
                        "approved_at": row[7],
                        "approved_by": row[8],
                        "last_login_at": row[9],
                    }
        except Exception:
            pass

    # Check mock users
    email = user_id_or_email.lower().strip()
    if email in MOCK_USERS:
        return MOCK_USERS[email]
    for u in MOCK_USERS.values():
        if u.get("id") == user_id_or_email:
            return u
    return None


def list_users(db_connection_func: Callable | None = None) -> list[dict[str, Any]]:
    """List all user profiles for Admin governance."""
    if db_connection_func:
        try:
            with db_connection_func() as conn:
                rows = conn.execute(
                    """
                    SELECT id::text, email, full_name, role, requested_role, status,
                           created_at::text, approved_at::text, approved_by::text, last_login_at::text
                    FROM aml.user_profiles
                    ORDER BY created_at DESC
                    """
                ).fetchall()
                if rows:
                    return [
                        {
                            "id": r[0],
                            "email": r[1],
                            "full_name": r[2],
                            "role": r[3],
                            "requested_role": r[4],
                            "status": r[5],
                            "created_at": r[6],
                            "approved_at": r[7],
                            "approved_by": r[8],
                            "last_login_at": r[9],
                        }
                        for r in rows
                    ]
        except Exception:
            pass

    return list(MOCK_USERS.values())


def save_user_profile(user: dict[str, Any], db_connection_func: Callable | None = None) -> bool:
    """Save or update user profile in PostgreSQL and in-memory store."""
    email = user["email"].lower().strip()
    MOCK_USERS[email] = {**MOCK_USERS.get(email, {}), **user}

    if db_connection_func:
        try:
            with db_connection_func() as conn:
                with conn.transaction():
                    conn.execute(
                        """
                        INSERT INTO aml.user_profiles
                            (id, email, full_name, role, requested_role, status, approved_at, approved_by, last_login_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                        ON CONFLICT (email) DO UPDATE SET
                            full_name = EXCLUDED.full_name,
                            role = EXCLUDED.role,
                            requested_role = EXCLUDED.requested_role,
                            status = EXCLUDED.status,
                            approved_at = EXCLUDED.approved_at,
                            approved_by = EXCLUDED.approved_by,
                            last_login_at = EXCLUDED.last_login_at,
                            updated_at = now()
                        """,
                        (
                            user["id"],
                            user["email"],
                            user["full_name"],
                            user.get("role", "analyst"),
                            user.get("requested_role"),
                            user.get("status", "pending"),
                            user.get("approved_at"),
                            user.get("approved_by"),
                            user.get("last_login_at"),
                        ),
                    )
                return True
        except Exception:
            return False
    return True


def record_audit(
    action: str,
    target_type: str,
    target_id: str,
    details: dict[str, Any] | None = None,
    actor_email: str = "system",
    actor_name: str = "System",
    actor_role: str = "system",
    ip_address: str = "127.0.0.1",
    db_connection_func: Callable | None = None,
) -> None:
    entry = {
        "id": str(uuid4()),
        "actor_email": actor_email,
        "actor_name": actor_name,
        "actor_role": actor_role,
        "action": action,
        "target_type": target_type,
        "target_id": target_id,
        "details": details or {},
        "ip_address": ip_address,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    MOCK_AUDIT_LOGS.insert(0, entry)

    if db_connection_func:
        try:
            with db_connection_func() as conn:
                with conn.transaction():
                    conn.execute(
                        """
                        INSERT INTO aml.audit_logs
                            (id, actor_email, actor_name, actor_role, action, target_type, target_id, details, ip_address, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
                        """,
                        (
                            entry["id"],
                            actor_email,
                            actor_name,
                            actor_role,
                            action,
                            target_type,
                            target_id,
                            json.dumps(entry["details"]),
                            ip_address,
                        ),
                    )
        except Exception:
            pass


def list_audit_logs(limit: int = 100, db_connection_func: Callable | None = None) -> list[dict[str, Any]]:
    if db_connection_func:
        try:
            with db_connection_func() as conn:
                rows = conn.execute(
                    """
                    SELECT id::text, actor_email, actor_name, actor_role, action,
                           target_type, target_id, details, ip_address, created_at::text
                    FROM aml.audit_logs
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                ).fetchall()
                if rows:
                    return [
                        {
                            "id": r[0],
                            "actor_email": r[1],
                            "actor_name": r[2],
                            "actor_role": r[3],
                            "action": r[4],
                            "target_type": r[5],
                            "target_id": r[6],
                            "details": r[7] if isinstance(r[7], dict) else json.loads(r[7] or "{}"),
                            "ip_address": r[8],
                            "created_at": r[9],
                        }
                        for r in rows
                    ]
        except Exception:
            pass
    return MOCK_AUDIT_LOGS[:limit]


def get_token_from_request() -> str | None:
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    # Also support cookie token if available
    cookie_token = request.cookies.get("fcip_token")
    if cookie_token:
        return cookie_token
    return None


def require_auth(allowed_roles: list[str] | set[str] | None = None, db_conn_func: Callable | None = None):
    """Decorator for role-based endpoint protection."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            token = get_token_from_request()
            if not token:
                return jsonify({
                    "ok": False,
                    "error": "Autentikasi diperlukan. Silakan login terlebih dahulu.",
                    "code": "AUTH_REQUIRED",
                }), 401

            try:
                payload = verify_jwt(token)
            except Exception as exc:
                return jsonify({
                    "ok": False,
                    "error": f"Sesi autentikasi tidak valid atau telah kedaluwarsa ({exc}). Silakan login kembali.",
                    "code": "INVALID_TOKEN",
                }), 401

            email = payload.get("email") or ""
            sub = payload.get("sub") or ""
            user = get_user_profile(sub or email, db_conn_func)
            if not user:
                # Construct temporary user from token claims
                user = {
                    "id": sub,
                    "email": email,
                    "full_name": payload.get("full_name", email),
                    "role": payload.get("role", "analyst"),
                    "status": payload.get("status", "active"),
                }

            # Enforce Account Status Check
            if user.get("status") != "active":
                status = user.get("status", "pending")
                msg = (
                    "Akun Anda masih berstatus PENDING dan menunggu persetujuan Admin."
                    if status == "pending"
                    else "Akun Anda telah dinonaktifkan atau ditolak oleh Admin."
                )
                return jsonify({
                    "ok": False,
                    "error": msg,
                    "status": status,
                    "code": "ACCOUNT_NOT_ACTIVE",
                }), 403

            # Enforce Role-Based Access Control (RBAC)
            if allowed_roles:
                user_role = user.get("role", "analyst")
                if user_role not in allowed_roles:
                    return jsonify({
                        "ok": False,
                        "error": f"Akses ditolak. Tindakan ini memerlukan role: {', '.join(sorted(allowed_roles))}. Role Anda saat ini: {user_role}.",
                        "code": "FORBIDDEN_ROLE",
                        "current_role": user_role,
                        "required_roles": list(allowed_roles),
                    }), 403

            # Inject authenticated user context onto request
            request.current_user = user
            return f(*args, **kwargs)
        return wrapper
    return decorator
