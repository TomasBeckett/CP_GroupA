import sys
import json

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
import app

client = app.app.test_client()

print("=== 1. Health Check ===")
res = client.get("/api/health")
print("Status:", res.status_code, res.get_json())
assert res.status_code == 200

print("\n=== 2. Auth Config ===")
res = client.get("/api/auth/config")
print("Status:", res.status_code, res.get_json())
assert res.status_code == 200

print("\n=== 3. Login 5 Roles with Role Validation ===")
roles = ["admin", "analyst", "investigator", "supervisor", "auditor"]
tokens = {}
for role in roles:
    email = f"{role}@fcip.id"
    # Login dengan role yang sesuai di database -> berhasil 200
    res = client.post("/api/auth/login", json={"email": email, "password": "Password123!", "role": role})
    data = res.get_json()
    assert res.status_code == 200, f"Login failed for {email}: {data}"
    assert data["user"]["role"] == role
    tokens[role] = data["token"]
    print(f"✓ {role} login OK dengan role matching (name: {data['user']['full_name']})")

# Test role mismatch: akun analyst mencoba login sebagai admin -> harus DITOLAK 403
mismatch_res = client.post("/api/auth/login", json={
    "email": "analyst@fcip.id",
    "password": "Password123!",
    "role": "admin"
})
assert mismatch_res.status_code == 403, f"Expected 403 for role mismatch, got {mismatch_res.status_code}"
assert mismatch_res.get_json().get("code") == "ROLE_MISMATCH"
print("✓ Role mismatch validation verified: analyst dilarang login sebagai admin (403 ROLE_MISMATCH)")

print("\n=== 4. GET /api/auth/me for each role ===")
for role in roles:
    res = client.get("/api/auth/me", headers={"Authorization": f"Bearer {tokens[role]}"})
    data = res.get_json()
    assert res.status_code == 200 and data["user"]["role"] == role
    print(f"✓ /api/auth/me verified for {role}")

print("\n=== 5. Registration & Approval Flow ===")
reg_res = client.post("/api/auth/register", json={
    "email": "new_candidate@fcip.id",
    "password": "Password123!",
    "full_name": "New Candidate",
    "requested_role": "investigator"
})
reg_data = reg_res.get_json()
print("Register status:", reg_res.status_code, reg_data.get("message"))
assert reg_res.status_code == 201
new_user_id = reg_data["user"]["id"]
assert reg_data["user"]["status"] == "pending"

# Attempt login before approval
pre_login = client.post("/api/auth/login", json={"email": "new_candidate@fcip.id", "password": "Password123!"})
print("Pre-approval login status:", pre_login.status_code, pre_login.get_json().get("error"))
assert pre_login.status_code == 403

# Admin approves user
appr_res = client.post(
    f"/api/admin/users/{new_user_id}/approve",
    headers={"Authorization": f"Bearer {tokens['admin']}"},
    json={"role": "investigator"}
)
print("Admin approve status:", appr_res.status_code, appr_res.get_json().get("message"))
assert appr_res.status_code == 200

# Login after approval
post_login = client.post("/api/auth/login", json={"email": "new_candidate@fcip.id", "password": "Password123!"})
assert post_login.status_code == 200
print("Post-approval login OK, role:", post_login.get_json()["user"]["role"])

print("\n=== 6. RBAC Enforcement ===")
# Non-admin access to /api/admin/users
forbidden_res = client.get("/api/admin/users", headers={"Authorization": f"Bearer {tokens['analyst']}"})
print("Analyst accessing /api/admin/users:", forbidden_res.status_code)
assert forbidden_res.status_code == 403

# Admin access to /api/admin/users
admin_users_res = client.get("/api/admin/users", headers={"Authorization": f"Bearer {tokens['admin']}"})
print("Admin accessing /api/admin/users:", admin_users_res.status_code, "Count:", len(admin_users_res.get_json().get("users", [])))
assert admin_users_res.status_code == 200

# Investigator accessing dataset upload
inv_up = client.post("/api/datasets/upload", headers={"Authorization": f"Bearer {tokens['investigator']}"}, json={})
print("Investigator upload dataset:", inv_up.status_code)
assert inv_up.status_code == 403

# Analyst accessing dataset upload
ana_up = client.post("/api/datasets/upload", headers={"Authorization": f"Bearer {tokens['analyst']}"}, json={})
print("Analyst upload dataset (RBAC check passed):", ana_up.status_code)
assert ana_up.status_code != 403  # Expect 400 because no file was supplied in json, but NOT 403

print("\n=== 7. Model Prediction (Partial & Full) ===")
# Partial features with valid AML feature names
partial_pred = client.post("/api/predict", json={
    "transaction": {
        "log_amount_idr": 18.82,
        "hour": 2,
        "day_of_week": 6,
        "fx_rate_reference": 1.0,
        "account_age_days": 540
    }
})
part_data = partial_pred.get_json()
print("Partial predict status:", partial_pred.status_code, "Prediction:", part_data.get("prediction"), "Risk score:", part_data.get("risk_score"))
assert partial_pred.status_code == 200
assert "prediction" in part_data
assert "risk_score" in part_data

print("\n=== 8. Audit Logs ===")
audit_res = client.get("/api/audit-logs", headers={"Authorization": f"Bearer {tokens['auditor']}"})
audit_data = audit_res.get_json()
print("Auditor get audit logs:", audit_res.status_code, "Total logs:", len(audit_data.get("logs", [])))
assert audit_res.status_code == 200
assert len(audit_data.get("logs", [])) > 0

print("\n=== 9. Frontend HTML Integrity ===")
with open(BASE_DIR / "dashboard" / "dashboard-aml-fraud-integrated.html", "r", encoding="utf-8") as f:
    html = f.read()

required_elements = [
    "authShell",
    "authLoginBox",
    "authRegisterBox",
    "authForgotBox",
    "dash-admin",
    "dash-analyst",
    "dash-investigator",
    "dash-supervisor",
    "dash-auditor",
    "view-dashboard",
    "view-alerts",
    "view-alert-detail",
    "view-cases",
    "view-rules",
    "view-operations",
    "view-reporting",
    "view-governance",
    "quickRoleSwitch",
    "topRoleBadge",
    "auditorBanner",
    "adminPendingTableBody",
    "adminRecentAuditBody",
    "auditorLogBody",
    "govPendingUsersBody",
]
for elem in required_elements:
    assert elem in html, f"Missing element {elem} in HTML"
    print(f"✓ Element {elem} verified present")

print(f"HTML size: {len(html)} chars. All views and role modules verified!")
print("\n>>> ALL 9 VERIFICATION TEST SUITES PASSED SUCCESSFULLY! <<<")
