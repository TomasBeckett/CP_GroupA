"""Flask API + live FCIP dashboard backend. Run after train_fcip_model.py."""
from __future__ import annotations
import json
import csv
import re
import uuid
import subprocess
import sys
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from train_fcip_model import FEATURES, NUMERIC, CATEGORICAL, DATA, MODELS


MODEL_PATH = MODELS / "fcip_aml_random_forest_pipeline.joblib"
SCORED_PATH = DATA / "fcip_scored_transactions.csv"
METRICS_PATH = MODELS / "fcip_model_metrics.json"
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DASHBOARD_DIR = PROJECT_ROOT / "Frontend"
DASHBOARD = "dashboard_antigrafity.html"
STAGING_DIR = DATA / "staging_uploads"
REGISTRY_PATH = STAGING_DIR / "dataset_registry.json"
app = Flask(__name__)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

model = None
load_error = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        load_error = f"Model belum ditemukan: {MODEL_PATH}"
except Exception as exc:
    load_error = f"{type(exc).__name__}: {exc}"

_CACHED_SCORED_DF: pd.DataFrame | None = None
_CACHED_MTIME: float = 0.0


def load_scored(force_reload: bool = False) -> pd.DataFrame:
    global _CACHED_SCORED_DF, _CACHED_MTIME
    if not SCORED_PATH.exists():
        raise FileNotFoundError("Skor belum tersedia. Jalankan: python train_fcip_model.py")
    try:
        current_mtime = SCORED_PATH.stat().st_mtime
    except OSError:
        current_mtime = 0.0
    if not force_reload and _CACHED_SCORED_DF is not None and _CACHED_MTIME == current_mtime:
        return _CACHED_SCORED_DF
    df = pd.read_csv(SCORED_PATH, low_memory=False)
    for col in ["risk_score", "risk_percent", "amount_idr"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    _CACHED_SCORED_DF = df
    _CACHED_MTIME = current_mtime
    return _CACHED_SCORED_DF


def metrics_data() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/")
def dashboard():
    return send_from_directory(DASHBOARD_DIR, DASHBOARD)

@app.get("/api/health")
def health():
    if model is None:
        return jsonify({"ok": False, "error": load_error or "Model belum dilatih."}), 503
    try:
        rows = len(load_scored())
    except Exception:
        rows = 0
    return jsonify({"ok": True, "model_type": type(getattr(model, "named_steps", {}).get("classifier", model)).__name__,
                    "model_path": str(MODEL_PATH.name), "feature_count": len(FEATURES),
                    "scored_transaction_rows": rows, "dataset_connected": rows > 0})


@app.get("/api/summary")
def summary():
    try:
        df = load_scored()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    threshold = request.args.get("threshold", default="0.5")
    try:
        threshold = float(threshold)
        if not 0 <= threshold <= 1:
            raise ValueError
    except ValueError:
        return jsonify({"ok": False, "error": "threshold harus 0 sampai 1"}), 400
    suspicious = df["risk_score"].ge(threshold)
    labels = df["source_label"].astype(str).str.upper()

    is_uploaded = df.get("is_uploaded")
    if is_uploaded is None:
        is_uploaded = ~df["transaction_id"].astype(str).str.match(r"^TXN-\d{9}$")
    else:
        is_uploaded = is_uploaded.fillna(False).astype(bool) | (~df["transaction_id"].astype(str).str.match(r"^TXN-\d{9}$"))

    uploaded_count = int(is_uploaded.sum())
    uploaded_suspicious = int((is_uploaded & suspicious).sum())
    uploaded_normal = int((is_uploaded & ~suspicious).sum())

    # Ambil metadata upload terbaru untuk dashboard scope filter
    latest_upload = None
    registry = _read_registry()
    if registry:
        first = registry[0]
        stats = first.get("scoring_stats", {})
        label_summary = first.get("label_summary", "")
        src_susp = 0
        src_norm = 0
        for part in label_summary.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                k = k.strip().upper()
                try:
                    cnt = int(v.strip())
                    if k == "SUSPICIOUS":
                        src_susp = cnt
                    elif k == "NORMAL":
                        src_norm = cnt
                except ValueError:
                    pass

        # Hitung akun unik dengan alert jika ada
        up_rows = df[is_uploaded] if is_uploaded.any() else pd.DataFrame()
        acc_alerts = stats.get("accounts_with_alerts")
        if acc_alerts is None:
            if len(up_rows) and "account_id" in up_rows.columns:
                acc_alerts = int(up_rows.loc[up_rows["risk_score"].ge(threshold), "account_id"].nunique())
            else:
                acc_alerts = stats.get("predicted_suspicious", 0)

        latest_upload = {
            "dataset_id": first.get("dataset_id"),
            "dataset_name": first.get("dataset_name"),
            "filename": first.get("filename"),
            "uploaded_at": first.get("uploaded_at"),
            "total_transactions": first.get("rows", stats.get("total_rows", 0)),
            "predicted_suspicious": stats.get("predicted_suspicious", 0),
            "predicted_normal": stats.get("predicted_normal", 0),
            "high_risk": stats.get("high_risk", 0),
            "medium_risk": stats.get("medium_risk", 0),
            "low_risk": stats.get("low_risk", 0),
            "accounts_with_alerts": acc_alerts,
            "source_suspicious": src_susp,
            "source_normal": src_norm,
        }

    return jsonify({
        "ok": True, "total_transactions": int(len(df)),
        "predicted_suspicious": int(suspicious.sum()),
        "predicted_normal": int((~suspicious).sum()),
        "source_suspicious": int(labels.eq("SUSPICIOUS").sum()),
        "source_normal": int(labels.eq("NORMAL").sum()),
        "edge_cases": int(labels.eq("EDGE_CASE").sum()),
        "accounts_with_alerts": int(df.loc[suspicious, "account_id"].nunique()),
        "high_risk": int(df["risk_score"].ge(0.8).sum()),
        "medium_risk": int(df["risk_score"].ge(0.5).lt(0.8).sum()),
        "low_risk": int(df["risk_score"].lt(0.5).sum()),
        "threshold": threshold,
        "uploaded_transactions": uploaded_count,
        "uploaded_suspicious": uploaded_suspicious,
        "uploaded_normal": uploaded_normal,
        "latest_upload": latest_upload,
        "warning": "Dataset FCIP sintetis. Skor seluruh dataset mencakup baris yang dipakai training (in-sample); evaluasi performa harus mengacu pada metrik held-out test.",
    })


@app.get("/api/alerts")
def alerts():
    try:
        df = load_scored()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    try:
        threshold = float(request.args.get("threshold", "0.5"))
        limit = int(request.args.get("limit", "1000"))
        if not 0 <= threshold <= 1 or not 1 <= limit <= 5000:
            raise ValueError
    except ValueError:
        return jsonify({"ok": False, "error": "threshold 0-1 dan limit 1-5000"}), 400

    is_uploaded = df.get("is_uploaded")
    if is_uploaded is None:
        is_uploaded = ~df["transaction_id"].astype(str).str.match(r"^TXN-\d{9}$")
    else:
        is_uploaded = is_uploaded.fillna(False).astype(bool) | (~df["transaction_id"].astype(str).str.match(r"^TXN-\d{9}$"))

    # Transaksi upload terbaru selalu diprioritaskan di baris paling atas agar langsung terlihat
    uploaded_rows = df[is_uploaded].copy()
    uploaded_rows["is_uploaded"] = True

    # Transaksi historis difilter ambang batas dan diurutkan risiko tertinggi
    historical_rows = df[~is_uploaded & df["risk_score"].ge(threshold)].copy()
    historical_rows["is_uploaded"] = False
    historical_rows = historical_rows.sort_values("risk_score", ascending=False)

    combined = pd.concat([uploaded_rows, historical_rows], ignore_index=True)
    q = combined.head(limit).replace({np.nan: None})
    cols = ["transaction_id", "account_id", "merchant_id", "device_id", "transaction_time", "amount_idr", "direction", "transaction_type", "channel", "source_label", "synthetic_pattern", "risk_score", "risk_percent", "prediction", "is_edge_case", "is_heldout_test_period", "is_uploaded"]
    rows = q[cols].to_dict(orient="records")
    return jsonify({"ok": True, "count": len(rows), "threshold": threshold, "alerts": rows, "uploaded_count": int(len(uploaded_rows))})


@app.get("/api/model-metrics")
def model_metrics():
    data = metrics_data()
    if not data:
        return jsonify({"ok": False, "error": "Metrik belum tersedia. Jalankan training."}), 503
    return jsonify({"ok": True, "metrics": data})




_REF_DATA: dict[str, pd.DataFrame] | None = None


def _get_ref_data() -> dict[str, pd.DataFrame]:
    global _REF_DATA
    if _REF_DATA is not None:
        return _REF_DATA
    ref = {}
    try:
        acc_path = DATA / "FCIP_E04_Account_Wallet_FINAL_15000.csv"
        cust_path = DATA / "customer_10000_fcip_synthetic.csv"
        dev_path = DATA / "FCIP_Device_10000_FINAL.csv"
        mer_path = DATA / "merchant_2000_fcip_synthetic.csv"

        acc = pd.read_csv(acc_path, low_memory=False) if acc_path.exists() else pd.DataFrame()
        cust = pd.read_csv(cust_path, low_memory=False) if cust_path.exists() else pd.DataFrame()
        dev = pd.read_csv(dev_path, low_memory=False) if dev_path.exists() else pd.DataFrame()
        mer = pd.read_csv(mer_path, low_memory=False) if mer_path.exists() else pd.DataFrame()

        for frame, cols in [
            (acc, ["account_id", "customer_id"]),
            (cust, ["customer_id"]),
            (dev, ["device_id"]),
            (mer, ["merchant_id"]),
        ]:
            for col in cols:
                if col in frame.columns:
                    frame[col] = frame[col].astype("string").str.strip()

        ref["accounts"] = acc
        ref["customers"] = cust
        ref["devices"] = dev
        ref["merchants"] = mer
    except Exception as exc:
        print(f"Warning: Gagal memuat tabel referensi FCIP: {exc}")
        ref = {"accounts": pd.DataFrame(), "customers": pd.DataFrame(), "devices": pd.DataFrame(), "merchants": pd.DataFrame()}
    _REF_DATA = ref
    return _REF_DATA


def score_and_integrate_dataset(upload_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Otomatis memproses fitur ML, melakukan scoring dengan Random Forest, dan memperbarui dataset dashboard."""
    global model
    if model is None:
        raise ValueError(load_error or "Model Random Forest belum dimuat atau belum siap.")

    upload = upload_df.copy()
    upload.columns = [str(c).strip() for c in upload.columns]

    if "transaction_id" not in upload.columns or upload["transaction_id"].isna().all():
        upload["transaction_id"] = [f"TX-UPLOAD-{uuid.uuid4().hex[:8].upper()}" for _ in range(len(upload))]
    else:
        upload["transaction_id"] = upload["transaction_id"].astype(str).str.strip()

    if "account_id" not in upload.columns:
        upload["account_id"] = "ACC-UNKNOWN"
    else:
        upload["account_id"] = upload["account_id"].astype(str).str.strip()

    if "merchant_id" not in upload.columns:
        upload["merchant_id"] = "MER-DEFAULT"
    else:
        upload["merchant_id"] = upload["merchant_id"].astype(str).str.strip()

    if "device_id" not in upload.columns:
        upload["device_id"] = "DEV-DEFAULT"
    else:
        upload["device_id"] = upload["device_id"].astype(str).str.strip()

    time_col = None
    for candidate in ["value_datetime_utc", "transaction_time", "transaction_timestamp", "created_at"]:
        if candidate in upload.columns:
            time_col = candidate
            break

    if time_col:
        upload["_tx_time"] = pd.to_datetime(upload[time_col], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        upload["_tx_time"] = pd.NaT

    upload["_tx_time"] = upload["_tx_time"].fillna(datetime.now(timezone.utc).replace(tzinfo=None))

    amt_col = None
    for candidate in ["amount_idr", "amount", "amount_original"]:
        if candidate in upload.columns:
            amt_col = candidate
            break

    if amt_col:
        upload["amount_idr"] = pd.to_numeric(upload[amt_col], errors="coerce").fillna(0.0)
    else:
        upload["amount_idr"] = 0.0

    upload["fx_rate_reference"] = pd.to_numeric(upload.get("fx_rate_reference", 1.0), errors="coerce").fillna(1.0)
    upload["hour"] = upload["_tx_time"].dt.hour.fillna(12).astype(int)
    upload["day_of_week"] = upload["_tx_time"].dt.dayofweek.fillna(0).astype(int)
    upload["log_amount_idr"] = np.log1p(upload["amount_idr"].clip(lower=0))

    already_has_features = all(f in upload.columns for f in FEATURES)
    if already_has_features:
        feat_df = upload.copy()
    else:
        ref = _get_ref_data()
        accounts = ref.get("accounts", pd.DataFrame())
        customers = ref.get("customers", pd.DataFrame())
        devices = ref.get("devices", pd.DataFrame())
        merchants = ref.get("merchants", pd.DataFrame())

        acct_cols = ["account_id", "customer_id", "owner_type", "account_type", "currency", "opened_date"]
        tx = upload.merge(accounts[[c for c in acct_cols if c in accounts.columns]], on="account_id", how="left", suffixes=("", "_account"))
        cust_cols = ["customer_id", "occupation", "income_band", "nationality", "kyc_status", "customer_status", "kyc_completeness_score", "onboarding_date"]
        tx = tx.merge(customers[[c for c in cust_cols if c in customers.columns]], on="customer_id", how="left", suffixes=("", "_customer"))
        tx = tx.merge(devices, on="device_id", how="left", suffixes=("", "_device"))
        tx = tx.merge(merchants, on="merchant_id", how="left", suffixes=("", "_merchant"))

        opened = pd.to_datetime(tx.get("opened_date"), errors="coerce")
        onboarding = pd.to_datetime(tx.get("onboarding_date"), errors="coerce")
        first_seen = pd.to_datetime(tx.get("first_seen"), errors="coerce", utc=True).dt.tz_convert(None)
        merchant_onboarded = pd.to_datetime(tx.get("onboarded_date"), errors="coerce")

        tx["account_age_days"] = (tx["_tx_time"] - opened).dt.total_seconds() / 86400.0
        tx["customer_tenure_days"] = (tx["_tx_time"] - onboarding).dt.total_seconds() / 86400.0
        tx["customer_kyc_completeness_score"] = pd.to_numeric(tx.get("kyc_completeness_score"), errors="coerce")
        tx["device_age_days"] = (tx["_tx_time"] - first_seen).dt.total_seconds() / 86400.0
        tx["merchant_age_days"] = (tx["_tx_time"] - merchant_onboarded).dt.total_seconds() / 86400.0
        tx["merchant_outlet_count"] = pd.to_numeric(tx.get("outlet_count"), errors="coerce")
        tx["customer_record_available_at_tx"] = ((onboarding.notna()) & (onboarding <= tx["_tx_time"])).astype(float)
        tx["device_seen_before_tx"] = ((first_seen.notna()) & (first_seen <= tx["_tx_time"])).astype(float)
        tx["merchant_opened_before_tx"] = ((merchant_onboarded.notna()) & (merchant_onboarded <= tx["_tx_time"])).astype(float)

        for h in ["prior_tx_count_1h", "prior_tx_count_24h", "prior_amount_idr_1h", "prior_amount_idr_24h", "prior_in_amount_idr_24h", "prior_out_amount_idr_24h", "minutes_since_previous_tx"]:
            if h not in tx.columns:
                tx[h] = 0.0

        mapping = {
            "direction": "direction", "currency_original": "currency_original", "channel": "channel",
            "transaction_type": "transaction_type", "counterparty_country": "counterparty_country",
            "source_system_code": "source_system_code", "account_type": "account_type",
            "account_currency": "currency", "account_owner_type": "owner_type",
            "customer_occupation": "occupation", "customer_income_band": "income_band",
            "customer_nationality": "nationality", "customer_kyc_status": "kyc_status",
            "customer_status": "customer_status", "device_type": "device_type",
            "device_os_family": "os_family", "device_app_version": "app_version",
            "device_is_emulator_flag": "is_emulator_flag", "device_is_rooted_flag": "is_rooted_flag",
            "device_specificity_flag": "specificity_flag", "merchant_mcc": "mcc",
            "merchant_mcc_description": "mcc_description", "merchant_volume_band": "declared_expected_volume_band",
            "merchant_ticket_band": "declared_expected_ticket_band", "merchant_operating_hours": "operating_hours_declared",
            "merchant_mcc_flag": "mcc_flag",
        }
        for feat, src in mapping.items():
            if feat not in tx.columns:
                tx[feat] = tx[src] if src in tx.columns else np.nan
        feat_df = tx

    for col in NUMERIC:
        feat_df[col] = pd.to_numeric(feat_df.get(col), errors="coerce")
    for col in CATEGORICAL:
        if col in feat_df.columns:
            feat_df[col] = feat_df[col].where(feat_df[col].notna(), np.nan)
        else:
            feat_df[col] = np.nan

    classes = list(model.classes_)
    suspicious_idx = classes.index(1) if 1 in classes else classes.index(np.int64(1))
    probabilities = model.predict_proba(feat_df[FEATURES])[:, suspicious_idx]

    direction_val = upload.get("direction", "OUT").fillna("OUT").astype(str)
    tx_type_val = upload.get("transaction_type", "TRANSFER").fillna("TRANSFER").astype(str)
    channel_val = upload.get("channel", "MOBILE_APP").fillna("MOBILE_APP").astype(str)
    source_label_val = upload.get("scenario_label", upload.get("source_label", "NEW_UPLOAD")).fillna("NEW_UPLOAD").astype(str)
    pattern_val = upload.get("synthetic_pattern", "BATCH_UPLOAD").fillna("BATCH_UPLOAD").astype(str)

    new_scored = pd.DataFrame({
        "transaction_id": upload["transaction_id"].astype(str),
        "account_id": upload["account_id"].astype(str),
        "merchant_id": upload["merchant_id"].astype(str),
        "device_id": upload["device_id"].astype("string"),
        "transaction_time": upload["_tx_time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "amount_idr": upload["amount_idr"],
        "direction": direction_val,
        "transaction_type": tx_type_val,
        "channel": channel_val,
        "source_label": source_label_val,
        "synthetic_pattern": pattern_val,
        "risk_score": probabilities,
        "risk_percent": np.round(probabilities * 100, 2),
        "prediction": np.where(probabilities >= 0.5, "SUSPICIOUS", "NORMAL"),
        "is_edge_case": source_label_val.str.upper().eq("EDGE_CASE"),
        "is_heldout_test_period": False,
        "is_uploaded": True,
    })

    # Gabungkan baris baru di posisi paling atas agar langsung terlihat di stream & alerts
    existing_df = load_scored()
    combined = pd.concat([new_scored, existing_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["transaction_id"], keep="first")
    
    # Simpan ke CSV utama
    combined.to_csv(SCORED_PATH, index=False)

    global _CACHED_SCORED_DF, _CACHED_MTIME
    _CACHED_SCORED_DF = combined
    try:
        _CACHED_MTIME = SCORED_PATH.stat().st_mtime
    except OSError:
        pass

    susp_mask = new_scored["prediction"] == "SUSPICIOUS"
    stats = {
        "total_rows": int(len(new_scored)),
        "predicted_suspicious": int(susp_mask.sum()),
        "predicted_normal": int((new_scored["prediction"] == "NORMAL").sum()),
        "high_risk": int((new_scored["risk_score"] >= 0.8).sum()),
        "medium_risk": int(((new_scored["risk_score"] >= 0.5) & (new_scored["risk_score"] < 0.8)).sum()),
        "low_risk": int((new_scored["risk_score"] < 0.5).sum()),
        "accounts_with_alerts": int(new_scored.loc[susp_mask, "account_id"].nunique()) if "account_id" in new_scored else int(susp_mask.sum()),
    }
    return new_scored, stats


# Dataset staging endpoints. These validate and store uploads only; they never replace
# the training dataset or retrain/activate a model automatically.
REQUIRED_UPLOAD_COLUMNS = {
    "transaction_id",
    "account_id",
    "merchant_id",
    "device_id",
    "direction",
    "amount_original",
    "currency_original",
    "amount_idr",
    "fx_rate_reference",
    "value_datetime_utc",
    "channel",
    "transaction_type",
    "counterparty_country",
    "source_system_code",
}
ALLOWED_LABELS = {"NORMAL", "SUSPICIOUS", "EDGE_CASE", "FRAUD", "AML", "HIGH_RISK", "LOW_RISK"}


def _read_registry() -> list[dict[str, Any]]:
    if not REGISTRY_PATH.exists():
        return []
    try:
        data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_registry(rows: list[dict[str, Any]]) -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


@app.get("/api/datasets")
def list_datasets():
    return jsonify({"ok": True, "datasets": _read_registry()})


def _update_dataset_registry(dataset_id: str, **updates) -> None:
    registry = _read_registry()
    for item in registry:
        if item.get("dataset_id") == dataset_id:
            item.update(updates)
            break
    _write_registry(registry)


def _process_staged_dataset(dataset_id: str) -> dict[str, Any]:
    """Run the staged processor in a child process so SystemExit cannot kill Flask."""
    processor_path = ROOT.parent / "fcip_upload_processor.py"
    if not processor_path.exists():
        return {"ok": False, "error": f"Processor tidak ditemukan: {processor_path.name}"}
    try:
        completed = subprocess.run(
            [sys.executable, str(processor_path), dataset_id, "--commit"],
            cwd=str(ROOT.parent),
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Pemrosesan melebihi batas 15 menit; periksa status Supabase sebelum mencoba ulang."}
    except Exception as exc:
        return {"ok": False, "error": f"Gagal menjalankan processor: {type(exc).__name__}: {exc}"}

    output = (completed.stdout or "").strip()
    error = (completed.stderr or "").strip()
    ok = completed.returncode == 0
    return {
        "ok": ok,
        "return_code": completed.returncode,
        "output": output[-12000:],
        "error": error[-6000:] if error else None,
    }


@app.post("/api/datasets/process")
def process_dataset_endpoint():
    payload = request.get_json(silent=True) or {}
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        return jsonify({"ok": False, "error": "dataset_id wajib diisi."}), 400
    entry = next((x for x in _read_registry() if x.get("dataset_id") == dataset_id), None)
    if not entry:
        return jsonify({"ok": False, "error": "Dataset tidak ditemukan."}), 404
    if entry.get("processing_status") == "PROCESSED":
        return jsonify({"ok": False, "error": "Dataset ini sudah diproses. Tidak dijalankan ulang untuk mencegah duplikasi."}), 409

    _update_dataset_registry(dataset_id, processing_status="PROCESSING")
    result = _process_staged_dataset(dataset_id)
    _update_dataset_registry(
        dataset_id,
        processing_status="PROCESSED" if result.get("ok") else "FAILED",
        processing_result=result,
        processed_at=datetime.now(timezone.utc).isoformat() if result.get("ok") else None,
    )
    return jsonify({"ok": result.get("ok", False), "dataset_id": dataset_id, "processing": result}), (200 if result.get("ok") else 422)


@app.post("/api/datasets/upload")
def upload_dataset():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"ok": False, "error": "Pilih file CSV atau Excel (.xlsx) terlebih dahulu."}), 400
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        return jsonify({"ok": False, "error": "Format file harus CSV (.csv) atau Excel (.xlsx, .xls)."}), 400
    raw = upload.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        return jsonify({"ok": False, "error": "File kosong."}), 400
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"ok": False, "error": "Ukuran file melebihi batas 50 MB."}), 413
    try:
        from io import BytesIO
        if suffix == ".csv":
            frame = pd.read_csv(BytesIO(raw), low_memory=False)
        else:
            frame = pd.read_excel(BytesIO(raw), engine="openpyxl")
    except Exception as exc:
        return jsonify({"ok": False, "error": f"File tidak dapat dibaca: {exc}"}), 400
    if frame.empty:
        return jsonify({"ok": False, "error": "Dataset tidak memiliki baris data."}), 400

    frame.columns = [str(c).strip() for c in frame.columns]

    has_amount = any(c in frame.columns for c in ["amount_idr", "amount", "amount_original"])
    has_time = any(c in frame.columns for c in ["value_datetime_utc", "transaction_time", "transaction_timestamp", "created_at"])
    if not has_amount or not has_time:
        return jsonify({"ok": False, "error": "Dataset harus memiliki kolom tanggal transaksi (value_datetime_utc / transaction_time) dan nominal (amount_idr / amount)."}), 400

    label_col = "scenario_label" if "scenario_label" in frame.columns else "source_label" if "source_label" in frame.columns else None
    if label_col:
        labels = frame[label_col].astype(str).str.strip().str.upper()
        label_counts = labels.value_counts(dropna=False).to_dict()
        unknown = sorted(set(labels.unique()) - ALLOWED_LABELS - {"NAN", "NONE", ""})
        label_summary = ", ".join(f"{k}: {int(v)}" for k, v in label_counts.items())
        label_warning = f"Label tidak dikenal: {', '.join(unknown)}" if unknown else None
    else:
        label_summary = "Tidak ada kolom label"
        label_warning = None

    time_col = "value_datetime_utc" if "value_datetime_utc" in frame.columns else "transaction_time" if "transaction_time" in frame.columns else None
    dt = pd.to_datetime(frame[time_col], errors="coerce", utc=True) if time_col else pd.Series([pd.Timestamp.now()] * len(frame))
    amount_col = "amount_idr" if "amount_idr" in frame.columns else "amount" if "amount" in frame.columns else "amount_original"
    amount = pd.to_numeric(frame[amount_col], errors="coerce")
    invalid_dt = int(dt.isna().sum())
    invalid_amount = int(amount.isna().sum())

    dataset_name = (request.form.get("dataset_name") or Path(upload.filename).stem).strip()[:80]
    source = (request.form.get("source") or "Upload User").strip()[:120]
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(upload.filename).name)[:100] or f"dataset{suffix}"
    dataset_id = uuid.uuid4().hex[:12]
    stored_name = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{dataset_id}_{safe_name}"
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    target = STAGING_DIR / stored_name
    target.write_bytes(raw)

    # Otomatis lakukan scoring ML pada data baru dan perbarui dataset dashboard
    scoring_error = None
    scoring_stats = {}
    try:
        new_scored, scoring_stats = score_and_integrate_dataset(frame)
    except Exception as exc:
        scoring_error = str(exc)
        print(f"Error scoring upload: {exc}")

    entry = {
        "dataset_id": dataset_id, "dataset_name": dataset_name, "source": source,
        "filename": stored_name, "rows": int(len(frame)), "columns": int(len(frame.columns)),
        "column_names": list(frame.columns), "label_summary": label_summary,
        "invalid_datetime_rows": invalid_dt, "invalid_amount_rows": invalid_amount,
        "label_warning": label_warning, "status": "VALIDATED" if invalid_dt == 0 and invalid_amount == 0 else "NEEDS_REVIEW",
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "note": "Scored with ML Random Forest and integrated into dashboard.",
        "processing_status": "PROCESSED" if not scoring_error else "FAILED",
        "scoring_stats": scoring_stats,
    }

    registry = _read_registry()
    registry.insert(0, entry)
    _write_registry(registry)

    # Coba jalankan sinkronisasi staging processor jika diizinkan
    processing = None
    if request.form.get("auto_process", "true").strip().lower() not in {"0", "false", "no"}:
        try:
            processing = _process_staged_dataset(dataset_id)
            if processing:
                _update_dataset_registry(
                    dataset_id,
                    processing_result=processing,
                    processed_at=datetime.now(timezone.utc).isoformat() if processing.get("ok") else None,
                )
        except Exception as exc:
            processing = {"ok": False, "error": str(exc)}

    susp_count = scoring_stats.get("predicted_suspicious", 0)
    high_count = scoring_stats.get("high_risk", 0)
    msg = (f"Berhasil mengimpor {len(frame):,} transaksi dari {upload.filename}. "
           f"Model ML mendeteksi {susp_count:,} transaksi mencurigakan ({high_count:,} risiko tinggi). "
           f"Dashboard dan Alert Queue langsung diperbarui!")

    return jsonify({
        "ok": True,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "filename": stored_name,
        "rows": int(len(frame)),
        "label_summary": label_summary,
        "status": entry["status"],
        "scoring_stats": scoring_stats,
        "predicted_suspicious": susp_count,
        "predicted_normal": scoring_stats.get("predicted_normal", 0),
        "high_risk": high_count,
        "processing_status": entry["processing_status"],
        "message": msg,
        "scoring_error": scoring_error,
    }), 201


@app.errorhandler(413)
def upload_too_large(_error):
    return jsonify({"ok": False, "error": "Ukuran file melebihi batas 50 MB."}), 413


@app.post("/api/predict")
def predict_one():
    if model is None:
        return jsonify({"ok": False, "error": load_error or "Model belum tersedia"}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "Body harus objek JSON"}), 400
    transaction = payload.get("transaction", payload)
    threshold = payload.get("threshold", 0.5)
    try:
        threshold = float(threshold)
        if not 0 <= threshold <= 1:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "threshold harus 0-1"}), 400
    unknown = sorted(set(transaction) - set(FEATURES))
    if unknown:
        return jsonify({"ok": False, "error": "Fitur tidak dikenal", "unknown_features": unknown}), 400
    row = {}
    for feature in FEATURES:
        value = transaction.get(feature)
        if feature in NUMERIC and value not in (None, ""):
            try:
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": f"{feature} harus angka atau null"}), 400
        elif value == "":
            value = np.nan
        row[feature] = value if value is not None else np.nan
    try:
        classes = list(model.classes_)
        idx = classes.index(1) if 1 in classes else classes.index(np.int64(1))
        score = float(model.predict_proba(pd.DataFrame([row], columns=FEATURES))[0][idx])
    except Exception as exc:
        return jsonify({"ok": False, "error": "Prediksi gagal", "details": str(exc)}), 500
    return jsonify({"ok": True, "prediction": "SUSPICIOUS" if score >= threshold else "NORMAL",
                    "risk_score": round(score, 6), "risk_percent": round(score * 100, 2), "threshold": threshold})



@app.route("/api/dummy/generate", methods=["GET", "POST"])
def api_generate_dummy():
    try:
        from datetime import datetime, timezone
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()
        
        records = [
            {
                'transaction_id': f'TXN-BATCH-{batch_id}-001',
                'source_transaction_reference': f'REF-AML-{batch_id}-001',
                'idempotency_key': f'IDEM-{batch_id}-001',
                'source_system_code': 'PAYMENT_GATEWAY',
                'account_id': 'ACC-000101',
                'entity_id': 'ENT-000101',
                'counterparty_id': 'CP-000101',
                'merchant_id': 'MER-000010',
                'device_id': 'DEV-000010',
                'direction': 'OUT',
                'amount_original': 450000000.0,
                'currency_original': 'IDR',
                'amount_idr': 450000000.0,
                'fx_rate_reference': 1.0,
                'value_datetime_utc': now_iso,
                'business_date': date_str,
                'channel': 'WEB_PORTAL',
                'transaction_type': 'TRANSFER',
                'counterparty_country': 'SG',
                'ip_address': '103.20.10.1',
                'status_from_source': 'SUCCESS',
                'batch_id': f'BATCH-{batch_id}',
                'late_arrival_flag': False,
                'lag_days': 0,
                'created_at': now_iso,
                'synthetic_pattern': 'STRUCTURING',
                'scenario_label': 'SUSPICIOUS'
            },
            {
                'transaction_id': f'TXN-BATCH-{batch_id}-002',
                'source_transaction_reference': f'REF-AML-{batch_id}-002',
                'idempotency_key': f'IDEM-{batch_id}-002',
                'source_system_code': 'WALLET_CORE',
                'account_id': 'ACC-000102',
                'entity_id': 'ENT-000102',
                'counterparty_id': 'CP-000102',
                'merchant_id': 'MER-000011',
                'device_id': 'DEV-000011',
                'direction': 'OUT',
                'amount_original': 320000000.0,
                'currency_original': 'IDR',
                'amount_idr': 320000000.0,
                'fx_rate_reference': 1.0,
                'value_datetime_utc': now_iso,
                'business_date': date_str,
                'channel': 'MOBILE_APP',
                'transaction_type': 'TRANSFER',
                'counterparty_country': 'ID',
                'ip_address': '103.20.10.2',
                'status_from_source': 'SUCCESS',
                'batch_id': f'BATCH-{batch_id}',
                'late_arrival_flag': False,
                'lag_days': 0,
                'created_at': now_iso,
                'synthetic_pattern': 'RAPID_CASHOUT',
                'scenario_label': 'SUSPICIOUS'
            },
            {
                'transaction_id': f'TXN-BATCH-{batch_id}-003',
                'source_transaction_reference': f'REF-AML-{batch_id}-003',
                'idempotency_key': f'IDEM-{batch_id}-003',
                'source_system_code': 'PAYMENT_GATEWAY',
                'account_id': 'ACC-000103',
                'entity_id': 'ENT-000103',
                'counterparty_id': 'CP-000103',
                'merchant_id': 'MER-000012',
                'device_id': 'DEV-000012',
                'direction': 'OUT',
                'amount_original': 125000000.0,
                'currency_original': 'IDR',
                'amount_idr': 125000000.0,
                'fx_rate_reference': 1.0,
                'value_datetime_utc': now_iso,
                'business_date': date_str,
                'channel': 'WEB_PORTAL',
                'transaction_type': 'TRANSFER',
                'counterparty_country': 'ID',
                'ip_address': '103.20.10.3',
                'status_from_source': 'SUCCESS',
                'batch_id': f'BATCH-{batch_id}',
                'late_arrival_flag': False,
                'lag_days': 0,
                'created_at': now_iso,
                'synthetic_pattern': 'U_TURN_TRANSFER',
                'scenario_label': 'SUSPICIOUS'
            },
            {
                'transaction_id': f'TXN-BATCH-{batch_id}-004',
                'source_transaction_reference': f'REF-AML-{batch_id}-004',
                'idempotency_key': f'IDEM-{batch_id}-004',
                'source_system_code': 'WALLET_CORE',
                'account_id': 'ACC-000104',
                'entity_id': 'ENT-000104',
                'counterparty_id': 'CP-000104',
                'merchant_id': 'MER-000013',
                'device_id': 'DEV-000013',
                'direction': 'OUT',
                'amount_original': 88000000.0,
                'currency_original': 'IDR',
                'amount_idr': 88000000.0,
                'fx_rate_reference': 1.0,
                'value_datetime_utc': now_iso,
                'business_date': date_str,
                'channel': 'MOBILE_APP',
                'transaction_type': 'TRANSFER',
                'counterparty_country': 'MY',
                'ip_address': '103.20.10.4',
                'status_from_source': 'SUCCESS',
                'batch_id': f'BATCH-{batch_id}',
                'late_arrival_flag': False,
                'lag_days': 0,
                'created_at': now_iso,
                'synthetic_pattern': 'PEER_TO_PEER_SMURFING',
                'scenario_label': 'SUSPICIOUS'
            },
            {
                'transaction_id': f'TXN-BATCH-{batch_id}-005',
                'source_transaction_reference': f'REF-AML-{batch_id}-005',
                'idempotency_key': f'IDEM-{batch_id}-005',
                'source_system_code': 'PAYMENT_GATEWAY',
                'account_id': 'ACC-000105',
                'entity_id': 'ENT-000105',
                'counterparty_id': 'CP-000105',
                'merchant_id': 'MER-000014',
                'device_id': 'DEV-000014',
                'direction': 'OUT',
                'amount_original': 65000000.0,
                'currency_original': 'IDR',
                'amount_idr': 65000000.0,
                'fx_rate_reference': 1.0,
                'value_datetime_utc': now_iso,
                'business_date': date_str,
                'channel': 'MOBILE_APP',
                'transaction_type': 'TRANSFER',
                'counterparty_country': 'ID',
                'ip_address': '103.20.10.5',
                'status_from_source': 'SUCCESS',
                'batch_id': f'BATCH-{batch_id}',
                'late_arrival_flag': False,
                'lag_days': 0,
                'created_at': now_iso,
                'synthetic_pattern': 'VELOCITY_SPIKE',
                'scenario_label': 'SUSPICIOUS'
            },
            {
                'transaction_id': f'TXN-BATCH-{batch_id}-006',
                'source_transaction_reference': f'REF-AML-{batch_id}-006',
                'idempotency_key': f'IDEM-{batch_id}-006',
                'source_system_code': 'MERCHANT_PLATFORM',
                'account_id': 'ACC-000106',
                'entity_id': 'ENT-000106',
                'counterparty_id': 'CP-000106',
                'merchant_id': 'MER-000015',
                'device_id': 'DEV-000015',
                'direction': 'OUT',
                'amount_original': 350000.0,
                'currency_original': 'IDR',
                'amount_idr': 350000.0,
                'fx_rate_reference': 1.0,
                'value_datetime_utc': now_iso,
                'business_date': date_str,
                'channel': 'POS',
                'transaction_type': 'PAYMENT',
                'counterparty_country': 'ID',
                'ip_address': '182.1.10.11',
                'status_from_source': 'SUCCESS',
                'batch_id': f'BATCH-{batch_id}',
                'late_arrival_flag': False,
                'lag_days': 0,
                'created_at': now_iso,
                'synthetic_pattern': 'NORMAL',
                'scenario_label': 'NORMAL'
            },
            {
                'transaction_id': f'TXN-BATCH-{batch_id}-007',
                'source_transaction_reference': f'REF-AML-{batch_id}-007',
                'idempotency_key': f'IDEM-{batch_id}-007',
                'source_system_code': 'PAYMENT_GATEWAY',
                'account_id': 'ACC-000107',
                'entity_id': 'ENT-000107',
                'counterparty_id': 'CP-000107',
                'merchant_id': 'MER-000016',
                'device_id': 'DEV-000016',
                'direction': 'OUT',
                'amount_original': 1250000.0,
                'currency_original': 'IDR',
                'amount_idr': 1250000.0,
                'fx_rate_reference': 1.0,
                'value_datetime_utc': now_iso,
                'business_date': date_str,
                'channel': 'WEB_PORTAL',
                'transaction_type': 'PAYMENT',
                'counterparty_country': 'ID',
                'ip_address': '182.1.10.12',
                'status_from_source': 'SUCCESS',
                'batch_id': f'BATCH-{batch_id}',
                'late_arrival_flag': False,
                'lag_days': 0,
                'created_at': now_iso,
                'synthetic_pattern': 'NORMAL',
                'scenario_label': 'NORMAL'
            },
            {
                'transaction_id': f'TXN-BATCH-{batch_id}-008',
                'source_transaction_reference': f'REF-AML-{batch_id}-008',
                'idempotency_key': f'IDEM-{batch_id}-008',
                'source_system_code': 'WALLET_CORE',
                'account_id': 'ACC-000108',
                'entity_id': 'ENT-000108',
                'counterparty_id': 'CP-000108',
                'merchant_id': 'MER-000017',
                'device_id': 'DEV-000017',
                'direction': 'OUT',
                'amount_original': 500000.0,
                'currency_original': 'IDR',
                'amount_idr': 500000.0,
                'fx_rate_reference': 1.0,
                'value_datetime_utc': now_iso,
                'business_date': date_str,
                'channel': 'MOBILE_APP',
                'transaction_type': 'TRANSFER',
                'counterparty_country': 'ID',
                'ip_address': '182.1.10.13',
                'status_from_source': 'SUCCESS',
                'batch_id': f'BATCH-{batch_id}',
                'late_arrival_flag': False,
                'lag_days': 0,
                'created_at': now_iso,
                'synthetic_pattern': 'NORMAL',
                'scenario_label': 'NORMAL'
            },
            {
                'transaction_id': f'TXN-BATCH-{batch_id}-009',
                'source_transaction_reference': f'REF-AML-{batch_id}-009',
                'idempotency_key': f'IDEM-{batch_id}-009',
                'source_system_code': 'MERCHANT_PLATFORM',
                'account_id': 'ACC-000109',
                'entity_id': 'ENT-000109',
                'counterparty_id': 'CP-000109',
                'merchant_id': 'MER-000018',
                'device_id': 'DEV-000018',
                'direction': 'OUT',
                'amount_original': 85000.0,
                'currency_original': 'IDR',
                'amount_idr': 85000.0,
                'fx_rate_reference': 1.0,
                'value_datetime_utc': now_iso,
                'business_date': date_str,
                'channel': 'MOBILE_APP',
                'transaction_type': 'PAYMENT',
                'counterparty_country': 'ID',
                'ip_address': '182.1.10.14',
                'status_from_source': 'SUCCESS',
                'batch_id': f'BATCH-{batch_id}',
                'late_arrival_flag': False,
                'lag_days': 0,
                'created_at': now_iso,
                'synthetic_pattern': 'NORMAL',
                'scenario_label': 'NORMAL'
            },
            {
                'transaction_id': f'TXN-BATCH-{batch_id}-010',
                'source_transaction_reference': f'REF-AML-{batch_id}-010',
                'idempotency_key': f'IDEM-{batch_id}-010',
                'source_system_code': 'PAYMENT_GATEWAY',
                'account_id': 'ACC-000110',
                'entity_id': 'ENT-000110',
                'counterparty_id': 'CP-000110',
                'merchant_id': 'MER-000019',
                'device_id': 'DEV-000019',
                'direction': 'IN',
                'amount_original': 12000000.0,
                'currency_original': 'IDR',
                'amount_idr': 12000000.0,
                'fx_rate_reference': 1.0,
                'value_datetime_utc': now_iso,
                'business_date': date_str,
                'channel': 'WEB_PORTAL',
                'transaction_type': 'TRANSFER',
                'counterparty_country': 'ID',
                'ip_address': '182.1.10.15',
                'status_from_source': 'SUCCESS',
                'batch_id': f'BATCH-{batch_id}',
                'late_arrival_flag': False,
                'lag_days': 0,
                'created_at': now_iso,
                'synthetic_pattern': 'NORMAL',
                'scenario_label': 'NORMAL'
            }
        ]
        dummy_df = pd.DataFrame(records)
        default_csv = DATA / "dummy_transactions_siap_upload.csv"
        default_xlsx = DATA / "dummy_transactions_siap_upload.xlsx"
        dummy_df.to_csv(default_csv, index=False)
        dummy_df.to_excel(default_xlsx, index=False, engine="openpyxl")
        
        return jsonify({
            "ok": True,
            "batch_id": f"BATCH-{batch_id}",
            "filename_csv": default_csv.name,
            "filename_xlsx": default_xlsx.name,
            "rows": len(dummy_df),
            "message": f"Berhasil membuat batch baru BATCH-{batch_id} dengan 10 ID unik. File dummy_transactions_siap_upload.csv diperbarui dan siap diunggah!"
        })
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
