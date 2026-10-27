"""Local Flask API for the AML Random Forest pipeline.

Run with: python app.py
The model should be in the same directory as this file.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "aml_random_forest_pipeline.joblib"
DASHBOARD_PATH = "dashboard-aml-fraud-integrated.html"

FEATURES = [
    "hour", "day_of_week", "log_amount_idr", "fx_rate_reference",
    "account_age_days", "customer_tenure_days", "customer_kyc_completeness_score",
    "device_age_days", "merchant_age_days", "merchant_outlet_count",
    "prior_tx_count_1h", "prior_tx_count_24h", "prior_amount_idr_1h",
    "prior_amount_idr_24h", "prior_in_amount_idr_24h", "prior_out_amount_idr_24h",
    "minutes_since_previous_tx", "customer_record_available_at_tx",
    "device_seen_before_tx", "merchant_opened_before_tx", "direction",
    "currency_original", "channel", "transaction_type", "counterparty_country",
    "source_system_code", "account_type", "account_currency", "account_owner_type",
    "customer_occupation", "customer_income_band", "customer_nationality",
    "customer_kyc_status", "customer_status", "device_type", "device_os_family",
    "device_app_version", "device_is_emulator_flag", "device_is_rooted_flag",
    "device_specificity_flag", "merchant_mcc", "merchant_mcc_description",
    "merchant_volume_band", "merchant_ticket_band", "merchant_operating_hours",
    "merchant_mcc_flag",
]

NUMERIC_FEATURES = {
    "hour", "day_of_week", "log_amount_idr", "fx_rate_reference",
    "account_age_days", "customer_tenure_days", "customer_kyc_completeness_score",
    "device_age_days", "merchant_age_days", "merchant_outlet_count",
    "prior_tx_count_1h", "prior_tx_count_24h", "prior_amount_idr_1h",
    "prior_amount_idr_24h", "prior_in_amount_idr_24h", "prior_out_amount_idr_24h",
    "minutes_since_previous_tx", "customer_record_available_at_tx",
    "device_seen_before_tx", "merchant_opened_before_tx",
}

app = Flask(__name__)
model = None
model_load_error: str | None = None


def load_pipeline() -> None:
    """Load the serialized model once; retain any error for the health endpoint."""
    global model, model_load_error
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model tidak ditemukan: {MODEL_PATH.name}")
        model = joblib.load(MODEL_PATH)
        loaded_features = list(getattr(model, "feature_names_in_", []))
        if loaded_features and loaded_features != FEATURES:
            raise RuntimeError(
                "Daftar fitur pipeline tidak cocok dengan skema API. Periksa versi model dan FEATURES di app.py."
            )
        model_load_error = None
    except Exception as exc:  # keep the app running so /api/health can report the issue
        model = None
        model_load_error = f"{type(exc).__name__}: {exc}"


load_pipeline()


def model_metadata() -> dict[str, Any]:
    if model is None:
        return {}
    classifier = getattr(model, "named_steps", {}).get("classifier")
    return {
        "model_type": type(classifier).__name__ if classifier is not None else type(model).__name__,
        "feature_count": len(FEATURES),
        "n_estimators": getattr(classifier, "n_estimators", None),
        "max_depth": getattr(classifier, "max_depth", None),
        "classes": [str(value) for value in getattr(model, "classes_", [])],
    }


@app.get("/")
def dashboard():
    return send_from_directory(BASE_DIR, DASHBOARD_PATH)


@app.get("/api/health")
def health():
    if model is None:
        return jsonify({
            "ok": False,
            "status": "model_unavailable",
            "message": "Model belum berhasil dimuat. Gunakan scikit-learn 1.6.1 sesuai versi saat model disimpan.",
            "details": model_load_error,
        }), 503
    return jsonify({
        "ok": True,
        "status": "model_ready",
        "message": "Pipeline berhasil dimuat dan siap menerima prediksi.",
        "metadata": model_metadata(),
    })


@app.post("/api/predict")
def predict():
    if model is None:
        return jsonify({
            "ok": False,
            "error": "Model belum tersedia. Periksa /api/health dan versi dependensi.",
            "details": model_load_error,
        }), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "Body harus berupa objek JSON."}), 400

    threshold = payload.get("threshold", 0.5)
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "threshold harus berupa angka antara 0 dan 1."}), 400
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        return jsonify({"ok": False, "error": "threshold harus berada di rentang 0 sampai 1."}), 400

    transaction = payload.get("transaction")
    if transaction is None:
        transaction = {key: value for key, value in payload.items() if key != "threshold"}
    if not isinstance(transaction, dict):
        return jsonify({"ok": False, "error": "transaction harus berupa objek JSON."}), 400

    unknown = sorted(set(transaction) - set(FEATURES))
    if unknown:
        return jsonify({
            "ok": False,
            "error": "Ada nama fitur yang tidak dikenali.",
            "unknown_features": unknown,
            "expected_feature_count": len(FEATURES),
        }), 400

    provided = [key for key, value in transaction.items() if key in FEATURES and value is not None and value != ""]
    if not provided:
        return jsonify({"ok": False, "error": "Isi setidaknya satu fitur transaksi sebelum meminta prediksi."}), 400

    row: dict[str, Any] = {}
    for feature in FEATURES:
        value = transaction.get(feature, np.nan)
        if value is None or value == "":
            row[feature] = np.nan
            continue
        if feature in NUMERIC_FEATURES:
            if isinstance(value, bool):
                row[feature] = int(value)
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                return jsonify({"ok": False, "error": f"Fitur '{feature}' harus berupa angka atau null."}), 400
            if not math.isfinite(number):
                return jsonify({"ok": False, "error": f"Fitur '{feature}' harus berupa angka terbatas atau null."}), 400
            row[feature] = number
        else:
            if isinstance(value, (dict, list)):
                return jsonify({"ok": False, "error": f"Fitur '{feature}' harus berupa teks, angka, boolean, atau null."}), 400
            row[feature] = str(value)

    frame = pd.DataFrame([row], columns=FEATURES)
    try:
        probabilities = model.predict_proba(frame)[0]
        classes = list(model.classes_)
        suspicious_index = next((i for i, cls in enumerate(classes) if str(cls) == "1"), None)
        if suspicious_index is None:
            return jsonify({
                "ok": False,
                "error": "Pipeline tidak memiliki kelas target 1. Pastikan label 1 memang berarti SUSPICIOUS.",
            }), 500
        score = float(probabilities[suspicious_index])
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": "Prediksi gagal dijalankan oleh pipeline.",
            "details": f"{type(exc).__name__}: {exc}",
        }), 500

    missing = [feature for feature in FEATURES if feature not in transaction or transaction.get(feature) is None or transaction.get(feature) == ""]
    is_suspicious = score >= threshold
    return jsonify({
        "ok": True,
        "prediction": "SUSPICIOUS" if is_suspicious else "NORMAL",
        "risk_score": round(score, 6),
        "risk_percent": round(score * 100, 2),
        "threshold": threshold,
        "provided_feature_count": len(provided),
        "total_feature_count": len(FEATURES),
        "missing_features": missing,
        "missing_feature_count": len(missing),
        "warning": (
            "Sebagian fitur tidak diberikan dan diisi oleh preprocessing pipeline. "
            "Gunakan semua fitur yang tersedia dan validasi model sebelum memakai skor untuk keputusan operasional."
            if missing else
            "Prediksi berhasil. Validasi performa dan ambang batas model sebelum penggunaan operasional."
        ),
        "model": model_metadata(),
    })


if __name__ == "__main__":
    # Debug is deliberately off; use a production WSGI server for deployment.
    app.run(host="127.0.0.1", port=5000, debug=False)
