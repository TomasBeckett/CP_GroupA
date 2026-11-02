"""Local Flask API for the AML Random Forest pipeline.

Run with: python app.py
The model should be in the same directory as this file.
"""
from __future__ import annotations

import hmac
import math
import os
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR.parent / ".env")
MODEL_PATH = Path(os.environ.get(
    "AML_MODEL_PATH",
    BASE_DIR / "models" / "aml_random_forest_pipeline_trained.joblib",
)).expanduser().resolve()
DASHBOARD_PATH = "dashboard-aml-fraud-integrated.html"
DASHBOARD_DIR = BASE_DIR / "dashboard"
DATA_DIR = Path(
    os.environ.get("AML_DATA_DIR", BASE_DIR.parent / "data")
).expanduser().resolve()
DB_CONFIG = {
    "host": os.environ.get("AML_DB_HOST", "localhost"),
    "port": int(os.environ.get("AML_DB_PORT", "5432")),
    "dbname": os.environ.get("AML_DB_NAME", "aml_fcip"),
    "user": os.environ.get("AML_DB_USER", "postgres"),
    "password": os.environ.get("AML_DB_PASSWORD", ""),
    "sslmode": os.environ.get("AML_DB_SSLMODE", "prefer"),
}
DB_CONFIGURED = bool(DB_CONFIG["password"])
APP_USERNAME = os.environ.get("AML_APP_USERNAME", "")
APP_PASSWORD = os.environ.get("AML_APP_PASSWORD", "")
ACTIVE_TRANSACTION_COLUMNS = [
    "transaction_id", "source_transaction_reference", "idempotency_key",
    "source_system_code", "account_id", "entity_id", "counterparty_id",
    "merchant_id", "device_id", "direction", "amount_original",
    "currency_original", "amount_idr", "fx_rate_reference",
    "value_datetime_utc", "business_date", "channel", "transaction_type",
    "counterparty_country", "ip_address", "status_from_source", "batch_id",
    "late_arrival_flag", "lag_days", "created_at", "synthetic_pattern",
    "scenario_label",
]
REQUIRED_UPLOAD_COLUMNS = {
    "transaction_id", "account_id", "entity_id", "merchant_id", "device_id",
    "source_system_code", "direction", "amount_idr", "currency_original",
    "fx_rate_reference", "value_datetime_utc", "channel", "transaction_type",
    "counterparty_country",
}
MAX_UPLOAD_BYTES = 250 * 1024 * 1024

DATA_FILES = {
    "transactions": "FCIP_Transaction_350000_FK_FIXED.csv",
    "accounts": "FCIP_E04_Account_Wallet_FINAL_15000.csv",
    "customers": "customer_10000_fcip_synthetic.csv",
    "devices": "FCIP_Device_10000_FINAL.csv",
    "merchants": "merchant_2000_fcip_synthetic.csv",
}

DATA_COLUMNS = {
    "transactions": {
        "transaction_id", "source_system_code", "account_id", "entity_id",
        "merchant_id", "device_id", "direction", "currency_original",
        "amount_idr", "fx_rate_reference", "value_datetime_utc", "channel",
        "transaction_type", "counterparty_country",
    },
    "accounts": {
        "account_id", "customer_id", "owner_type", "account_type",
        "currency", "opened_date",
    },
    "customers": {
        "customer_id", "resolved_entity_id", "onboarding_date", "ingested_at",
        "occupation", "income_band", "nationality", "kyc_status",
        "kyc_completeness_score", "customer_status",
    },
    "devices": {
        "device_id", "device_type", "os_family", "app_version", "first_seen",
        "is_emulator_flag", "is_rooted_flag", "specificity_flag",
    },
    "merchants": {
        "merchant_id", "onboarded_date", "mcc", "mcc_description",
        "declared_expected_volume_band", "declared_expected_ticket_band",
        "operating_hours_declared", "outlet_count", "mcc_flag",
    },
}

FEATURE_ENGINEERING_NOTES = [
    "log_amount_idr dihitung dengan log1p(amount_idr) untuk nilai non-negatif.",
    "hour dan day_of_week dihitung dari value_datetime_utc dalam zona waktu UTC.",
    "Fitur umur menggunakan selisih waktu transaksi terhadap tanggal referensi CSV.",
    "Fitur prior dihitung per account_id dari transaksi bertimestamp lebih awal; transaksi dengan waktu sama tidak saling menghitung.",
    "customer_record_available_at_tx memakai ingested_at; device_seen_before_tx memakai first_seen; merchant_opened_before_tx memakai onboarded_date.",
    "Dataset tidak menyediakan semua 46 fitur maupun dokumentasi transformasi training; validasi fitur dan performa model tetap diperlukan.",
]

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
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
model = None
model_load_error: str | None = None
_dataset_lock = threading.RLock()
_dataset_cache: pd.DataFrame | None = None
_dataset_cache_signature: tuple[tuple[str, int, str], ...] | None = None


@app.before_request
def require_app_auth():
    if not APP_USERNAME and not APP_PASSWORD:
        return None
    credentials = request.authorization
    if (
        not APP_USERNAME
        or not APP_PASSWORD
        or credentials is None
        or credentials.type.lower() != "basic"
        or not hmac.compare_digest(credentials.username or "", APP_USERNAME)
        or not hmac.compare_digest(credentials.password or "", APP_PASSWORD)
    ):
        response = jsonify({
            "ok": False,
            "error": "Login diperlukan untuk mengakses dashboard.",
        })
        response.status_code = 401
        response.headers["WWW-Authenticate"] = 'Basic realm="AML Dashboard", charset="UTF-8"'
        return response
    return None


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
        "positive_class": "POTENTIAL_FRAUD_PROXY",
        "positive_class_source_label": "SUSPICIOUS",
        "positive_class_interpretation": (
            "Indikasi potensi fraud berdasarkan label sintetis SUSPICIOUS; "
            "bukan fraud terkonfirmasi."
        ),
    }


def dataset_inventory() -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for key, filename in DATA_FILES.items():
        path = DATA_DIR / filename
        available = path.is_file()
        details: dict[str, Any] = {
            "filename": filename,
            "available": available,
        }
        if available:
            stat = path.stat()
            details["size_bytes"] = stat.st_size
            details["modified_at"] = pd.Timestamp(stat.st_mtime, unit="s", tz="UTC").isoformat()
        files[key] = details
    missing = [entry["filename"] for entry in files.values() if not entry["available"]]
    return {
        "ok": not missing,
        "data_directory": str(DATA_DIR),
        "files": files,
        "missing_files": missing,
        "loaded": _dataset_cache is not None,
    }


def get_database_connection():
    if not DB_CONFIGURED:
        raise RuntimeError(
            "Koneksi database belum dikonfigurasi. Atur AML_DB_PASSWORD pada environment backend."
        )
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "Driver PostgreSQL belum terpasang. Jalankan pip install -r requirements.txt."
        ) from exc
    return psycopg.connect(**DB_CONFIG, connect_timeout=5)


def _read_reference_table(connection: Any, table: str, columns: list[str]) -> pd.DataFrame:
    allowed_tables = {"accounts", "customers", "devices", "merchants"}
    if table not in allowed_tables:
        raise ValueError(f"Tabel referensi tidak dikenal: {table}")
    query = f"SELECT {', '.join(columns)} FROM aml.{table}"
    rows = connection.execute(query).fetchall()
    return pd.DataFrame.from_records(rows, columns=columns)


def load_database_references(connection: Any) -> dict[str, pd.DataFrame]:
    return {
        "accounts": _read_reference_table(connection, "accounts", [
            "account_id", "customer_id", "owner_type", "account_type", "currency", "opened_date",
        ]),
        "customers": _read_reference_table(connection, "customers", [
            "customer_id", "resolved_entity_id", "onboarding_date", "ingested_at",
            "occupation", "income_band", "nationality", "kyc_status",
            "kyc_completeness_score", "customer_status",
        ]),
        "devices": _read_reference_table(connection, "devices", [
            "device_id", "device_type", "os_family", "app_version", "first_seen",
            "is_emulator_flag", "is_rooted_flag", "specificity_flag",
        ]),
        "merchants": _read_reference_table(connection, "merchants", [
            "merchant_id", "onboarded_date", "mcc", "mcc_description",
            "declared_expected_volume_band", "declared_expected_ticket_band",
            "operating_hours_declared", "outlet_count", "mcc_flag",
        ]),
    }


def _model_version() -> str:
    stat = MODEL_PATH.stat()
    return f"{MODEL_PATH.name}:{stat.st_size}:{stat.st_mtime_ns}"


def _normalize_upload_csv(file_storage: Any) -> tuple[pd.DataFrame, str]:
    filename = secure_filename(file_storage.filename or "")
    if not filename or not filename.lower().endswith(".csv"):
        raise ValueError("Unggah satu file CSV transaksi dengan ekstensi .csv.")
    file_storage.stream.seek(0)
    try:
        header = pd.read_csv(file_storage.stream, nrows=0)
        columns = set(header.columns.astype(str))
        missing = sorted(REQUIRED_UPLOAD_COLUMNS - columns)
        if missing:
            raise ValueError("Kolom CSV wajib tidak ditemukan: " + ", ".join(missing))
        file_storage.stream.seek(0)
        transactions = pd.read_csv(
            file_storage.stream,
            usecols=lambda column: column in ACTIVE_TRANSACTION_COLUMNS,
            dtype="string",
            low_memory=False,
        )
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"CSV tidak dapat dibaca: {exc}") from exc

    for column in ACTIVE_TRANSACTION_COLUMNS:
        if column not in transactions:
            transactions[column] = pd.NA
    transactions = transactions[ACTIVE_TRANSACTION_COLUMNS].copy()
    transactions["transaction_id"] = transactions["transaction_id"].str.strip()
    if transactions.empty:
        raise ValueError("CSV tidak berisi baris transaksi.")
    if transactions["transaction_id"].isna().any() or transactions["transaction_id"].eq("").any():
        raise ValueError("Setiap baris harus memiliki transaction_id.")
    if transactions["transaction_id"].duplicated().any():
        raise ValueError("transaction_id duplikat ditemukan dalam file upload.")
    transactions["_row_number"] = np.arange(len(transactions))
    return transactions, filename


def _persist_uploaded_dataset(
    transactions: pd.DataFrame,
    features: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    filename: str,
) -> tuple[str, str]:
    upload_id = str(uuid4())
    model_version = _model_version()
    probabilities = np.asarray(probabilities, dtype=float)
    if len(probabilities) != len(transactions):
        raise ValueError("Jumlah prediksi tidak sama dengan jumlah transaksi.")

    tx_rows = transactions[ACTIVE_TRANSACTION_COLUMNS].astype(object).where(
        pd.notna(transactions[ACTIVE_TRANSACTION_COLUMNS]), None
    )
    missing_counts = features[FEATURES].isna().sum(axis=1).to_numpy()
    prediction_rows = [
        (
            str(transactions.iloc[position]["transaction_id"]),
            model_version,
            float(score),
            "SUSPICIOUS" if score >= threshold else "NORMAL",
            threshold,
            int(missing_counts[position]),
        )
        for position, score in enumerate(probabilities)
    ]

    transaction_copy = (
        "COPY aml.active_transactions ("
        + ", ".join(ACTIVE_TRANSACTION_COLUMNS)
        + ") FROM STDIN"
    )
    prediction_copy = (
        "COPY aml.model_predictions "
        "(transaction_id, model_version, risk_score, prediction, threshold, missing_feature_count) "
        "FROM STDIN"
    )
    with get_database_connection() as connection:
        with connection.transaction():
            connection.execute("DELETE FROM aml.model_predictions")
            connection.execute("DELETE FROM aml.active_transactions")
            with connection.cursor().copy(transaction_copy) as copy:
                for row in tx_rows.itertuples(index=False, name=None):
                    copy.write_row(row)
            with connection.cursor().copy(prediction_copy) as copy:
                for row in prediction_rows:
                    copy.write_row(row)
            connection.execute(
                """
                INSERT INTO aml.dataset_state
                    (singleton, upload_id, filename, transaction_count, threshold, model_version, uploaded_at)
                VALUES (true, %s, %s, %s, %s, %s, now())
                ON CONFLICT (singleton) DO UPDATE SET
                    upload_id = EXCLUDED.upload_id,
                    filename = EXCLUDED.filename,
                    transaction_count = EXCLUDED.transaction_count,
                    threshold = EXCLUDED.threshold,
                    model_version = EXCLUDED.model_version,
                    uploaded_at = EXCLUDED.uploaded_at
                """,
                (upload_id, filename, len(transactions), threshold, model_version),
            )
    return upload_id, model_version


def database_dataset_status() -> dict[str, Any]:
    with get_database_connection() as connection:
        existing = connection.execute(
            """
            SELECT
                to_regclass('aml.customers') IS NOT NULL,
                to_regclass('aml.accounts') IS NOT NULL,
                to_regclass('aml.devices') IS NOT NULL,
                to_regclass('aml.merchants') IS NOT NULL,
                to_regclass('aml.active_transactions') IS NOT NULL,
                to_regclass('aml.model_predictions') IS NOT NULL,
                to_regclass('aml.dataset_state') IS NOT NULL
            """
        ).fetchone()
        required_names = [
            "customers", "accounts", "devices", "merchants",
            "active_transactions", "model_predictions", "dataset_state",
        ]
        tables = dict(zip(required_names, existing))
        missing_tables = [name for name, present in tables.items() if not present]
        if missing_tables:
            return {
                "ok": False,
                "status": "schema_incomplete",
                "message": "Jalankan database_schema.sql setelah tabel master.",
                "missing_tables": missing_tables,
                "tables": tables,
            }

        reference_counts = {
            table: int(connection.execute(f"SELECT count(*) FROM aml.{table}").fetchone()[0])
            for table in ("customers", "accounts", "devices", "merchants")
        }
        state = connection.execute(
            """
            SELECT upload_id, filename, transaction_count, threshold, model_version, uploaded_at
            FROM aml.dataset_state WHERE singleton = true
            """
        ).fetchone()
        status: dict[str, Any] = {
            "ok": True,
            "status": "ready" if state else "awaiting_upload",
            "tables": tables,
            "reference_counts": reference_counts,
            "active_dataset": None,
        }
        if state:
            status["active_dataset"] = {
                "upload_id": state[0],
                "filename": state[1],
                "transaction_count": state[2],
                "threshold": state[3],
                "model_version": state[4],
                "uploaded_at": state[5].isoformat(),
            }
        return status


def _read_dataset_file(key: str) -> pd.DataFrame:
    filename = DATA_FILES[key]
    path = DATA_DIR / filename
    if not path.is_file():
        raise ValueError(f"File dataset tidak ditemukan: {filename}")
    try:
        frame = pd.read_csv(path, usecols=sorted(DATA_COLUMNS[key]), low_memory=False)
    except ValueError as exc:
        raise ValueError(f"Kolom dataset {filename} tidak sesuai skema: {exc}") from exc
    if frame.empty:
        raise ValueError(f"Dataset {filename} tidak memiliki baris data.")
    return frame


def _calculate_account_history(data: pd.DataFrame) -> None:
    """Create strictly-prior rolling transaction features, grouped by account."""
    outputs = {
        "prior_tx_count_1h": np.full(len(data), np.nan),
        "prior_tx_count_24h": np.full(len(data), np.nan),
        "prior_amount_idr_1h": np.full(len(data), np.nan),
        "prior_amount_idr_24h": np.full(len(data), np.nan),
        "prior_in_amount_idr_24h": np.full(len(data), np.nan),
        "prior_out_amount_idr_24h": np.full(len(data), np.nan),
        "minutes_since_previous_tx": np.full(len(data), np.nan),
    }
    valid_time = data["_tx_time"].notna()
    amount = pd.to_numeric(data["amount_idr"], errors="coerce").to_numpy(dtype=float)
    amount = np.nan_to_num(amount, nan=0.0)
    direction = data["direction"].astype("string").str.upper()
    incoming = np.where(direction.eq("IN").fillna(False).to_numpy(), amount, 0.0)
    outgoing = np.where(direction.eq("OUT").fillna(False).to_numpy(), amount, 0.0)
    timestamps = data["_tx_time"].array.asi8
    valid_positions = np.flatnonzero(valid_time.to_numpy())
    hour_ns = 60 * 60 * 1_000_000_000
    day_ns = 24 * hour_ns

    for positions in data.loc[valid_time].groupby("account_id", sort=False).indices.values():
        positions = valid_positions[np.asarray(positions, dtype=np.int64)]
        positions = positions[np.argsort(timestamps[positions], kind="stable")]
        times = timestamps[positions]
        ends = np.searchsorted(times, times, side="left")
        starts_1h = np.searchsorted(times, times - hour_ns, side="left")
        starts_24h = np.searchsorted(times, times - day_ns, side="left")
        count_1h = ends - starts_1h
        count_24h = ends - starts_24h

        def prior_sum(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
            prefix = np.concatenate(([0.0], np.cumsum(values[positions], dtype=float)))
            return prefix[ends] - prefix[starts]

        outputs["prior_tx_count_1h"][positions] = count_1h
        outputs["prior_tx_count_24h"][positions] = count_24h
        outputs["prior_amount_idr_1h"][positions] = prior_sum(amount, starts_1h)
        outputs["prior_amount_idr_24h"][positions] = prior_sum(amount, starts_24h)
        outputs["prior_in_amount_idr_24h"][positions] = prior_sum(incoming, starts_24h)
        outputs["prior_out_amount_idr_24h"][positions] = prior_sum(outgoing, starts_24h)
        previous = ends - 1
        has_previous = previous >= 0
        elapsed = np.full(len(positions), np.nan)
        elapsed[has_previous] = (
            times[has_previous] - times[previous[has_previous]]
        ) / (60 * 1_000_000_000)
        outputs["minutes_since_previous_tx"][positions] = elapsed

    for name, values in outputs.items():
        data[name] = values


def _build_dataset_features(
    transactions: pd.DataFrame | None = None,
    references: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    if transactions is None:
        transactions = _read_dataset_file("transactions")
    if references is None:
        references = {
            "accounts": _read_dataset_file("accounts"),
            "customers": _read_dataset_file("customers"),
            "devices": _read_dataset_file("devices"),
            "merchants": _read_dataset_file("merchants"),
        }
    accounts = references["accounts"]
    customers = references["customers"]
    devices = references["devices"]
    merchants = references["merchants"]

    reference_keys = [
        (accounts, "account_id", DATA_FILES["accounts"]),
        (customers, "customer_id", DATA_FILES["customers"]),
        (devices, "device_id", DATA_FILES["devices"]),
        (merchants, "merchant_id", DATA_FILES["merchants"]),
    ]
    for reference, key, filename in reference_keys:
        if reference[key].dropna().duplicated().any():
            raise ValueError(f"Kolom kunci {key} harus unik di {filename}.")
    if transactions["transaction_id"].dropna().duplicated().any():
        raise ValueError("transaction_id harus unik di dataset transaksi.")

    data = transactions.copy()
    data["_row_number"] = np.arange(len(data))
    data = data.merge(
        accounts[["account_id", "customer_id", "owner_type", "account_type", "currency", "opened_date"]]
        .rename(columns={
            "customer_id": "_account_customer_id",
            "owner_type": "account_owner_type",
            "account_type": "account_type",
            "currency": "account_currency",
            "opened_date": "_account_opened_date",
        }),
        on="account_id",
        how="left",
        validate="many_to_one",
    )

    entity_counts = customers.groupby("resolved_entity_id")["customer_id"].nunique()
    unique_entities = entity_counts[entity_counts == 1].index
    entity_to_customer = (
        customers[customers["resolved_entity_id"].isin(unique_entities)]
        .set_index("resolved_entity_id")["customer_id"]
        .to_dict()
    )
    data["_resolved_customer_id"] = data["_account_customer_id"].fillna(
        data["entity_id"].map(entity_to_customer)
    )
    customer_features = customers[[
        "customer_id", "onboarding_date", "ingested_at", "occupation",
        "income_band", "nationality", "kyc_status", "kyc_completeness_score",
        "customer_status",
    ]].rename(columns={
        "customer_id": "_resolved_customer_id",
        "onboarding_date": "_customer_onboarding_date",
        "ingested_at": "_customer_ingested_at",
        "occupation": "customer_occupation",
        "income_band": "customer_income_band",
        "nationality": "customer_nationality",
        "kyc_status": "customer_kyc_status",
        "kyc_completeness_score": "customer_kyc_completeness_score",
        "customer_status": "customer_status",
    })
    data = data.merge(
        customer_features,
        on="_resolved_customer_id",
        how="left",
        validate="many_to_one",
    )

    data = data.merge(
        devices[[
            "device_id", "device_type", "os_family", "app_version", "first_seen",
            "is_emulator_flag", "is_rooted_flag", "specificity_flag",
        ]].rename(columns={
            "os_family": "device_os_family",
            "app_version": "device_app_version",
            "first_seen": "_device_first_seen",
            "is_emulator_flag": "device_is_emulator_flag",
            "is_rooted_flag": "device_is_rooted_flag",
            "specificity_flag": "device_specificity_flag",
        }),
        on="device_id",
        how="left",
        validate="many_to_one",
    )
    data = data.merge(
        merchants[[
            "merchant_id", "onboarded_date", "mcc", "mcc_description",
            "declared_expected_volume_band", "declared_expected_ticket_band",
            "operating_hours_declared", "outlet_count", "mcc_flag",
        ]].rename(columns={
            "onboarded_date": "_merchant_onboarded_date",
            "mcc": "merchant_mcc",
            "mcc_description": "merchant_mcc_description",
            "declared_expected_volume_band": "merchant_volume_band",
            "declared_expected_ticket_band": "merchant_ticket_band",
            "operating_hours_declared": "merchant_operating_hours",
            "outlet_count": "merchant_outlet_count",
            "mcc_flag": "merchant_mcc_flag",
        }),
        on="merchant_id",
        how="left",
        validate="many_to_one",
    )

    data["_tx_time"] = pd.to_datetime(
        data["value_datetime_utc"], utc=True, format="mixed", errors="coerce"
    )
    date_sources = {
        "_account_opened_date": "_account_opened_at",
        "_customer_onboarding_date": "_customer_onboarded_at",
        "_customer_ingested_at": "_customer_available_at",
        "_device_first_seen": "_device_first_seen_at",
        "_merchant_onboarded_date": "_merchant_opened_at",
    }
    for source, target in date_sources.items():
        data[target] = pd.to_datetime(data[source], utc=True, format="mixed", errors="coerce")

    data["hour"] = data["_tx_time"].dt.hour
    data["day_of_week"] = data["_tx_time"].dt.dayofweek
    amount_idr = pd.to_numeric(data["amount_idr"], errors="coerce")
    data["log_amount_idr"] = np.log1p(amount_idr.where(amount_idr >= 0))
    for feature, date_column in (
        ("account_age_days", "_account_opened_at"),
        ("customer_tenure_days", "_customer_onboarded_at"),
        ("device_age_days", "_device_first_seen_at"),
        ("merchant_age_days", "_merchant_opened_at"),
    ):
        data[feature] = (data["_tx_time"] - data[date_column]).dt.total_seconds() / 86400

    for feature, date_column in (
        ("customer_record_available_at_tx", "_customer_available_at"),
        ("device_seen_before_tx", "_device_first_seen_at"),
        ("merchant_opened_before_tx", "_merchant_opened_at"),
    ):
        known = data[date_column].notna() & data["_tx_time"].notna()
        data[feature] = np.nan
        data.loc[known, feature] = (
            data.loc[known, date_column] <= data.loc[known, "_tx_time"]
        ).astype(int)

    _calculate_account_history(data)
    for feature, source in (
        ("fx_rate_reference", "fx_rate_reference"),
        ("customer_kyc_completeness_score", "customer_kyc_completeness_score"),
        ("merchant_outlet_count", "merchant_outlet_count"),
    ):
        data[feature] = pd.to_numeric(data[source], errors="coerce")
    for feature in FEATURES:
        if feature in NUMERIC_FEATURES:
            data[feature] = pd.to_numeric(data[feature], errors="coerce")
        elif feature not in data:
            raise ValueError(f"Fitur model tidak berhasil dibentuk: {feature}")

    data = data.sort_values("_row_number", kind="stable").reset_index(drop=True)
    return data


def get_dataset(force_reload: bool = False) -> pd.DataFrame:
    global _dataset_cache, _dataset_cache_signature
    with _dataset_lock:
        inventory = dataset_inventory()
        if not inventory["ok"]:
            raise ValueError(
                "Dataset belum lengkap. File yang tidak ditemukan: "
                + ", ".join(inventory["missing_files"])
            )
        signature = tuple(
            (entry["filename"], entry["size_bytes"], entry["modified_at"])
            for entry in inventory["files"].values()
        )
        if force_reload or _dataset_cache is None or signature != _dataset_cache_signature:
            _dataset_cache = _build_dataset_features()
            _dataset_cache_signature = signature
        return _dataset_cache


@app.get("/")
def dashboard():
    return send_from_directory(DASHBOARD_DIR, DASHBOARD_PATH)


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


@app.get("/api/datasets/status")
def datasets_status():
    if DB_CONFIGURED:
        try:
            status = database_dataset_status()
            status["feature_count"] = len(FEATURES)
            status["database"] = {
                "host": DB_CONFIG["host"],
                "port": DB_CONFIG["port"],
                "database": DB_CONFIG["dbname"],
            }
            return jsonify(status), 200 if status["ok"] else 503
        except Exception as exc:
            return jsonify({
                "ok": False,
                "status": "database_unavailable",
                "message": "Backend tidak dapat membaca PostgreSQL.",
                "details": f"{type(exc).__name__}: {exc}",
            }), 503

    inventory = dataset_inventory()
    inventory["message"] = (
        "Semua file CSV sumber ditemukan." if inventory["ok"]
        else "Satu atau lebih file CSV sumber belum tersedia."
    )
    inventory["feature_count"] = len(FEATURES)
    inventory["feature_engineering_notes"] = FEATURE_ENGINEERING_NOTES
    if _dataset_cache is not None:
        inventory["transaction_count"] = len(_dataset_cache)
    return jsonify(inventory), 200 if inventory["ok"] else 503


@app.get("/api/db/health")
def database_health():
    if not DB_CONFIGURED:
        return jsonify({
            "ok": False,
            "status": "not_configured",
            "message": "Set AML_DB_PASSWORD pada environment sebelum menjalankan backend.",
        }), 503
    try:
        with get_database_connection() as connection:
            server_version = connection.execute("SHOW server_version").fetchone()[0]
        status = database_dataset_status()
        return jsonify({
            "ok": status["ok"],
            "status": status["status"],
            "message": "Koneksi PostgreSQL berhasil.",
            "server_version": server_version,
            "dataset": status,
        }), 200 if status["ok"] else 503
    except Exception as exc:
        return jsonify({
            "ok": False,
            "status": "database_unavailable",
            "message": "Backend tidak dapat terhubung ke PostgreSQL atau membaca skema AML.",
            "details": f"{type(exc).__name__}: {exc}",
        }), 503


@app.post("/api/datasets/upload")
def upload_transactions():
    if model is None:
        return jsonify({
            "ok": False,
            "error": "Model belum tersedia. Periksa /api/health.",
            "details": model_load_error,
        }), 503
    if not DB_CONFIGURED:
        return jsonify({
            "ok": False,
            "error": "Koneksi PostgreSQL belum dikonfigurasi.",
            "details": "Set AML_DB_PASSWORD pada environment backend.",
        }), 503
    upload = request.files.get("file")
    if upload is None:
        return jsonify({"ok": False, "error": "Pilih file CSV pada field 'file'."}), 400
    try:
        threshold = float(request.form.get("threshold", "0.5"))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "threshold harus berupa angka antara 0 dan 1."}), 400
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        return jsonify({"ok": False, "error": "threshold harus berada di rentang 0 sampai 1."}), 400

    try:
        transactions, filename = _normalize_upload_csv(upload)
        with get_database_connection() as connection:
            references = load_database_references(connection)
        features = _build_dataset_features(transactions, references)

        classes = list(model.classes_)
        suspicious_index = next((i for i, label in enumerate(classes) if str(label) == "1"), None)
        if suspicious_index is None:
            return jsonify({
                "ok": False,
                "error": "Model tidak memiliki kelas target 1 untuk SUSPICIOUS.",
            }), 500
        scores: list[float] = []
        batch_size = 5000
        for start in range(0, len(features), batch_size):
            batch = features.iloc[start:start + batch_size]
            probabilities = model.predict_proba(batch[FEATURES])
            scores.extend(float(score) for score in probabilities[:, suspicious_index])
        upload_id, model_version = _persist_uploaded_dataset(
            transactions, features, np.asarray(scores), threshold, filename
        )
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return jsonify({
            "ok": False,
            "error": "Upload dibatalkan; dataset aktif tidak diubah.",
            "details": f"{type(exc).__name__}: {exc}",
        }), 400
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": "Upload atau penyimpanan prediksi gagal; dataset aktif tidak diubah.",
            "details": f"{type(exc).__name__}: {exc}",
        }), 503

    count = len(scores)
    suspicious_count = sum(score >= threshold for score in scores)
    average_score = float(np.mean(scores)) if scores else 0.0
    return jsonify({
        "ok": True,
        "upload_id": upload_id,
        "filename": filename,
        "transaction_count": count,
        "suspicious_count": suspicious_count,
        "potential_fraud_count": suspicious_count,
        "normal_count": count - suspicious_count,
        "high_risk_percent": round(suspicious_count * 100 / count, 2) if count else 0.0,
        "average_risk_score": round(average_score, 6),
        "average_risk_percent": round(average_score * 100, 2),
        "threshold": threshold,
        "model_version": model_version,
        "message": "Dataset aktif telah diganti dan semua transaksi berhasil diprediksi.",
    })


@app.get("/api/dashboard/summary")
def dashboard_summary():
    if not DB_CONFIGURED:
        return jsonify({"ok": False, "error": "Koneksi PostgreSQL belum dikonfigurasi."}), 503
    try:
        threshold = float(request.args.get("threshold", "0.5"))
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold harus berada di rentang 0 sampai 1.")
        with get_database_connection() as connection:
            total, suspicious, average_score = connection.execute(
                """
                SELECT count(*),
                       count(*) FILTER (WHERE risk_score >= %s),
                       coalesce(avg(risk_score), 0)
                FROM aml.model_predictions
                """,
                (threshold,),
            ).fetchone()
            state = connection.execute(
                "SELECT filename, uploaded_at, model_version FROM aml.dataset_state WHERE singleton = true"
            ).fetchone()
        total = int(total)
        suspicious = int(suspicious)
        average_score = float(average_score)
        return jsonify({
            "ok": True,
            "total_transactions": total,
            "suspicious_count": suspicious,
            "potential_fraud_count": suspicious,
            "normal_count": total - suspicious,
            "high_risk_percent": round(suspicious * 100 / total, 2) if total else 0.0,
            "average_risk_score": round(average_score, 6),
            "average_risk_percent": round(average_score * 100, 2),
            "threshold": threshold,
            "filename": state[0] if state else None,
            "uploaded_at": state[1].isoformat() if state else None,
            "model_version": state[2] if state else None,
        })
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": "Ringkasan dataset tidak dapat dibaca dari PostgreSQL.",
            "details": f"{type(exc).__name__}: {exc}",
        }), 503


@app.get("/api/alerts")
def dataset_alerts():
    if not DB_CONFIGURED:
        return jsonify({"ok": False, "error": "Koneksi PostgreSQL belum dikonfigurasi."}), 503
    try:
        threshold = float(request.args.get("threshold", "0.5"))
        limit = int(request.args.get("limit", "50"))
        offset = int(request.args.get("offset", "0"))
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold harus berada di rentang 0 sampai 1.")
        if not 1 <= limit <= 100:
            raise ValueError("limit harus berada di rentang 1 sampai 100.")
        if offset < 0:
            raise ValueError("offset harus nol atau lebih.")
        with get_database_connection() as connection:
            total = int(connection.execute(
                "SELECT count(*) FROM aml.model_predictions WHERE risk_score >= %s",
                (threshold,),
            ).fetchone()[0])
            rows = connection.execute(
                """
                SELECT p.transaction_id, t.account_id, t.value_datetime_utc,
                       t.direction, t.amount_idr, p.risk_score, p.missing_feature_count
                FROM aml.model_predictions p
                JOIN aml.active_transactions t USING (transaction_id)
                WHERE p.risk_score >= %s
                ORDER BY p.risk_score DESC, p.transaction_id
                LIMIT %s OFFSET %s
                """,
                (threshold, limit, offset),
            ).fetchall()
        alerts = [
            {
                "id": row[0],
                "account_id": row[1],
                "value_datetime_utc": row[2],
                "direction": row[3],
                "amount_idr": float(row[4]) if row[4] not in (None, "") else None,
                "risk_score": round(float(row[5]), 6),
                "risk_percent": round(float(row[5]) * 100, 2),
                "potential_fraud_class": "POTENTIAL_FRAUD",
                "missing_feature_count": int(row[6]),
            }
            for row in rows
        ]
        return jsonify({
            "ok": True,
            "threshold": threshold,
            "total_count": total,
            "limit": limit,
            "offset": offset,
            "returned_count": len(alerts),
            "alerts": alerts,
        })
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": "Daftar alert tidak dapat dibaca dari PostgreSQL.",
            "details": f"{type(exc).__name__}: {exc}",
        }), 503


@app.post("/api/datasets/predict")
def predict_dataset():
    if model is None:
        return jsonify({
            "ok": False,
            "error": "Model belum tersedia. Periksa /api/health dan versi dependensi.",
            "details": model_load_error,
        }), 503

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"ok": False, "error": "Body harus berupa objek JSON."}), 400

    try:
        threshold = float(payload.get("threshold", 0.5))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "threshold harus berupa angka antara 0 dan 1."}), 400
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        return jsonify({"ok": False, "error": "threshold harus berada di rentang 0 sampai 1."}), 400

    offset = payload.get("offset", 0)
    limit = payload.get("limit", 100)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return jsonify({"ok": False, "error": "offset harus berupa integer nol atau lebih."}), 400
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        return jsonify({"ok": False, "error": "limit harus berupa integer antara 1 dan 1000."}), 400

    if DB_CONFIGURED:
        try:
            with get_database_connection() as connection:
                total_count, suspicious_count, normal_count = connection.execute(
                    """
                    SELECT count(*),
                           count(*) FILTER (WHERE risk_score >= %s),
                           count(*) FILTER (WHERE risk_score < %s)
                    FROM aml.model_predictions
                    """,
                    (threshold, threshold),
                ).fetchone()
                rows = connection.execute(
                    """
                    SELECT p.transaction_id, t.account_id, t.value_datetime_utc,
                           t.direction, t.amount_idr, p.prediction, p.risk_score,
                           p.missing_feature_count
                    FROM aml.model_predictions p
                    JOIN aml.active_transactions t USING (transaction_id)
                    ORDER BY p.risk_score DESC, p.transaction_id
                    LIMIT %s OFFSET %s
                    """,
                    (limit, offset),
                ).fetchall()
            predictions = [
                {
                    "transaction_id": row[0],
                    "account_id": row[1],
                    "value_datetime_utc": row[2],
                    "direction": row[3],
                    "amount_idr": (
                        float(row[4]) if row[4] not in (None, "") else None
                    ),
                    "prediction": "SUSPICIOUS" if float(row[6]) >= threshold else "NORMAL",
                    "potential_fraud_class": (
                        "POTENTIAL_FRAUD" if float(row[6]) >= threshold else "NOT_FLAGGED"
                    ),
                    "risk_score": round(float(row[6]), 6),
                    "risk_percent": round(float(row[6]) * 100, 2),
                    "missing_feature_count": int(row[7]),
                }
                for row in rows
            ]
            return jsonify({
                "ok": True,
                "total_count": int(total_count),
                "offset": offset,
                "returned_count": len(predictions),
                "threshold": threshold,
                "suspicious_count": int(suspicious_count),
                "potential_fraud_count": int(suspicious_count),
                "normal_count": int(normal_count),
                "predictions": predictions,
            })
        except Exception as exc:
            return jsonify({
                "ok": False,
                "error": "Hasil prediksi tidak dapat dibaca dari PostgreSQL.",
                "details": f"{type(exc).__name__}: {exc}",
            }), 503

    try:
        data = get_dataset()
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return jsonify({
            "ok": False,
            "error": "Dataset CSV belum dapat dimuat.",
            "details": f"{type(exc).__name__}: {exc}",
        }), 503

    page = data.iloc[offset:offset + limit]
    if page.empty:
        return jsonify({
            "ok": True,
            "total_count": len(data),
            "offset": offset,
            "returned_count": 0,
            "predictions": [],
            "warning": "Tidak ada transaksi pada rentang offset yang diminta.",
        })

    try:
        probabilities = model.predict_proba(page[FEATURES])
        classes = list(model.classes_)
        suspicious_index = next((i for i, cls in enumerate(classes) if str(cls) == "1"), None)
        if suspicious_index is None:
            return jsonify({
                "ok": False,
                "error": "Pipeline tidak memiliki kelas target 1. Pastikan label 1 berarti SUSPICIOUS.",
            }), 500
    except Exception as exc:
        return jsonify({
            "ok": False,
            "error": "Prediksi dataset gagal dijalankan oleh pipeline.",
            "details": f"{type(exc).__name__}: {exc}",
        }), 500

    scores = probabilities[:, suspicious_index]
    predictions: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(page.iterrows()):
        score = float(scores[position])
        amount_value = pd.to_numeric(row["amount_idr"], errors="coerce")
        predictions.append({
            "transaction_id": str(row["transaction_id"]),
            "account_id": None if pd.isna(row["account_id"]) else str(row["account_id"]),
            "value_datetime_utc": (
                None if pd.isna(row["value_datetime_utc"]) else str(row["value_datetime_utc"])
            ),
            "direction": None if pd.isna(row["direction"]) else str(row["direction"]),
            "amount_idr": None if pd.isna(amount_value) else float(amount_value),
            "prediction": "SUSPICIOUS" if score >= threshold else "NORMAL",
            "potential_fraud_class": (
                "POTENTIAL_FRAUD" if score >= threshold else "NOT_FLAGGED"
            ),
            "risk_score": round(score, 6),
            "risk_percent": round(score * 100, 2),
            "missing_feature_count": int(row[FEATURES].isna().sum()),
        })
    suspicious_count = sum(item["prediction"] == "SUSPICIOUS" for item in predictions)
    return jsonify({
        "ok": True,
        "total_count": len(data),
        "offset": offset,
        "returned_count": len(predictions),
        "threshold": threshold,
        "suspicious_count": suspicious_count,
        "potential_fraud_count": suspicious_count,
        "normal_count": len(predictions) - suspicious_count,
        "predictions": predictions,
        "feature_engineering_notes": FEATURE_ENGINEERING_NOTES,
        "warning": (
            "Skor memakai fitur yang direkayasa dari CSV dan asumsi yang tercantum. "
            "Kelas positif adalah proxy potensi fraud dari label sintetis SUSPICIOUS, bukan fraud terkonfirmasi. "
            "Validasi transformasi terhadap pipeline training sebelum penggunaan operasional."
        ),
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
        "potential_fraud_class": (
            "POTENTIAL_FRAUD" if is_suspicious else "NOT_FLAGGED"
        ),
        "risk_score": round(score, 6),
        "risk_percent": round(score * 100, 2),
        "threshold": threshold,
        "provided_feature_count": len(provided),
        "total_feature_count": len(FEATURES),
        "missing_features": missing,
        "missing_feature_count": len(missing),
        "warning": (
            "Model mempelajari label sintetis SUSPICIOUS sebagai proxy potensi fraud, "
            "bukan fraud terkonfirmasi. Sebagian fitur tidak diberikan dan diisi oleh preprocessing pipeline. "
            "Gunakan semua fitur yang tersedia dan validasi model sebelum memakai skor untuk keputusan operasional."
            if missing else
            "Label positif adalah proxy potensi fraud dari data sintetis SUSPICIOUS, bukan fraud terkonfirmasi. "
            "Validasi performa dan ambang batas model sebelum penggunaan operasional."
        ),
        "model": model_metadata(),
    })


if __name__ == "__main__":
    # Debug is deliberately off; use a production WSGI server for deployment.
    app.run(host="127.0.0.1", port=5000, debug=False)
