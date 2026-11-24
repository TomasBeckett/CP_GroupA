"""Train, regularize, and evaluate dual models:
1. AML Detection Model (Anti-Money Laundering)
2. Fraud Detection Model
From FCIP synthetic transactions with chronological holdout split.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import app

PACKAGE_DIR = Path(__file__).resolve().parent
MODELS_DIR = PACKAGE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

AML_MODEL_PATH = MODELS_DIR / "aml_random_forest_pipeline_trained.joblib"
FRAUD_MODEL_PATH = MODELS_DIR / "fraud_random_forest_pipeline_trained.joblib"

AML_REPORT_PATH = MODELS_DIR / "aml_training_report.json"
FRAUD_REPORT_PATH = MODELS_DIR / "fraud_training_report.json"
DUAL_REPORT_PATH = MODELS_DIR / "dual_model_evaluation_report.json"

AML_PATTERNS = {
    "P02_STRUCTURING",
    "P03_RAPID_FUND_MOVEMENT",
    "P09_MANY_TO_ONE",
    "P10_UNUSUAL_SETTLEMENT",
    "P11_CIRCULAR_PATTERN",
}

FRAUD_PATTERNS = {
    "P04_SUDDEN_SPIKE",
    "P05_DORMANT_REACTIVATION",
    "P06_HIGH_VELOCITY",
    "P07_SHARED_DEVICE_NETWORK",
}


def build_regularized_pipeline(
    trees: int = 100,
    max_depth: int = 12,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    random_state: int = 42,
    n_jobs: int = 8,
) -> Pipeline:
    numeric_features = [name for name in app.FEATURES if name in app.NUMERIC_FEATURES]
    categorical_features = [name for name in app.FEATURES if name not in app.NUMERIC_FEATURES]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=trees,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=random_state,
        )),
    ])


def evaluate_split(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "average_precision": float(average_precision_score(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1_score": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": cm,
        "support_negative": int((y == 0).sum()),
        "support_positive": int((y == 1).sum()),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as temp:
        temp_path = Path(temp.name)
        json.dump(payload, temp, indent=2, ensure_ascii=False, allow_nan=False)
        temp.write("\n")
    os.replace(temp_path, path)


def _atomic_write_model(path: Path, pipeline: Pipeline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as temp:
        temp_path = Path(temp.name)
    try:
        joblib.dump(pipeline, temp_path, compress=3)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    print("=" * 80)
    print("PELATIHAN 2 MODEL FCIP (AML vs FRAUD DETECTION)")
    print("Strategi: Regularisasi Hyperparameter untuk Mitigasi Overfitting")
    print("=" * 80)

    start_total = time.perf_counter()

    # 1. Load dataset with 46 features
    print("\n[1/5] Memuat dataset gabungan dari database/CSV...")
    t0 = time.perf_counter()
    data = app.get_dataset(force_reload=True)
    print(f"Dataset dimuat: {len(data):,} baris dalam {time.perf_counter() - t0:.2f} detik")

    # 2. Load pattern & scenario labels
    tx_path = app.DATA_DIR / app.DATA_FILES["transactions"]
    raw_labels = pd.read_csv(
        tx_path,
        usecols=["transaction_id", "synthetic_pattern", "scenario_label"],
        low_memory=False,
    )
    data = data.merge(raw_labels, on="transaction_id", how="left")

    # Exclude EDGE_CASE
    clean_data = data[data["scenario_label"].isin(["NORMAL", "SUSPICIOUS"])].copy()
    print(f"Data bersih untuk pelatihan: {len(clean_data):,} baris (16,500 EDGE_CASE di-exclude)")

    # 3. Create target labels
    clean_data["target_aml"] = clean_data["synthetic_pattern"].isin(AML_PATTERNS).astype(int)
    clean_data["target_fraud"] = clean_data["synthetic_pattern"].isin(FRAUD_PATTERNS).astype(int)

    print(f"Distribusi Target AML: {clean_data['target_aml'].value_counts().to_dict()}")
    print(f"Distribusi Target Fraud: {clean_data['target_fraud'].value_counts().to_dict()}")

    # 4. Temporal split
    clean_data = clean_data.sort_values("_tx_time").reset_index(drop=True)
    unique_times = np.sort(clean_data["_tx_time"].unique())
    split_index = int(len(unique_times) * 0.8)
    cutoff = pd.Timestamp(unique_times[split_index])

    train_mask = clean_data["_tx_time"] < cutoff
    test_mask = ~train_mask

    train_df = clean_data[train_mask]
    test_df = clean_data[test_mask]

    print(f"\n[2/5] Split Temporal: Cutoff {cutoff.isoformat()}")
    print(f"  Train: {len(train_df):,} baris (AML pos: {train_df['target_aml'].sum():,}, Fraud pos: {train_df['target_fraud'].sum():,})")
    print(f"  Test : {len(test_df):,} baris (AML pos: {test_df['target_aml'].sum():,}, Fraud pos: {test_df['target_fraud'].sum():,})")

    # Evaluate on full test and 50k stratified sample of train for fast & robust metrics evaluation
    # (Full train prediction of 266k rows with 100 trees takes ~15s, so let's use full train!)
    X_train = train_df[app.FEATURES]
    X_test = test_df[app.FEATURES]

    # =========================================================================
    # MODEL 1: AML DETECTION MODEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("[3/5] TRAINING MODEL 1: AML DETECTION (ANTI-MONEY LAUNDERING)")
    print("=" * 80)
    print("Parameter: max_depth=12, min_samples_leaf=10, min_samples_split=20, n_estimators=100")

    aml_pipeline = build_regularized_pipeline(
        trees=100,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=8,
    )
    t_aml = time.perf_counter()
    aml_pipeline.fit(X_train, train_df["target_aml"])
    aml_fit_sec = time.perf_counter() - t_aml
    print(f"Model AML berhasil dilatih dalam {aml_fit_sec:.2f} detik.")

    print("Mengevaluasi performa Train vs Test Model AML...")
    aml_train_metrics = evaluate_split(aml_pipeline, X_train, train_df["target_aml"].to_numpy())
    aml_test_metrics = evaluate_split(aml_pipeline, X_test, test_df["target_aml"].to_numpy())

    # =========================================================================
    # MODEL 2: FRAUD DETECTION MODEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("[4/5] TRAINING MODEL 2: FRAUD DETECTION")
    print("=" * 80)
    print("Parameter: max_depth=12, min_samples_leaf=10, min_samples_split=20, n_estimators=100")

    fraud_pipeline = build_regularized_pipeline(
        trees=100,
        max_depth=12,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=8,
    )
    t_fraud = time.perf_counter()
    fraud_pipeline.fit(X_train, train_df["target_fraud"])
    fraud_fit_sec = time.perf_counter() - t_fraud
    print(f"Model Fraud berhasil dilatih dalam {fraud_fit_sec:.2f} detik.")

    print("Mengevaluasi performa Train vs Test Model Fraud...")
    fraud_train_metrics = evaluate_split(fraud_pipeline, X_train, train_df["target_fraud"].to_numpy())
    fraud_test_metrics = evaluate_split(fraud_pipeline, X_test, test_df["target_fraud"].to_numpy())

    # =========================================================================
    # SAVE MODELS & REPORTS
    # =========================================================================
    print("\n[5/5] Menyimpan model artifacts dan laporan komparatif...")
    _atomic_write_model(AML_MODEL_PATH, aml_pipeline)
    _atomic_write_model(FRAUD_MODEL_PATH, fraud_pipeline)

    aml_summary = {
        "model_name": "FCIP Anti-Money Laundering Detection Model",
        "target": "target_aml",
        "target_typologies": sorted(AML_PATTERNS),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_metrics": aml_train_metrics,
        "test_metrics": aml_test_metrics,
        "fit_seconds": aml_fit_sec,
        "parameters": {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "class_weight": "balanced_subsample",
        },
    }
    _atomic_write_json(AML_REPORT_PATH, aml_summary)

    fraud_summary = {
        "model_name": "FCIP Fraud Detection Model",
        "target": "target_fraud",
        "target_typologies": sorted(FRAUD_PATTERNS),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_metrics": fraud_train_metrics,
        "test_metrics": fraud_test_metrics,
        "fit_seconds": fraud_fit_sec,
        "parameters": {
            "n_estimators": 100,
            "max_depth": 12,
            "min_samples_split": 20,
            "min_samples_leaf": 10,
            "class_weight": "balanced_subsample",
        },
    }
    _atomic_write_json(FRAUD_REPORT_PATH, fraud_summary)

    dual_summary = {
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "total_seconds": time.perf_counter() - start_total,
        "split_cutoff": cutoff.isoformat(),
        "aml_model": aml_summary,
        "fraud_model": fraud_summary,
    }
    _atomic_write_json(DUAL_REPORT_PATH, dual_summary)

    print("\n" + "=" * 90)
    print("BUKTI PERBANDINGAN METRIK (TRAIN vs TEST) - MODEL 1: AML DETECTION")
    print("=" * 90)
    print(f"{'METRIK':<25} | {'TRAIN':<12} | {'TEST':<12} | {'GAP (Train - Test)':<20} | {'STATUS GENERALISASI'}")
    print("-" * 90)
    metrics_list = ["accuracy", "balanced_accuracy", "roc_auc", "average_precision", "precision", "recall", "f1_score"]
    for m in metrics_list:
        val_tr = aml_train_metrics[m]
        val_te = aml_test_metrics[m]
        gap = val_tr - val_te
        status = "WELL-FITTED (STABIL)" if abs(gap) <= 0.06 else ("OVERFITTING KECIL" if gap > 0 else "TEST LEBIH TINGGI")
        print(f"{m:<25} | {val_tr:<12.4f} | {val_te:<12.4f} | {gap:+12.4f} ({gap*100:+6.2f}%) | {status}")
    print(f"Confusion Matrix Train: {aml_train_metrics['confusion_matrix']}")
    print(f"Confusion Matrix Test : {aml_test_metrics['confusion_matrix']}")

    print("\n" + "=" * 90)
    print("BUKTI PERBANDINGAN METRIK (TRAIN vs TEST) - MODEL 2: FRAUD DETECTION")
    print("=" * 90)
    print(f"{'METRIK':<25} | {'TRAIN':<12} | {'TEST':<12} | {'GAP (Train - Test)':<20} | {'STATUS GENERALISASI'}")
    print("-" * 90)
    for m in metrics_list:
        val_tr = fraud_train_metrics[m]
        val_te = fraud_test_metrics[m]
        gap = val_tr - val_te
        status = "WELL-FITTED (STABIL)" if abs(gap) <= 0.06 else ("OVERFITTING KECIL" if gap > 0 else "TEST LEBIH TINGGI")
        print(f"{m:<25} | {val_tr:<12.4f} | {val_te:<12.4f} | {gap:+12.4f} ({gap*100:+6.2f}%) | {status}")
    print(f"Confusion Matrix Train: {fraud_train_metrics['confusion_matrix']}")
    print(f"Confusion Matrix Test : {fraud_test_metrics['confusion_matrix']}")

    print("\n" + "=" * 90)
    print(f"SELESAI! Waktu total: {time.perf_counter() - start_total:.2f} detik")
    print("Artifact Model AML   :", AML_MODEL_PATH)
    print("Artifact Model Fraud :", FRAUD_MODEL_PATH)
    print("Artifact Report Dual :", DUAL_REPORT_PATH)
    print("=" * 90)


if __name__ == "__main__":
    main()
