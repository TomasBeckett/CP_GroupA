"""Train and evaluate the AML Random Forest model from the local FCIP CSVs.

Run with: python train_model.py
"""
from __future__ import annotations

import argparse
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app import DATA_DIR, DATA_FILES, FEATURES, NUMERIC_FEATURES, get_dataset

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PACKAGE_DIR / "models" / "aml_random_forest_pipeline_trained.joblib"
DEFAULT_REPORT_PATH = PACKAGE_DIR / "models" / "aml_training_report.json"
TARGET_COLUMN = "scenario_label"
TARGET_MAP = {"NORMAL": 0, "SUSPICIOUS": 1}
POSITIVE_CLASS_NAME = "POTENTIAL_FRAUD_PROXY"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_pipeline(trees: int, max_depth: int, random_state: int) -> Pipeline:
    numeric_features = [name for name in FEATURES if name in NUMERIC_FEATURES]
    categorical_features = [name for name in FEATURES if name not in NUMERIC_FEATURES]
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
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            n_jobs=min(4, os.cpu_count() or 1),
            random_state=random_state,
        )),
    ])


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(payload, temporary, indent=2, ensure_ascii=False, allow_nan=False)
        temporary.write("\n")
    os.replace(temporary_path, path)


def _atomic_write_model(path: Path, pipeline: Pipeline) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        delete=False,
        suffix=".tmp",
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        joblib.dump(pipeline, temporary_path, compress=3)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    if args.trees < 1:
        raise ValueError("--trees harus minimal 1.")
    if args.max_depth < 1:
        raise ValueError("--max-depth harus minimal 1.")
    if not 0.05 <= args.test_fraction <= 0.5:
        raise ValueError("--test-fraction harus berada di rentang 0.05 sampai 0.5.")

    start = time.perf_counter()
    transactions_path = DATA_DIR / DATA_FILES["transactions"]
    labels = pd.read_csv(
        transactions_path,
        usecols=["transaction_id", TARGET_COLUMN],
        low_memory=False,
    )
    if labels["transaction_id"].duplicated().any():
        raise ValueError("transaction_id duplikat; training dibatalkan.")

    unsupported = sorted(set(labels[TARGET_COLUMN].dropna().astype(str)) - set(TARGET_MAP) - {"EDGE_CASE"})
    if unsupported:
        raise ValueError(
            f"Label {TARGET_COLUMN} tidak dikenali: {unsupported}. "
            f"Label yang didukung: NORMAL, SUSPICIOUS, EDGE_CASE."
        )
    if labels[TARGET_COLUMN].isna().any():
        raise ValueError("Label scenario_label kosong; training dibatalkan.")

    data = get_dataset(force_reload=True)
    data = data.merge(
        labels,
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )
    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Ada transaksi yang tidak memiliki label; training dibatalkan.")

    edge_case_count = int(data[TARGET_COLUMN].eq("EDGE_CASE").sum())
    labeled = data[data[TARGET_COLUMN].isin(TARGET_MAP)].copy()
    if labeled.empty:
        raise ValueError("Tidak ada baris NORMAL/SUSPICIOUS untuk training.")
    labeled["_target"] = labeled[TARGET_COLUMN].map(TARGET_MAP).astype("int8")

    valid_time = labeled["_tx_time"].notna()
    labeled = labeled.loc[valid_time].copy()
    unique_times = np.sort(labeled["_tx_time"].unique())
    if len(unique_times) < 2:
        raise ValueError("Timestamp transaksi tidak cukup untuk membuat split temporal.")
    split_index = int(len(unique_times) * (1 - args.test_fraction))
    split_index = min(max(split_index, 1), len(unique_times) - 1)
    cutoff = pd.Timestamp(unique_times[split_index])
    train_mask = labeled["_tx_time"] < cutoff
    test_mask = ~train_mask

    train = labeled.loc[train_mask]
    test = labeled.loc[test_mask]
    if train["_target"].nunique() != 2 or test["_target"].nunique() != 2:
        raise ValueError(
            "Split temporal tidak memuat kedua kelas di train dan test. "
            "Periksa sebaran label terhadap waktu sebelum melatih."
        )

    pipeline = build_pipeline(args.trees, args.max_depth, args.random_state)
    fit_started = time.perf_counter()
    pipeline.fit(train[FEATURES], train["_target"])
    fit_seconds = time.perf_counter() - fit_started

    probabilities = pipeline.predict_proba(test[FEATURES])
    classes = list(pipeline.classes_)
    suspicious_index = classes.index(1)
    scores = probabilities[:, suspicious_index]
    predictions = (scores >= 0.5).astype("int8")
    y_test = test["_target"].to_numpy()
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["NORMAL", POSITIVE_CLASS_NAME],
        output_dict=True,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision_potential_fraud_proxy": float(precision_score(y_test, predictions, zero_division=0)),
        "recall_potential_fraud_proxy": float(recall_score(y_test, predictions, zero_division=0)),
        "f1_potential_fraud_proxy": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "average_precision": float(average_precision_score(y_test, scores)),
        "confusion_matrix_labels": ["NORMAL", POSITIVE_CLASS_NAME],
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    model_path = args.model_output.resolve()
    report_path = args.report_output.resolve()
    _atomic_write_model(model_path, pipeline)
    summary: dict[str, Any] = {
        "target": TARGET_COLUMN,
        "target_mapping": TARGET_MAP,
        "positive_class": POSITIVE_CLASS_NAME,
        "positive_class_source_label": "SUSPICIOUS",
        "positive_class_interpretation": (
            "Proxy for potential fraud risk only; the synthetic SUSPICIOUS label is not "
            "a confirmed fraud outcome."
        ),
        "excluded_edge_case_rows": edge_case_count,
        "dropped_invalid_timestamp_rows": int((~valid_time).sum()),
        "dataset_rows": len(data),
        "labeled_rows": len(labeled),
        "feature_count": len(FEATURES),
        "train_rows": len(train),
        "test_rows": len(test),
        "split_method": "chronological unique-timestamp holdout",
        "test_fraction_requested": args.test_fraction,
        "test_start_utc": cutoff.isoformat(),
        "train_class_counts": {
            "NORMAL": int((train["_target"] == 0).sum()),
            POSITIVE_CLASS_NAME: int((train["_target"] == 1).sum()),
        },
        "test_class_counts": {
            "NORMAL": int((test["_target"] == 0).sum()),
            POSITIVE_CLASS_NAME: int((test["_target"] == 1).sum()),
        },
        "threshold_metrics": metrics,
        "random_forest": {
            "n_estimators": args.trees,
            "max_depth": args.max_depth,
            "class_weight": "balanced_subsample",
            "min_samples_leaf": 2,
            "random_state": args.random_state,
        },
        "fit_seconds": round(fit_seconds, 3),
        "total_seconds": round(time.perf_counter() - start, 3),
        "model_output": str(model_path),
        "data_directory": str(DATA_DIR),
        "data_caveat": (
            "Labels and records are synthetic. EDGE_CASE rows were excluded; this model "
            "learns NORMAL versus SUSPICIOUS scenario_label, using SUSPICIOUS only as a "
            "proxy for potential fraud risk, not confirmed fraud."
        ),
    }
    _atomic_write_json(report_path, summary)
    print(json.dumps({
        "model_output": str(model_path),
        "report_output": str(report_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "excluded_edge_case_rows": edge_case_count,
        "metrics": {key: value for key, value in metrics.items() if key != "classification_report"},
        "random_forest": {"trees": args.trees, "max_depth": args.max_depth},
        "fit_seconds": summary["fit_seconds"],
        "total_seconds": summary["total_seconds"],
        "caveat": summary["data_caveat"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
