"""
Train AML Random Forest from FCIP_AML_SAMPLED_22500.csv.
Run from this folder: python train_aml_random_forest.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

BASE = Path(__file__).resolve().parent
df = pd.read_csv(BASE / "FCIP_AML_SAMPLED_22500.csv")
y = df["aml_label"].map({"NORMAL": 0, "SUSPICIOUS": 1})
if y.isna().any():
    raise ValueError("Unexpected values in aml_label.")

excluded = [
    "transaction_id", "source_transaction_reference", "idempotency_key",
    "account_id", "entity_id", "counterparty_id", "merchant_id", "device_id",
    "ip_address", "batch_id", "synthetic_pattern", "scenario_label",
    "aml_type", "aml_label"
]
X = df.drop(columns=[c for c in excluded if c in df.columns]).copy()

for col in ["value_datetime_utc", "business_date", "created_at"]:
    if col in X:
        X[col] = pd.to_datetime(X[col], errors="coerce", utc=True)
if "value_datetime_utc" in X:
    X["transaction_hour"] = X["value_datetime_utc"].dt.hour
    X["transaction_dayofweek"] = X["value_datetime_utc"].dt.dayofweek
    X["transaction_month"] = X["value_datetime_utc"].dt.month
if "business_date" in X:
    X["business_dayofweek"] = X["business_date"].dt.dayofweek
    X["business_month"] = X["business_date"].dt.month
X = X.drop(columns=[c for c in ["value_datetime_utc", "business_date", "created_at"] if c in X])
for col in X.select_dtypes(include=["bool"]).columns:
    X[col] = X[col].astype("string")

num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
prep = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])
pipeline = Pipeline([
    ("preprocessor", prep),
    ("classifier", RandomForestClassifier(
        n_estimators=300, max_depth=16, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    ))
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)
proba = pipeline.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred, target_names=["NORMAL", "SUSPICIOUS"]))
print("Confusion matrix (NORMAL, SUSPICIOUS):")
print(confusion_matrix(y_test, pred, labels=[0, 1]))
print("ROC-AUC:", roc_auc_score(y_test, proba))
joblib.dump(pipeline, BASE / "aml_random_forest_pipeline.joblib")
print("Saved:", BASE / "aml_random_forest_pipeline.joblib")
