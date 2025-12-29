#!/usr/bin/env python3
"""
milestone2_train.py
Robust training script for Heart Disease ensemble model.

Features:
- Uses cleaned dataset if available
- Reuses saved preprocessor if present (models/preprocessor_v1.pkl)
- Windows-safe (n_jobs=1)
- Calibrated probabilities via CalibratedClassifierCV
- Cross-validated ROC-AUC summary (5-fold)
- Saves model and metrics
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer


# -----------------------
# Paths / config
# -----------------------
CLEANED_PATH = "dataset/milestone1_cleaned_heart.csv"
RAW_PATH_1 = "dataset/heart.csv"
RAW_PATH_2 = "heart.csv"

PREPROCESSOR_PATH = "models/preprocessor_v1.pkl"
MODEL_PATH = "models/final_heart_model.pkl"
METRICS_PATH = "models/train_metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.15
CV_FOLDS = 5

CATEGORICAL_COLS = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
NUMERICAL_COLS = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Athero_Score']
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS
TARGET_COL = "HeartDisease"


# -----------------------
# Helpers
# -----------------------
def load_data():
    if os.path.exists(CLEANED_PATH):
        print("📥 Using cleaned dataset:", CLEANED_PATH)
        return pd.read_csv(CLEANED_PATH)
    if os.path.exists(RAW_PATH_1):
        print("📥 Using dataset:", RAW_PATH_1)
        return pd.read_csv(RAW_PATH_1)
    if os.path.exists(RAW_PATH_2):
        print("📥 Using dataset:", RAW_PATH_2)
        return pd.read_csv(RAW_PATH_2)
    raise FileNotFoundError("No dataset found. Put heart.csv in dataset/ or project root.")


def load_or_build_preprocessor(feature_cols):
    # If preprocessor exists, use it for consistent inference
    if os.path.exists(PREPROCESSOR_PATH):
        try:
            pre = joblib.load(PREPROCESSOR_PATH)
            print("♻️ Loaded existing preprocessor:", PREPROCESSOR_PATH)
            return pre, True
        except Exception as e:
            print("⚠️ Could not load preprocessor, will build a new one:", e)

    # Build new preprocessor (safe defaults)
    print("🧱 Building new preprocessor (KNNImputer + RobustScaler for numeric, mode + OHE for cat).")
    numeric_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', RobustScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    pre = ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERICAL_COLS),
        ('cat', categorical_transformer, CATEGORICAL_COLS)
    ], remainder='drop')
    return pre, False


def build_voting_ensemble():
    # Windows-safe: n_jobs=1
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        n_jobs=1,
        random_state=RANDOM_STATE
    )
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        class_weight='balanced',
        n_jobs=1,
        random_state=RANDOM_STATE
    )
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_depth=10,
        random_state=RANDOM_STATE
    )

    voting = VotingClassifier(
        estimators=[('rf', rf), ('et', et), ('hgb', hgb)],
        voting='soft',
        n_jobs=1
    )
    return voting


def save_metrics(metrics: dict):
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print("📊 Metrics saved to", METRICS_PATH)


# -----------------------
# Main
# -----------------------
def main():
    print("🚀 Starting training pipeline...")

    df = load_data()

    # Basic cleaning fallback: replace zeros with medians (safety)
    if 'Cholesterol' in df.columns:
        df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan).fillna(df['Cholesterol'].median())
    if 'RestingBP' in df.columns:
        df['RestingBP'] = df['RestingBP'].replace(0, np.nan).fillna(df['RestingBP'].median())

    # Ensure Athero_Score exists
    if 'Athero_Score' not in df.columns:
        df['Athero_Score'] = (df['RestingBP'].fillna(df['RestingBP'].median()) *
                              df['Cholesterol'].fillna(df['Cholesterol'].median())) / 100.0

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Validate columns exist
    missing = [c for c in FEATURE_COLS if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # Load or build preprocessor
    preprocessor, loaded = load_or_build_preprocessor(FEATURE_COLS)
    if not loaded:
        # fit preprocessor on entire dataset feature columns
        preprocessor.fit(X[FEATURE_COLS])
        # persist it for later inference consistency
        os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        print("💾 New preprocessor saved to", PREPROCESSOR_PATH)

    # Build base ensemble
    base_model = build_voting_ensemble()

    # Wrap with calibration to get better probabilities
    print("⚖️ Wrapping ensemble with CalibratedClassifierCV (sigmoid) for probability calibration.")
    calibrated = CalibratedClassifierCV(base_model, cv=3, method='sigmoid')

    # Final pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', calibrated)
    ])

    # Train/Test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X[FEATURE_COLS], y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("🧠 Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print("💾 Trained model saved to", MODEL_PATH)

    # Evaluate on test set
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    test_acc = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_proba) if y_test_proba is not None else None

    print("\n------------------ EVALUATION ------------------")
    print(f"🎯 Train Accuracy: {train_acc:.4f}")
    print(f"🏆 Test Accuracy:  {test_acc:.4f}")
    if roc_auc is not None:
        print(f"💙 ROC-AUC:       {roc_auc:.4f}")
    print("------------------------------------------------\n")
    print("Classification Report (Test):\n")
    print(classification_report(y_test, y_test_pred))

    # Cross-validated ROC-AUC (on whole data, using pipeline with preprocessor only)
    try:
        print("\n🔁 Cross-validating ROC-AUC (stratified KFold)...")
        # cross_val_score wants an estimator that supports fit/predict; use pipeline without calibration wrapper for CV speed
        cv_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', build_voting_ensemble())])
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(cv_pipeline, X[FEATURE_COLS], y, scoring='roc_auc', cv=cv, n_jobs=1)
        print(f"CV ROC-AUC scores: {cv_scores}")
        print("CV ROC-AUC mean:", np.mean(cv_scores))
    except Exception as e:
        print("⚠️ Cross-validation failed:", e)
        cv_scores = []

    # Save metrics
    metrics = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "cv_roc_auc_scores": [float(s) for s in cv_scores] if len(cv_scores) else [],
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "feature_columns": FEATURE_COLS
    }
    save_metrics(metrics)

    print("\n✅ Training complete. Model + metrics saved.")


if __name__ == "__main__":
    main()
