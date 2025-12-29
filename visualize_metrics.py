#!/usr/bin/env python3
"""
generate_model_visuals.py
SAFE VERSION – Works with Pipeline + Calibration + VotingClassifier
Generates:
- Confusion matrix
- ROC curve
- Permutation Importance (correct numeric features)
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

# -------------------------------------------
# INITIAL SETUP
# -------------------------------------------
sns.set_theme(style="whitegrid")

MODEL_PATH = "models/final_heart_model.pkl"
DATA_PATH  = "dataset/heart.csv"
OUT_DIR = "visuals"
os.makedirs(OUT_DIR, exist_ok=True)

print("📊 Loading Data & Model...")

# -------------------------------------------
# LOAD MODEL + DATA
# -------------------------------------------
try:
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print("❌ ERROR loading model/data:", e)
    exit()


# -------------------------------------------
# MATCH TRAINING CLEANING
# -------------------------------------------
df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)
df["RestingBP"]   = df["RestingBP"].replace(0, np.nan)

df["Athero_Score"] = (df["RestingBP"].fillna(130) * df["Cholesterol"].fillna(220)) / 100

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# -------------------------------------------
# PREDICT
# -------------------------------------------
print("📌 Computing Predictions...")
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# -------------------------------------------
# 1) CONFUSION MATRIX
# -------------------------------------------
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"✅ Saved: {cm_path}")

# -------------------------------------------
# 2) ROC CURVE
# -------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, color="darkorange", label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--", lw=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve", fontsize=16)
plt.legend()

roc_path = os.path.join(OUT_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"✅ Saved: {roc_path}")

# -------------------------------------------
# IMPORTANT FIX:
# Get Preprocessed Numeric Data for Permutation Importance
# -------------------------------------------
print("⚙️ Preprocessing X_test for Permutation Importance...")

preprocessor = model.named_steps["preprocessor"]
X_test_transformed = preprocessor.transform(X_test)

# Extract transformed feature names
try:
    feature_names = preprocessor.get_feature_names_out()
except:
    feature_names = [f"f{i}" for i in range(X_test_transformed.shape[1])]

# -------------------------------------------
# 3) PERMUTATION IMPORTANCE
# -------------------------------------------
print("📌 Computing Feature Importance (Safe numeric version)...")

result = permutation_importance(
    model.named_steps["model"],          # calibrated voting classifier
    X_test_transformed,                  # numeric data
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=1
)

sorted_idx = result.importances_mean.argsort()

plt.figure(figsize=(12, 8))
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(feature_names)[sorted_idx]
)
plt.title("Permutation Importance (Model Risk Factors)", fontsize=16)
plt.tight_layout()

fi_path = os.path.join(OUT_DIR, "feature_importance.png")
plt.savefig(fi_path, dpi=300)
plt.close()
print(f"✅ Saved: {fi_path}")

print("\n🎉 All Model Visualizations Successfully Generated!")
