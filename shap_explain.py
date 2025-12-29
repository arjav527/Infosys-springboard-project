#!/usr/bin/env python3
"""
SHAP EXPLAINER (FIXED VERSION FOR PIPELINE WITH CATEGORICAL COLUMNS)
This script:
- Loads trained model pipeline
- Extracts internal preprocessor
- Converts X_test → numeric array
- Runs SHAP safely (KernelExplainer)
- Produces summary & per-sample plots
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


MODEL_PATH = "models/final_heart_model.pkl"
DATA_PATH  = "dataset/milestone1_cleaned_heart.csv"
OUT_DIR    = "models"

TOP_K = 12
BACKGROUND_SIZE = 50


# ---------------------------------------------------------
# 🔹 LOAD MODEL + DATA
# ---------------------------------------------------------
def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found:", MODEL_PATH)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Data not found:", DATA_PATH)

    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    return model, df


def prepare_data(df):
    target = "HeartDisease"
    X = df.drop(columns=[target])
    y = df[target]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)


# ---------------------------------------------------------
# 🔹 FIX: Preprocess FIRST → SHAP gets numeric only
# ---------------------------------------------------------
def preprocess_data(model, X_df):
    preprocessor = model.named_steps["preprocessor"]

    # Extract feature names after transformation
    try:
        feature_names = preprocessor.get_feature_names_out()
    except:
        feature_names = [f"f{i}" for i in range(preprocessor.transform(X_df[:1]).shape[1])]

    X_transformed = preprocessor.transform(X_df)
    X_transformed = np.array(X_transformed)

    return X_transformed, feature_names


# ---------------------------------------------------------
# 🔹 BUILD SHAP (NUMERIC SAFE)
# ---------------------------------------------------------
def build_shap(model, X_transformed):
    # Taking small background (KernelExplainer requirement)
    idx = np.random.choice(len(X_transformed), min(BACKGROUND_SIZE, len(X_transformed)), replace=False)
    background = X_transformed[idx]

    def predict_fn(data):
        return model.named_steps["model"].predict_proba(data)[:, 1]

    print("⚡ Using SHAP KernelExplainer on numeric data... (slow but safe)")
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_transformed)

    return explainer, shap_values


# ---------------------------------------------------------
# 🔹 GLOBAL SUMMARY PLOT
# ---------------------------------------------------------
def plot_global_summary(shap_values, feature_names):
    values = np.abs(shap_values).mean(axis=0)
    idx = np.argsort(values)[::-1][:TOP_K]

    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_names)[idx][::-1], values[idx][::-1])
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "shap_summary.png")
    plt.savefig(out)
    plt.close()
    print("Saved global summary:", out)

    # Save JSON
    top_features = {
        feature_names[i]: float(values[i]) for i in idx
    }
    with open(os.path.join(OUT_DIR, "shap_top_features.json"), "w") as f:
        json.dump(top_features, f, indent=2)

    print("Saved shap_top_features.json")


# ---------------------------------------------------------
# 🔹 LOCAL PLOTS (3 examples)
# ---------------------------------------------------------
def plot_local(shap_values, feature_names, X_transformed):
    for i in range(min(3, len(X_transformed))):
        try:
            plt.figure(figsize=(8, 5))
            shap.bar_plot(shap_values[i], feature_names=feature_names)
            out = os.path.join(OUT_DIR, f"shap_local_{i}.png")
            plt.savefig(out)
            plt.close()
            print("Saved local SHAP:", out)
        except Exception as e:
            print("Local plot error:", e)


# ---------------------------------------------------------
# 🔹 MAIN
# ---------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("📌 Loading model + dataset...")
    model, df = load_resources()
    X_test, _ = prepare_data(df)

    print("📌 Preprocessing data to numeric...")
    X_num, feature_names = preprocess_data(model, X_test)

    print("📌 Building SHAP explainability...")
    explainer, shap_values = build_shap(model, X_num)

    print("📌 Creating global summary...")
    plot_global_summary(shap_values, feature_names)

    print("📌 Creating local explanations...")
    plot_local(shap_values, feature_names, X_num)

    print("\n✅ SHAP explainability complete!")
    print("Check generated files in the 'models' folder.")


if __name__ == "__main__":
    main()
