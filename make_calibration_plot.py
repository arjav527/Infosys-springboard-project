#!/usr/bin/env python3
"""
make_calibration_plot.py

Generates calibration and reliability plots for the trained pipeline.

Outputs:
- models/calibration_curve.png
- models/reliability_diagram.png
- prints Brier score and basic stats
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

# Paths
MODEL_PATH = "models/final_heart_model.pkl"
DATA_PATH = "dataset/milestone1_cleaned_heart.csv"
OUT_DIR = "models"
CALIB_PLOT = os.path.join(OUT_DIR, "calibration_curve.png")
RELIABILITY_PLOT = os.path.join(OUT_DIR, "reliability_diagram.png")

def load_resources():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Cleaned data not found at {DATA_PATH}. Run milestone1_cleaning.py first.")
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    return model, df

def prepare_data(df):
    target = "HeartDisease"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' missing from dataset.")
    X = df.drop(columns=[target])
    y = df[target]
    # keep same split ratio the training script used
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def plot_calibration(y_true, y_prob, n_bins=10):
    # calibration_curve returns fraction_of_positives and mean_predicted_value
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(8, 6))
    plt.plot(mean_pred, frac_pos, "s-", label="Model")
    plt.plot([0,1], [0,1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(CALIB_PLOT)
    plt.close()
    print(f"Saved calibration curve to: {CALIB_PLOT}")

def plot_reliability(y_true, y_prob, n_bins=10):
    # reliability diagram (binned)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    prob_true = []
    prob_pred = []
    counts = []

    for i in range(n_bins):
        mask = binids == i
        if mask.sum() > 0:
            prob_true.append(y_true[mask].mean())
            prob_pred.append(y_prob[mask].mean())
            counts.append(mask.sum())
        else:
            prob_true.append(np.nan)
            prob_pred.append(np.nan)
            counts.append(0)

    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, prob_true, width=1.0/n_bins * 0.9, alpha=0.6, label="Observed (fraction)")
    plt.plot(bin_centers, prob_pred, "o-", color="red", label="Predicted (mean prob)")
    plt.xlabel("Predicted probability (bin centers)")
    plt.ylabel("Observed fraction / Predicted mean")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RELIABILITY_PLOT)
    plt.close()
    print(f"Saved reliability diagram to: {RELIABILITY_PLOT}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model, df = load_resources()
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Ensure model returns probabilities
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("Loaded model does not implement predict_proba.")

    # Predict probabilities on test set
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        # If pipeline expects specific ordering of columns, try to subset columns
        print("Error calling predict_proba on full X_test:", e)
        # try ordering columns as in training
        feature_cols = [
            'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Athero_Score',
            'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'
        ]
        X_test2 = X_test[feature_cols]
        y_prob = model.predict_proba(X_test2)[:, 1]

    # Brier score
    brier = brier_score_loss(y_test, y_prob)
    print(f"Brier score (lower is better): {brier:.6f}")

    # Calibration curve
    plot_calibration(y_test.values, y_prob, n_bins=10)
    plot_reliability(y_test.values, y_prob, n_bins=10)

    print("Done.")

if __name__ == "__main__":
    main()
