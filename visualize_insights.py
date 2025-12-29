#!/usr/bin/env python3
"""
generate_medical_insights.py
GENERATES ALL INSIGHT GRAPHS FOR YOUR PROJECT
Safe for Windows + Seaborn + Matplotlib
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------
# SAFE GLOBAL SETTINGS
# --------------------------------------------------------------------
sns.set_theme(style="whitegrid")
plt.rcParams['figure.autolayout'] = True

# Output directory
OUT_DIR = "visuals"
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------------------------
# LOAD & CLEAN DATA
# --------------------------------------------------------------------
df = pd.read_csv("dataset/heart.csv")

# Fix invalid cholesterol = 0
chol_median = df["Cholesterol"].replace(0, np.nan).median()
df["Cholesterol"] = df["Cholesterol"].replace(0, chol_median)

# Human readable target
df["Status"] = df["HeartDisease"].map({1: "Heart Disease", 0: "Normal"})

print("📊 Generating Medical Insight Visualizations...\n")

# --------------------------------------------------------------------
# 🔹 PLOT 1: Age Distribution (Healthy vs Diseased)
# --------------------------------------------------------------------
try:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=df,
        x="Age",
        hue="Status",
        fill=True,
        common_norm=False,
        alpha=0.4
    )
    plt.title("Age Distribution: Healthy vs Heart Disease", fontsize=16)
    plt.xlabel("Age (Years)")
    plt.ylabel("Density")
    path1 = os.path.join(OUT_DIR, "insight_age_dist.png")
    plt.savefig(path1, dpi=300)
    plt.close()
    print(f"✅ Saved: {path1}")

except Exception as e:
    print("❌ Error in PLOT 1:", e)

# --------------------------------------------------------------------
# 🔹 PLOT 2: Chest Pain Type Influence
# --------------------------------------------------------------------
try:
    plt.figure(figsize=(10, 6))
    order = ['ASY', 'NAP', 'ATA', 'TA']
    sns.countplot(
        data=df,
        x='ChestPainType',
        hue='Status',
        order=order
    )
    plt.title("Heart Disease Frequency by Chest Pain Type", fontsize=16)
    plt.xlabel("Chest Pain Type")
    plt.ylabel("Count")

    path2 = os.path.join(OUT_DIR, "insight_chest_pain.png")
    plt.savefig(path2, dpi=300)
    plt.close()
    print(f"✅ Saved: {path2}")

except Exception as e:
    print("❌ Error in PLOT 2:", e)

# --------------------------------------------------------------------
# 🔹 PLOT 3: Risk Factors – Cholesterol vs MaxHR
# --------------------------------------------------------------------
try:
    # Clip extreme cholesterol for cleaner visualization
    df["Cholesterol_clip"] = df["Cholesterol"].clip(0, 400)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="MaxHR",
        y="Cholesterol_clip",
        hue="Status",
        style="Sex",
        s=90,
        alpha=0.8
    )
    plt.title("Risk Analysis: Max Heart Rate vs Cholesterol", fontsize=16)
    plt.xlabel("Max Heart Rate Achieved")
    plt.ylabel("Serum Cholesterol")

    # High cholesterol line
    plt.axhline(y=240, color='red', linestyle='--', alpha=0.4)
    plt.text(75, 250, "High Cholesterol (>240 mg/dL)", color='red')

    path3 = os.path.join(OUT_DIR, "insight_scatter.png")
    plt.savefig(path3, dpi=300)
    plt.close()
    print(f"✅ Saved: {path3}")

except Exception as e:
    print("❌ Error in PLOT 3:", e)

# --------------------------------------------------------------------
# 🔹 PLOT 4: Correlation Heatmap
# --------------------------------------------------------------------
try:
    plt.figure(figsize=(12, 10))

    # convert non-numeric objects
    numeric_df = df.select_dtypes(include=[np.number])

    corr = numeric_df.corr()

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Feature Correlation Matrix", fontsize=16)

    path4 = os.path.join(OUT_DIR, "insight_heatmap.png")
    plt.savefig(path4, dpi=300)
    plt.close()
    print(f"✅ Saved: {path4}")

except Exception as e:
    print("❌ Error in PLOT 4:", e)

# --------------------------------------------------------------------
print("\n🎉 All Medical Insight Graphs Successfully Generated!")
