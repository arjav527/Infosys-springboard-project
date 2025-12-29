import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DATA_PATH = "dataset/heart.csv"
CLEANED_DATA_PATH = "dataset/milestone1_cleaned_heart.csv"
PREPROCESSOR_PATH = "models/preprocessor_v1.pkl"

# Columns in dataset
CATEGORICAL_COLS = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
NUMERICAL_COLS = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak', 'Athero_Score']


# ---------------------------------------------------------
# Utility: ensure directories exist
# ---------------------------------------------------------
def ensure_directories():
    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(PREPROCESSOR_PATH), exist_ok=True)


# ---------------------------------------------------------
# Step 1: Load raw dataset
# ---------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\n📥 Loaded dataset → {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------
# Step 2: Clean invalid values
# ---------------------------------------------------------
def clean_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧼 Cleaning invalid values...")

    zero_chol = (df['Cholesterol'] == 0).sum()
    zero_bp = (df['RestingBP'] == 0).sum()

    print(f"   → Cholesterol zeros found: {zero_chol}")
    print(f"   → RestingBP zeros found:   {zero_bp}")

    # Convert 0 values to NaN
    df['Cholesterol'] = df['Cholesterol'].replace(0, np.nan)
    df['RestingBP'] = df['RestingBP'].replace(0, np.nan)

    # Compute medians safely
    chol_med = df['Cholesterol'].median()
    bp_med = df['RestingBP'].median()

    # Fill missing with median
    df['Cholesterol'] = df['Cholesterol'].fillna(chol_med)
    df['RestingBP'] = df['RestingBP'].fillna(bp_med)

    # Handle cholesterol outliers ( > 500 mg/dL )
    df.loc[df['Cholesterol'] > 500, 'Cholesterol'] = chol_med

    print("   ✅ Invalid and outlier values fixed.")
    print(f"   → Cholesterol median: {chol_med}")
    print(f"   → RestingBP median: {bp_med}")

    return df


# ---------------------------------------------------------
# Step 3: Add engineered features
# ---------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n🧪 Adding engineered features...")

    # Athero_Score
    df['Athero_Score'] = (df['RestingBP'] * df['Cholesterol']) / 100.0

    print("   → Added: Athero_Score = (RestingBP * Cholesterol) / 100")
    return df


# ---------------------------------------------------------
# Step 4: Optimize datatypes
# ---------------------------------------------------------
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    print("\n📦 Optimizing numerical datatypes (float32)...")

    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    print("   ✅ All numerical columns set to float32.")
    return df


# ---------------------------------------------------------
# Step 5: Build preprocessing pipeline
# ---------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    print("\n🧠 Building preprocessing pipeline...")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', dtype=np.float32), CATEGORICAL_COLS)
        ],
        remainder='drop'
    )

    return preprocessor


# ---------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ---------------------------------------------------------
def main():
    ensure_directories()

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Clean data
    df = clean_invalid_values(df)

    # 3. Add Athero Score
    df = add_features(df)

    # 4. Optimize datatypes
    df = optimize_dtypes(df)

    # 5. Build & fit preprocessor
    print("\n📐 Fitting preprocessor on dataset...")

    feature_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    missing = [c for c in feature_cols if c not in df.columns]

    if missing:
        raise ValueError(f"❌ Missing required columns: {missing}")

    preprocessor = build_preprocessor()
    preprocessor.fit(df[feature_cols])

    print("   ✅ Preprocessor fitted successfully.")

    # 6. Save cleaned data
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"\n💾 Cleaned dataset saved → {CLEANED_DATA_PATH}")

    # 7. Save preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"💾 Preprocessor saved → {PREPROCESSOR_PATH}")

    print("\n🎉 MILESTONE 1 COMPLETED SUCCESSFULLY\n"
          "   → Dataset cleaned\n"
          "   → Feature engineered\n"
          "   → Preprocessor ready for model training\n")


if __name__ == "__main__":
    main()
