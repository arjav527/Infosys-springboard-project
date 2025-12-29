import pandas as pd
import numpy as np

df = pd.read_csv("dataset/heart.csv")

print("\n----------------------------------------------------")
print("🔍 DATASET SHAPE:", df.shape)
print("----------------------------------------------------\n")

# 1. Check missing values
print("1) Missing Values Per Column:")
print(df.isna().sum())
print("\n----------------------------------\n")

# 2. Check zero values in places where zero is impossible
issues = {}
for col in ["Cholesterol", "RestingBP", "MaxHR", "Age"]:
    zero_count = (df[col] == 0).sum()
    issues[col] = zero_count
    print(f"{col}: {zero_count} zeros")

print("\n----------------------------------\n")

# 3. Check value ranges (medical sanity)
print("2) MEDICAL RANGE CHECKS\n")

def check_range(col, min_v, max_v):
    bad = df[(df[col] < min_v) | (df[col] > max_v)]
    if len(bad) > 0:
        print(f"❌ {col}: {len(bad)} out-of-range values")
        print(bad[col].value_counts().head())
    else:
        print(f"✅ {col} range OK")

check_range("Age", 18, 100)
check_range("RestingBP", 70, 200)
check_range("Cholesterol", 80, 600)
check_range("MaxHR", 50, 220)

print("\n----------------------------------\n")

# 4. Categorical column value checking
print("3) Categorical Columns Check\n")

cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
for c in cat_cols:
    print(f"→ {c}:", df[c].unique())

print("\n----------------------------------\n")

# 5. Label distribution
print("4) TARGET DISTRIBUTION\n")
print(df["HeartDisease"].value_counts())
print("→ Percent:")
print(df["HeartDisease"].value_counts(normalize=True) * 100)

print("\n----------------------------------\n")

# 6. Duplicate rows
print("5) Duplicate Rows:", df.duplicated().sum())

print("\n----------------------------------\n")

# 7. Summary statistics
print("6) NUMERICAL SUMMARY:")
print(df.describe())

print("\n----------------------------------\n")
print("🔥 ANALYSIS COMPLETED. SHARE OUTPUT HERE.")
