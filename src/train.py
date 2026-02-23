import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# =========================
# PATH CONFIG
# =========================
DATA_PATH = Path("dataset/comprehensive_tourist_places_india_v19.csv")
MODEL_PATH = Path("models/xgb_ranker.pkl")

# =========================
# LOAD DATA
# =========================
print("Loading dataset...")
# df = pd.read_excel(DATA_PATH)
df = pd.read_csv(DATA_PATH)

print(f"Rows loaded: {len(df)}")

# =========================
# CLEAN COLUMNS
# =========================

# Rename columns for easier use (adjust if names differ)
df = df.rename(columns={
    "City": "city",
    "State": "state",
    "Name": "place_name",
    "Type": "type",
    "Significance": "significance",
    "Google review rating": "rating",
    "Number of google review in lakhs": "review_count",
    "time needed to visit in hrs": "visit_time",
    "Entrance Fee in INR": "fee"
})

# Fill missing values
df["rating"] = df["rating"].fillna(df["rating"].mean())
df["review_count"] = df["review_count"].fillna(0)
df["visit_time"] = df["visit_time"].fillna(df["visit_time"].median())
df["significance"] = df["significance"].fillna("Local")

# Convert review count to numeric log scale
df["log_reviews"] = np.log1p(df["review_count"])

# =========================
# ENCODE CATEGORICAL DATA
# =========================

le_city = LabelEncoder()
df["city_encoded"] = le_city.fit_transform(df["city"].astype(str))

le_type = LabelEncoder()
df["type_encoded"] = le_type.fit_transform(df["type"].astype(str))

le_sig = LabelEncoder()
df["sig_encoded"] = le_sig.fit_transform(df["significance"].astype(str))

# =========================
# CREATE SYNTHETIC TARGET
# =========================
# This simulates "how good a place is"

df["target_score"] = (
    df["rating"] * 0.6 +
    df["log_reviews"] * 0.3 +
    (1 / (df["visit_time"] + 1)) * 0.1
)

# =========================
# SELECT FEATURES
# =========================
features = [
    "city_encoded",
    "type_encoded",
    "sig_encoded",
    "rating",
    "log_reviews",
    "visit_time"
]

X = df[features]
y = df["target_score"]

# =========================
# TRAIN MODEL
# =========================
print("Training XGBoost model...")

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X, y)

# =========================
# SAVE MODEL
# =========================
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": model,
    "label_encoders": {
        "city": le_city,
        "type": le_type,
        "significance": le_sig
    },
    "features": features
}, MODEL_PATH)

print(f"Model saved at: {MODEL_PATH}")
print("Training complete!")
