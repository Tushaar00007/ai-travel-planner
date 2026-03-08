import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "dataset/cleaned_pan_india_tourism_dataset_updated.csv"
MODEL_PATH = BASE_DIR / "models/xgb_ranker.pkl"

df = pd.read_csv(DATA_PATH)

df = df.rename(columns={
    "City": "city",
    "State": "state",
    "Name": "place_name",
    "Type": "type",
    "Google review rating": "rating",
    "time needed to visit in hrs": "visit_time",
    "Entrance Fee in INR": "fee",
    "Budget Level": "budget",
    "Nightlife Spot": "nightlife",
    "Nightlife Score (0-10)": "nightlife_score",
    "Is_Shopping": "is_shopping",
    "Is_Beach": "is_beach",
    "Adventure_Available": "adventure_available",
    "Avg_Adventure_Base_Price": "adventure_price",
    "Famous Market": "famous_market",
    "Famous Restaurant": "famous_restaurant",
    "Nearest Airport": "airport",
    "Major Railway Station": "railway",
    "Food Specialty": "food_specialty",
    "Maps": "map_link"
})

df["rating"] = df["rating"].fillna(df["rating"].mean())
df["visit_time"] = df["visit_time"].fillna(df["visit_time"].median())
df["adventure_price"] = df["adventure_price"].fillna(0)
df["nightlife_score"] = df["nightlife_score"].fillna(0)

df["log_rating"] = np.log1p(df["rating"])

df["market_score"] = df["famous_market"].fillna("").apply(
    lambda x: len(str(x).split(",")) if str(x).strip() else 0
)

encoders = {}

for col in [
    "city","state","type","budget",
    "nightlife","is_shopping","is_beach","adventure_available"
]:
    le = LabelEncoder()
    df[col+"_encoded"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

df["target_score"] = (
    df["rating"] * 0.35 +
    df["nightlife_score"] * 0.2 +
    df["market_score"] * 0.15 +
    df["log_rating"] * 0.15 +
    (df["visit_time"] / df["visit_time"].max()) * 0.15
)

features = [
    "city_encoded","state_encoded",
    "type_encoded","budget_encoded",
    "nightlife_encoded","is_shopping_encoded",
    "is_beach_encoded","adventure_available_encoded",
    "rating","nightlife_score","market_score","visit_time"
]

X = df[features]
y = df["target_score"]

model = XGBRegressor(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.04,
    subsample=0.85,
    random_state=42
)

model.fit(X, y)

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": model,
    "label_encoders": encoders,
    "features": features
}, MODEL_PATH)

print("Model saved successfully.")