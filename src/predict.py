import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# =========================
# PATH CONFIG
# =========================
MODEL_PATH = Path("models/xgb_ranker.pkl")
DATA_PATH = Path("dataset/comprehensive_tourist_places_india_v19.csv")

# =========================
# LOAD MODEL (once)
# =========================
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
label_encoders = bundle["label_encoders"]
features = bundle["features"]

le_city = label_encoders["city"]
le_type = label_encoders["type"]
le_sig = label_encoders["significance"]


# =========================
# LOAD DATASET
# =========================
def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.rename(columns={
        "City": "city",
        "State": "state",
        "Name": "place_name",
        "Type": "type",
        "Significance": "significance",
        "Google review rating": "rating",
        "Number of google review in lakhs": "review_count",
        "time needed to visit in hrs": "visit_time",
        "Entrance Fee in INR": "fee",
        "Maps": "map_link" # Mapped the maps column
    })

    df["rating"] = df["rating"].fillna(df["rating"].mean())
    df["review_count"] = df["review_count"].fillna(0)
    df["visit_time"] = df["visit_time"].fillna(df["visit_time"].median())
    df["significance"] = df["significance"].fillna("Local")

    df["log_reviews"] = np.log1p(df["review_count"])

    return df


# =========================
# FEATURE BUILDER
# =========================
def build_features(df):
    # Encode safely (handle unseen values)
    df["city_encoded"] = df["city"].map(
        lambda x: le_city.transform([x])[0] if x in le_city.classes_ else 0
    )

    df["type_encoded"] = df["type"].map(
        lambda x: le_type.transform([x])[0] if x in le_type.classes_ else 0
    )

    df["sig_encoded"] = df["significance"].map(
        lambda x: le_sig.transform([x])[0] if x in le_sig.classes_ else 0
    )

    return df


# =========================
# MAIN FUNCTION
# =========================
def get_ranked_places(city_name: str, top_k: int = 10):
    """
    Returns top ranked places for a given city
    """

    df = load_data()

    # Filter by city
    city_df = df[df["city"].str.lower() == city_name.lower()].copy()

    if city_df.empty:
        return []

    city_df = build_features(city_df)

    X = city_df[features]

    # Predict scores
    scores = model.predict(X)

    city_df["ml_score"] = scores

    # Sort by ML score
    city_df = city_df.sort_values(by="ml_score", ascending=False)

    # ADDED 'map_link' to the results slice here
    results = city_df.head(top_k)[
        ["place_name", "rating", "visit_time", "ml_score", "map_link"]
    ]

    return results.to_dict(orient="records")


# =========================
# TEST RUN
# =========================
if __name__ == "__main__":
    city = "Jaipur"

    places = get_ranked_places(city, top_k=8)

    print(f"\nTop places in {city}:\n")

    for p in places:
        # Added the map link to the print output
        print(
            f"{p['place_name']} | Rating: {p['rating']} | "
            f"Time: {p['visit_time']} hrs | Score: {round(p['ml_score'], 3)}\n"
            f"  Map: {p['map_link']}\n"
        )