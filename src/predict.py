import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models/xgb_ranker.pkl"
DATA_PATH = BASE_DIR / "dataset/cleaned_pan_india_tourism_dataset_updated.csv"

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
label_encoders = bundle["label_encoders"]
features = bundle["features"]


def load_data():

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

    df["market_score"] = df["famous_market"].fillna("").apply(
        lambda x: len(str(x).split(",")) if str(x).strip() else 0
    )

    return df


def encode_features(df):

    for col, encoder in label_encoders.items():

        if col in df.columns:

            df[col] = df[col].astype(str)

            df[col + "_encoded"] = df[col].apply(
                lambda x: encoder.transform([x])[0]
                if x in encoder.classes_
                else 0
            )

    return df


def get_ranked_places(location_query, top_k=25):

    df = load_data()

    location_query = location_query.lower().strip()

    filtered_df = df[
        (df["city"].str.lower() == location_query) |
        (df["state"].str.lower() == location_query)
    ].copy()

    if filtered_df.empty:
        filtered_df = df[
            df["city"].str.lower().str.contains(location_query) |
            df["state"].str.lower().str.contains(location_query)
        ].copy()

    if filtered_df.empty:
        return []

    filtered_df = encode_features(filtered_df)

    filtered_df = filtered_df.drop_duplicates(subset=["place_name"])

    for col in features:
        if col not in filtered_df.columns:
            filtered_df[col] = 0

    X = filtered_df[features]

    filtered_df["ml_score"] = model.predict(X)

    return filtered_df.sort_values(
        by="ml_score",
        ascending=False
    ).head(top_k).to_dict(orient="records")