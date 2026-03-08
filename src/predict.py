import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/xgb_ranker.pkl"
DATA_PATH = BASE_DIR / "dataset/final_merged_tourism_dataset.csv"

# Load Model Bundle
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

bundle = load_bundle()

def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df

def get_ranked_places(location_query, start_date=None):
    df = load_data()
    if df.empty or bundle is None:
        return []

    # Filter by City OR State (Requirement #5)
    location_query = str(location_query).lower().strip()
    mask = (df['City'].str.lower().str.contains(location_query, na=False)) | \
           (df['State'].str.lower().str.contains(location_query, na=False))
    
    filtered_df = df[mask].copy()

    if filtered_df.empty:
        return []

    # Feature Engineering for Prediction (Must match train.py)
    model = bundle['model']
    features = bundle['features']
    
    # Pre-process columns used in features
    filtered_df['popularity_norm'] = filtered_df['Popularity Index (0-100)'].fillna(0) / 100
    filtered_df['tourism_score_norm'] = filtered_df['Tourism Score (1-10)'].fillna(0) / 10
    filtered_df['crowd_norm'] = filtered_df['Crowd_Level'].fillna(0) / 10
    filtered_df['rating'] = filtered_df['Google review rating'].fillna(0)
    filtered_df['nightlife_score'] = filtered_df['Nightlife Score (0-10)'].fillna(0)
    
    # Extract list counts (market/rest)
    filtered_df['market_score'] = filtered_df['Famous Market'].apply(lambda x: len(str(x).split(';')) if pd.notnull(x) and str(x).strip() != "" else 0)
    filtered_df['restaurant_score'] = filtered_df['Famous Restaurant'].apply(lambda x: len(str(x).split(';')) if pd.notnull(x) and str(x).strip() != "" else 0)
    
    # Handling Categorical Encoding
    # We use the encoders from the training bundle. Handle unseen labels gracefully.
    def safe_encode(le, series):
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        return series.apply(lambda x: mapping.get(x, mapping.get('Unknown', 0)))

    filtered_df['city_encoded'] = safe_encode(bundle['le_city'], filtered_df['City'].fillna('Unknown'))
    filtered_df['state_encoded'] = safe_encode(bundle['le_state'], filtered_df['State'].fillna('Unknown'))
    filtered_df['type_encoded'] = safe_encode(bundle['le_type'], filtered_df['Type'].fillna('Unknown'))
    filtered_df['budget_encoded'] = safe_encode(bundle['le_budget'], filtered_df['Budget Level'].fillna('Budget'))

    # Fill Numeric NAs
    filtered_df['time needed to visit in hrs'] = filtered_df['time needed to visit in hrs'].fillna(2.0)
    filtered_df['Recommended_Duration_Min'] = filtered_df['Recommended_Duration_Min'].fillna(1.0)
    
    def get_cost(val):
        try:
            if pd.isna(val): return 300
            import re
            nums = re.findall(r'\d+', str(val).replace(',', ''))
            return int(nums[0]) if nums else 300
        except:
            return 300
    filtered_df['Avg_Local_Transport_Cost'] = filtered_df['Avg_Local_Transport_Cost'].apply(get_cost)

    # Predict ML scores
    X = filtered_df[features]
    filtered_df['ml_score'] = model.predict(X)

    # Seasonal Awareness (Optional logic from previous turns if desired, though not in latest reqs)
    if start_date:
        try:
            # Simple month-based season matching
            from datetime import datetime
            dt = datetime.strptime(str(start_date), "%Y-%m-%d")
            month = dt.month
            
            # Map month to season
            if 3 <= month <= 6: current_season = "Summer"
            elif 7 <= month <= 9: current_season = "Monsoon"
            else: current_season = "Winter"
            
            # Penalty for season mismatch
            def seasonal_penalty(row):
                best_s = str(row.get('Best Season', 'All')).strip()
                if best_s == 'All' or current_season in best_s:
                    return row['ml_score']
                return row['ml_score'] * 0.7
            
            filtered_df['ml_score'] = filtered_df.apply(seasonal_penalty, axis=1)
        except:
            pass

    # Sort and take top 25 (Requirement #5)
    ranked = filtered_df.sort_values(by='ml_score', ascending=False).head(25)

    # Sanitize NaN values for JSON
    ranked = ranked.fillna({
        "Name": "Unknown",
        "City": "Unknown",
        "State": "Unknown",
        "Type": "Unknown",
        "Google review rating": 0.0,
        "time needed to visit in hrs": 2.0,
        "Latitude": 0.0,
        "Longitude": 0.0,
        "Maps": "",
        "Place_Image_URL": "",
        "Short_Description": "",
        "Significance": "",
        "Travel Tip": "",
        "Must Try Food": "",
        "Food Specialty": "",
        "Famous Market": "",
        "Famous Restaurant": "",
        "Crowd_Level": 5,
        "Best_Visit_Time": "Anytime",
        "Peak_Hours": "N/A",
        "Packing_Suggestions": "",
        "Local_Transport_Options": "",
        "Nearest_Metro_Station": "",
        "Recommended_Duration_Min": 1,
        "Recommended_Duration_Max": 3,
        "Day Priority": 1,
        "Time Slot": "Morning",
        "Entrance Fee in INR": 0,
        "Avg_Adventure_Base_Price": 0,
        "Place_Description_AI": "",
        "Common_Questions": "",
        "Nearest Airport": "Not Available",
        "Major Railway Station": "Not Available"
    })

    # Format output fields (Requirement #5)
    results = []
    for _, row in ranked.iterrows():
        results.append({
            "place_name": row["Name"],
            "city": row["City"],
            "state": row["State"],
            "type": row["Type"],
            "rating": float(row["Google review rating"]),
            "visit_time": float(row["time needed to visit in hrs"]),
            "latitude": float(row["Latitude"]),
            "longitude": float(row["Longitude"]),
            "map_link": row["Maps"],
            "place_image_url": row["Place_Image_URL"],
            "short_description": row["Short_Description"],
            "significance": row["Significance"],
            "travel_tip": row["Travel Tip"],
            "must_try_food": row["Must Try Food"],
            "food_specialty": row["Food Specialty"],
            "famous_market": row["Famous Market"],
            "famous_restaurant": row["Famous Restaurant"],
            "crowd_level": int(row["Crowd_Level"]),
            "best_visit_time": row["Best_Visit_Time"],
            "peak_hours": row["Peak_Hours"],
            "packing_suggestions": row["Packing_Suggestions"],
            "local_transport_options": row["Local_Transport_Options"],
            "nearest_metro_station": row["Nearest_Metro_Station"],
            "avg_local_transport_cost": int(row.get("Avg_Local_Transport_Cost", 300)),
            "ml_score": float(row["ml_score"]),
            "recommended_duration_min": int(row["Recommended_Duration_Min"]),
            "recommended_duration_max": int(row["Recommended_Duration_Max"]),
            "day_priority": int(row["Day Priority"]),
            "time_slot": row["Time Slot"],
            "entrance_fee": int(row["Entrance Fee in INR"]),
            "adventure_price": int(row["Avg_Adventure_Base_Price"]),
            "place_description_ai": row["Place_Description_AI"],
            "common_questions": row["Common_Questions"],
            "airport": row["Nearest Airport"],
            "railway": row["Major Railway Station"]
        })
    
    return results