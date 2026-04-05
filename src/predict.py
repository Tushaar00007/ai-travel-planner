import pandas as pd
import numpy as np
import joblib
import os
import requests as http_requests
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "")
_photo_cache = {}  # module-level cache to avoid repeat API calls

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/xgb_ranker.pkl"
DATA_PATH = BASE_DIR / "dataset/tourism_dataset_enriched_v3.csv"

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


def fetch_google_photo_url(place_name, city):
    """Fetch a Google Places photo URL for the given place. Results are cached."""
    cache_key = f"{place_name}_{city}"
    if cache_key in _photo_cache:
        return _photo_cache[cache_key]

    if not GOOGLE_PLACES_API_KEY:
        return ""

    try:
        search_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        params = {
            "input": f"{place_name} {city} India",
            "inputtype": "textquery",
            "fields": "photos",
            "key": GOOGLE_PLACES_API_KEY
        }
        res = http_requests.get(search_url, params=params, timeout=3)
        data = res.json()

        candidates = data.get("candidates", [])
        if candidates and candidates[0].get("photos"):
            photo_ref = candidates[0]["photos"][0]["photo_reference"]
            photo_url = (
                f"https://maps.googleapis.com/maps/api/place/photo"
                f"?maxwidth=800&photoreference={photo_ref}"
                f"&key={GOOGLE_PLACES_API_KEY}"
            )
            _photo_cache[cache_key] = photo_url
            return photo_url
    except Exception as e:
        print(f"[predict.py] Google photo fetch failed for {place_name}: {e}")

    _photo_cache[cache_key] = ""
    return ""

def get_ranked_places(location_query, start_date=None, preferences=None):
    df = load_data()
    if df.empty or bundle is None:
        return []

    # Filter by City OR State (Requirement #5)
    location_parts = [p.strip().lower() for p in str(location_query).split(',')]
    generic_terms = {'india'}
    query_parts = [p for p in location_parts if p and p not in generic_terms]
    
    if not query_parts:
        return []

    mask = pd.Series(False, index=df.index)
    for part in query_parts:
        mask |= (df['City'].str.lower().str.contains(part, na=False)) | \
               (df['State'].str.lower().str.contains(part, na=False))
    
    filtered_df = df[mask].copy()

    # Detect rows where Ola/Uber is not available (string "NA" in dataset)
    OLA_UBER_COLS = [
        "Ola_Car_Base", "Ola_Car_PerKm",
        "Uber_Car_Base", "Uber_Car_PerKm",
        "Uber_Auto_Min", "Uber_Auto_PerKm"
    ]
    no_ola_uber_mask = filtered_df[OLA_UBER_COLS].apply(
        lambda col: col.astype(str).str.strip() == "NA"
    ).any(axis=1)

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
    
    # preference_count = number of comma-separated values in the Trip_Preference_Tags column
    filtered_df['preference_count'] = filtered_df['Trip_Preference_Tags'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 1)
    
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

    # Soft preference boosting — uses score multipliers, never removes rows,
    # so we always return results even when preferences don't match perfectly.
    if preferences:
        style = preferences.get("style", "")
        pref_list = []
        if style and style.lower() not in ["any", "all", ""]:
            pref_list.append(style)
        if preferences.get("includeNightlife") == True:
            pref_list.append("Nightlife")
        if preferences.get("includeFood") == True:
            pref_list.append("Foodie")

        # avoidCrowds: penalise high-crowd places instead of removing them
        if preferences.get("avoidCrowds") == True:
            filtered_df['ml_score'] = filtered_df.apply(
                lambda row: row['ml_score'] * 0.5
                if pd.notna(row['Crowd_Level']) and row['Crowd_Level'] > 6
                else row['ml_score'],
                axis=1
            )

        # Style / nightlife / food: boost matching places 1.5×
        if pref_list:
            pref_mask = filtered_df['Trip_Preference_Tags'].apply(
                lambda x: any(p.strip().lower() in str(x).lower() for p in pref_list)
            )
            filtered_df['ml_score'] = filtered_df.apply(
                lambda row: row['ml_score'] * 1.5
                if pref_mask[row.name] else row['ml_score'],
                axis=1
            )

        # Budget: boost places matching the chosen budget tier 1.3×
        budget = preferences.get("budget", "")
        if budget:
            budget_map = {
                "budget": ["Low", "Budget"],
                "mid": ["Mid-Range", "Medium"],
                "luxury": ["High", "Luxury"]
            }
            for key, values in budget_map.items():
                if key in budget.lower():
                    budget_mask = filtered_df['Budget Level'].isin(values)
                    filtered_df['ml_score'] = filtered_df.apply(
                        lambda row: row['ml_score'] * 1.3
                        if budget_mask[row.name] else row['ml_score'],
                        axis=1
                    )

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
        "Major Railway Station": "Not Available",
        "Auto_Fare_Min": "N/A",
        "Auto_Fare_PerKm": "N/A",
        "Rapido_Bike_Min": "N/A",
        "Rapido_Bike_PerKm": "N/A",

        "City_Taxi_PerKm": "N/A",
        "Transport_Fare_Note": "N/A",
        "Surge_Pricing_Note": "N/A",
        "Venue_Type": "N/A",
        "Cover_Charge": "N/A",
        "Avg_Drinks_Price": "N/A",
        "Music_Genre": "N/A",
        "Club_Timings": "N/A",
        "Dress_Code": "N/A",
        "Ladies_Night_Info": "N/A",
        "Nightlife_Note": "N/A",
        "Itinerary_Role": "N/A"
    })

    # Separately fill NaN (not string "NA") in Ola/Uber cols so JSON serialization won't break
    for col in OLA_UBER_COLS:
        if col in ranked.columns:
            ranked[col] = ranked[col].fillna("N/A")

    # Helper: convert dataset "NA" or fallback "N/A" strings to user-friendly label
    def fmt_ola_uber(val):
        v = str(val).strip()
        return "Not Available" if v in ("NA", "N/A", "nan") else val

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
            "railway": row["Major Railway Station"],
            "itinerary_role": row["Itinerary_Role"],
            "auto_fare_min": row["Auto_Fare_Min"],
            "auto_fare_per_km": row["Auto_Fare_PerKm"],
            "rapido_bike_min": row["Rapido_Bike_Min"],
            "rapido_bike_per_km": row["Rapido_Bike_PerKm"],
            "ola_car_base": fmt_ola_uber(row["Ola_Car_Base"]),
            "ola_car_per_km": fmt_ola_uber(row["Ola_Car_PerKm"]),
            "uber_car_base": fmt_ola_uber(row["Uber_Car_Base"]),
            "uber_car_per_km": fmt_ola_uber(row["Uber_Car_PerKm"]),
            "uber_auto_min": fmt_ola_uber(row["Uber_Auto_Min"]),
            "uber_auto_per_km": fmt_ola_uber(row["Uber_Auto_PerKm"]),
            "city_taxi_per_km": row["City_Taxi_PerKm"],
            "surge_pricing_note": row["Surge_Pricing_Note"],
            "transport_fare_note": row["Transport_Fare_Note"],
            "venue_type": row["Venue_Type"],
            "cover_charge": row["Cover_Charge"],
            "avg_drinks_price": row["Avg_Drinks_Price"],
            "music_genre": row["Music_Genre"],
            "club_timings": row["Club_Timings"],
            "dress_code": row["Dress_Code"],
            "ladies_night_info": row["Ladies_Night_Info"],
            "nightlife_note": row["Nightlife_Note"],
            "photo_url": ""  # placeholder, filled in parallel below
        })
    
    # Fetch Google Photos in parallel (non-blocking batch)
    if results and GOOGLE_PLACES_API_KEY:
        def _fetch_photo(idx, place_name, city):
            return idx, fetch_google_photo_url(place_name, city)
        
        try:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(_fetch_photo, i, r["place_name"], r["city"])
                    for i, r in enumerate(results)
                ]
                for future in as_completed(futures):
                    try:
                        idx, url = future.result(timeout=10)
                        results[idx]["photo_url"] = url
                    except Exception:
                        pass
        except Exception as e:
            print(f"[predict.py] Parallel photo fetch failed: {e}")

    if results:
        print(f"[predict.py] Returning {len(results)} places for '{location_query}'")
        print(f"[predict.py] First result keys: {list(results[0].keys())}")
        print(f"[predict.py] Sample transport fares — auto_fare_min: {results[0].get('auto_fare_min')!r}, surge_pricing_note: {results[0].get('surge_pricing_note')!r}")

    return results