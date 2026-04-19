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
_photo_cache = {}

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/xgb_ranker.pkl"
DATA_PATH = BASE_DIR / "dataset/tourism_dataset_enriched_v4.csv"

# Import Stage 1 and Stage 3 engines
from cluster_engine import get_cluster_filtered_df
from preference_matcher import compute_cosine_scores

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

    # ── Location filter (unchanged) ──────────────────────────────
    location_parts = [p.strip().lower() for p in str(location_query).split(',')]
    generic_terms = {'india'}
    query_parts = [p for p in location_parts if p and p not in generic_terms]
    if not query_parts:
        return []

    mask = pd.Series(False, index=df.index)
    for part in query_parts:
        mask |= (df['City'].str.lower().str.contains(part, na=False)) | \
                (df['State'].str.lower().str.contains(part, na=False))
    location_filtered_df = df[mask].copy()

    if location_filtered_df.empty:
        return []

    print(f"[predict.py] {len(location_filtered_df)} places found "
          f"for location '{location_query}'")

    # ════════════════════════════════════════════════════════════
    # STAGE 1 — KMeans Cluster Filter
    # Narrows candidates to best-matching clusters
    # ════════════════════════════════════════════════════════════
    clustered_df = get_cluster_filtered_df(location_filtered_df, preferences)
    
    # Safety: if clustering removes everything, fall back to location filter
    if clustered_df.empty:
        print("[predict.py] Cluster filter returned empty — using location filter")
        clustered_df = location_filtered_df.copy()

    print(f"[predict.py] After Stage 1 (KMeans): {len(clustered_df)} candidates")

    # ════════════════════════════════════════════════════════════
    # STAGE 2 — XGBoost Quality Ranking
    # Scores candidates by objective quality metrics
    # ════════════════════════════════════════════════════════════
    OLA_UBER_COLS = [
        "Ola_Car_Base", "Ola_Car_PerKm",
        "Uber_Car_Base", "Uber_Car_PerKm",
        "Uber_Auto_Min", "Uber_Auto_PerKm"
    ]

    model = bundle['model']
    features = bundle['features']

    clustered_df['popularity_norm'] = clustered_df['Popularity Index (0-100)'].fillna(0) / 100
    clustered_df['tourism_score_norm'] = clustered_df['Tourism Score (1-10)'].fillna(0) / 10
    clustered_df['crowd_norm'] = clustered_df['Crowd_Level'].fillna(0) / 10
    clustered_df['rating'] = clustered_df['Google review rating'].fillna(0)
    clustered_df['nightlife_score'] = clustered_df['Nightlife Score (0-10)'].fillna(0)
    clustered_df['market_score'] = clustered_df['Famous Market'].apply(
        lambda x: len(str(x).split(';')) if pd.notnull(x) and str(x).strip() != "" else 0)
    clustered_df['restaurant_score'] = clustered_df['Famous Restaurant'].apply(
        lambda x: len(str(x).split(';')) if pd.notnull(x) and str(x).strip() != "" else 0)
    clustered_df['preference_count'] = clustered_df['Trip_Preference_Tags'].apply(
        lambda x: len(str(x).split(',')) if pd.notnull(x) else 1)

    def safe_encode(le, series):
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        return series.apply(lambda x: mapping.get(x, mapping.get('Unknown', 0)))

    clustered_df['city_encoded'] = safe_encode(
        bundle['le_city'], clustered_df['City'].fillna('Unknown'))
    clustered_df['state_encoded'] = safe_encode(
        bundle['le_state'], clustered_df['State'].fillna('Unknown'))
    clustered_df['type_encoded'] = safe_encode(
        bundle['le_type'], clustered_df['Type'].fillna('Unknown'))
    clustered_df['budget_encoded'] = safe_encode(
        bundle['le_budget'], clustered_df['Budget Level'].fillna('Budget'))

    clustered_df['time needed to visit in hrs'] = \
        clustered_df['time needed to visit in hrs'].fillna(2.0)
    clustered_df['Recommended_Duration_Min'] = \
        clustered_df['Recommended_Duration_Min'].fillna(1.0)

    def get_cost(val):
        try:
            if pd.isna(val): return 300
            import re
            nums = re.findall(r'\d+', str(val).replace(',', ''))
            return int(nums[0]) if nums else 300
        except:
            return 300
    clustered_df['Avg_Local_Transport_Cost'] = \
        clustered_df['Avg_Local_Transport_Cost'].apply(get_cost)

    X = clustered_df[features]
    clustered_df['xgb_score'] = model.predict(X)

    # Take top 50 by XGBoost score for Stage 3
    top50_df = clustered_df.sort_values(
        by='xgb_score', ascending=False).head(50).copy()

    print(f"[predict.py] After Stage 2 (XGBoost): top 50 selected")

    # ════════════════════════════════════════════════════════════
    # STAGE 3 — Cosine Similarity Preference Re-ranking
    # Finds which of the top 50 best match user preferences
    # ════════════════════════════════════════════════════════════
    top50_df = compute_cosine_scores(top50_df, preferences)

    # ── Seasonal penalty (unchanged from original) ───────────────
    if start_date:
        try:
            from datetime import datetime
            dt = datetime.strptime(str(start_date), "%Y-%m-%d")
            month = dt.month
            if 3 <= month <= 6: current_season = "Summer"
            elif 7 <= month <= 9: current_season = "Monsoon"
            else: current_season = "Winter"

            def seasonal_penalty(row):
                best_s = str(row.get('Best Season', 'All')).strip()
                if best_s == 'All' or current_season in best_s:
                    return row['xgb_score']
                return row['xgb_score'] * 0.7
            top50_df['xgb_score'] = top50_df.apply(seasonal_penalty, axis=1)
        except:
            pass

    # ── Combined final score ─────────────────────────────────────
    # Normalise xgb_score to 0-1 range first
    xgb_min = top50_df['xgb_score'].min()
    xgb_max = top50_df['xgb_score'].max()
    xgb_range = xgb_max - xgb_min if xgb_max != xgb_min else 1
    top50_df['xgb_norm'] = (top50_df['xgb_score'] - xgb_min) / xgb_range

    # Final score: 50% XGBoost quality + 30% cosine preference + 20% cluster bonus
    # cluster bonus: places that were in the TOP matching cluster get +0.2
    top50_df['cluster_bonus'] = top50_df.get(
        '_cluster_id', pd.Series(0, index=top50_df.index)
    ).apply(lambda x: 0.2 if x == 0 else 0.0)  
    # Note: cluster 0 = best match cluster; adjust if needed

    top50_df['final_score'] = (
        (top50_df['xgb_norm']      * 0.50) +
        (top50_df['cosine_score']  * 0.30) +
        (top50_df['cluster_bonus'] * 0.20)
    )

    print(f"[predict.py] Final score formula: "
          f"XGBoost(0.5) + Cosine(0.3) + ClusterBonus(0.2)")

    # ── Final sort and take top 25 ───────────────────────────────
    ranked = top50_df.sort_values(
        by='final_score', ascending=False).head(25)

    # ── Fill NaN for JSON safety (unchanged) ─────────────────────
    ranked = ranked.fillna({
        "Name": "Unknown", "City": "Unknown", "State": "Unknown",
        "Type": "Unknown", "Google review rating": 0.0,
        "time needed to visit in hrs": 2.0, "Latitude": 0.0,
        "Longitude": 0.0, "Maps": "", "Place_Image_URL": "",
        "Short_Description": "", "Significance": "",
        "Travel Tip": "", "Must Try Food": "", "Food Specialty": "",
        "Famous Market": "", "Famous Restaurant": "",
        "Crowd_Level": 5, "Best_Visit_Time": "Anytime",
        "Peak_Hours": "N/A", "Packing_Suggestions": "",
        "Local_Transport_Options": "", "Nearest_Metro_Station": "",
        "Recommended_Duration_Min": 1, "Recommended_Duration_Max": 3,
        "Day Priority": 1, "Time Slot": "Morning",
        "Entrance Fee in INR": 0, "Avg_Adventure_Base_Price": 0,
        "Place_Description_AI": "", "Common_Questions": "",
        "Nearest Airport": "Not Available",
        "Major Railway Station": "Not Available",
        "Auto_Fare_Min": "N/A", "Auto_Fare_PerKm": "N/A",
        "Rapido_Bike_Min": "N/A", "Rapido_Bike_PerKm": "N/A",
        "City_Taxi_PerKm": "N/A", "Transport_Fare_Note": "N/A",
        "Surge_Pricing_Note": "N/A", "Venue_Type": "N/A",
        "Cover_Charge": "N/A", "Avg_Drinks_Price": "N/A",
        "Music_Genre": "N/A", "Club_Timings": "N/A",
        "Dress_Code": "N/A", "Ladies_Night_Info": "N/A",
        "Nightlife_Note": "N/A", "Itinerary_Role": "N/A"
    })

    for col in OLA_UBER_COLS:
        if col in ranked.columns:
            ranked[col] = ranked[col].fillna("N/A")

    def fmt_ola_uber(val):
        v = str(val).strip()
        return "Not Available" if v in ("NA", "N/A", "nan") else val

    # ── Build results list (unchanged fields + new score fields) ──
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
            "avg_local_transport_cost": int(
                row.get("Avg_Local_Transport_Cost", 300)),
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
            # ── NEW: score transparency fields ───────────────────
            "xgb_score": float(row.get("xgb_score", 0)),
            "cosine_score": float(row.get("cosine_score", 0)),
            "final_score": float(row.get("final_score", 0)),
            "photo_url": ""
        })

    # Fetch Google Photos in parallel (unchanged)
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

    print(f"[predict.py] Pipeline complete — returning {len(results)} places")
    print(f"[predict.py] Top 3: {[r['place_name'] for r in results[:3]]}")

    return results