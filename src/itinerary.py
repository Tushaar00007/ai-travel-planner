import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

def calculate_haversine(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0
    R = 6371
    try:
        dlat = math.radians(float(lat2) - float(lat1))
        dlon = math.radians(float(lon2) - float(lon1))
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(float(lat1))) *
             math.cos(math.radians(float(lat2))) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.asin(math.sqrt(a))
    except:
        return 0

def _get_recommended_transport(place, transport_mode):
    mode = str(transport_mode).lower()
    if "rental" in mode:
        return "Scooter/bike rental recommended for flexibility"
    elif "private" in mode or "cab" in mode:
        taxi = place.get('city_taxi_per_km', 'N/A')
        return f"Private cab. Rate approx: {taxi}/km"
    else:
        auto = place.get('auto_fare_min', 'N/A')
        rapido = place.get('rapido_bike_min', 'N/A')
        return f"Auto: {auto} min fare | Rapido: {rapido} min fare"

def _classify_place(place):
    """Classify a place as nightlife, food, or general."""
    tags = str(place.get('itinerary_role', '') + ' ' +
               place.get('type', '') + ' ' +
               place.get('trip_preference_tags', '')).lower()
    
    nightlife_kw = ['nightlife', 'club', 'bar', 'party', 'night', 'pub']
    food_kw = ['restaurant', 'food', 'cafe', 'dhaba', 'market', 'foodie']
    
    if any(k in tags for k in nightlife_kw):
        return 'nightlife'
    if any(k in tags for k in food_kw):
        return 'food'
    return 'general'

def build_pro_itinerary(location, days, ranked_places, 
                         start_date=None, preferences=None):
    if not ranked_places:
        return {"transport": {"airport": "N/A", "railway": "N/A"}, "plan": {}}

    # ── Extract preferences ──────────────────────────────────────
    prefs = preferences or {}
    include_nightlife = prefs.get("includeNightlife", False)
    include_food = prefs.get("includeFood", False)
    avoid_crowds = prefs.get("avoidCrowds", False)
    style = prefs.get("style", "").lower()
    group = prefs.get("group", "").lower()
    transport_mode = prefs.get("transport", "Public")
    budget = prefs.get("budget", "mid").lower()

    print(f"[itinerary.py] Building itinerary with preferences: "
          f"style={style}, group={group}, nightlife={include_nightlife}, "
          f"food={include_food}, transport={transport_mode}, budget={budget}")

    # ── Separate places by category ──────────────────────────────
    nightlife_pool = []
    food_pool = []
    general_pool = []

    for p in ranked_places:
        category = _classify_place(p)
        if category == 'nightlife' and include_nightlife:
            nightlife_pool.append(p)
        elif category == 'food' and include_food:
            food_pool.append(p)
        else:
            general_pool.append(p)

    print(f"[itinerary.py] Place pools — general: {len(general_pool)}, "
          f"food: {len(food_pool)}, nightlife: {len(nightlife_pool)}")

    # ── Time slot definitions ────────────────────────────────────
    TIME_SLOTS = [
        {"name": "Morning",   "start": 9,  "icon": "🌅",
         "preferred_pool": "general"},
        {"name": "Afternoon", "start": 13, "icon": "☀️",
         "preferred_pool": "food" if include_food else "general"},
        {"name": "Evening",   "start": 17, "icon": "🌆",
         "preferred_pool": "general"},
        {"name": "Night",     "start": 20, "icon": "🌙",
         "preferred_pool": "nightlife" if include_nightlife else "general"},
    ]

    def format_time(hour_float):
        h = int(hour_float) % 24
        m = int((hour_float - int(hour_float)) * 60)
        suffix = "AM" if h < 12 else "PM"
        display_h = h if h <= 12 else h - 12
        if display_h == 0: display_h = 12
        return f"{display_h}:{m:02d} {suffix}"

    def _fare_str(base, per_km):
        b, p = str(base).strip(), str(per_km).strip()
        if b in ("NA", "N/A", "Not Available", "nan") or \
           p in ("NA", "N/A", "Not Available", "nan"):
            return "Not Available"
        return f"{base} base | {per_km}/km"

    # Working copies of pools (we pop from these as places are used)
    remaining_general   = general_pool.copy()
    remaining_food      = food_pool.copy()
    remaining_nightlife = nightlife_pool.copy()
    # Fallback pool: all places regardless of category
    remaining_all       = ranked_places.copy()

    def get_next_from_pool(pool_name, candidates_subset=None):
        """
        Get the next unused place from the specified pool.
        Falls back to remaining_all if pool is empty.
        """
        pool_map = {
            "general":   remaining_general,
            "food":      remaining_food,
            "nightlife": remaining_nightlife
        }
        pool = pool_map.get(pool_name, remaining_general)

        target = None
        if pool:
            # If we have a geographic seed, pick closest from pool
            if candidates_subset:
                pool_set = {p['place_name'] for p in pool}
                geo_matches = [p for p in candidates_subset 
                               if p['place_name'] in pool_set]
                if geo_matches:
                    target = geo_matches[0]
            if not target:
                target = pool[0]

        # Fallback to any remaining place
        if not target and remaining_all:
            target = remaining_all[0]

        return target

    def remove_from_all_pools(place_name):
        nonlocal remaining_general, remaining_food, \
                 remaining_nightlife, remaining_all
        remaining_general   = [p for p in remaining_general
                                if p['place_name'] != place_name]
        remaining_food      = [p for p in remaining_food
                                if p['place_name'] != place_name]
        remaining_nightlife = [p for p in remaining_nightlife
                                if p['place_name'] != place_name]
        remaining_all       = [p for p in remaining_all
                                if p['place_name'] != place_name]

    itinerary = {}

    for d in range(1, days + 1):
        day_date = ""
        if start_date:
            try:
                dt = (datetime.strptime(str(start_date), "%Y-%m-%d")
                      + timedelta(days=d - 1))
                day_date = f" ({dt.strftime('%b %d, %Y')})"
            except:
                pass

        day_key = f"Day {d}{day_date}"
        day_events = []

        if d == 1:
            day_events.append(
                f"Arrival in {location}. Check into your hotel.")

        if not remaining_all:
            itinerary[day_key] = day_events
            continue

        # Geographic seed: pick the highest-scored remaining place
        seed = remaining_all[0]

        # Sort remaining_all by proximity to seed for geo-clustering
        geo_sorted = sorted(
            remaining_all,
            key=lambda p: calculate_haversine(
                seed.get('latitude', 0), seed.get('longitude', 0),
                p.get('latitude', 0), p.get('longitude', 0)
            )
        )

        places_added_today = 0

        for slot in TIME_SLOTS:
            if not remaining_all:
                break

            preferred_pool = slot['preferred_pool']
            found_p = get_next_from_pool(preferred_pool, geo_sorted)

            if not found_p:
                continue

            raw_duration = found_p.get('recommended_duration_min', 120)
            duration = max(1.0, float(raw_duration) / 60.0)
            start_h = slot['start']
            end_h = start_h + duration

            # Crowd warning logic
            crowd_lvl = found_p.get('crowd_level', 5)
            crowd_warning = None
            if avoid_crowds and crowd_lvl > 6:
                crowd_warning = "High crowd expected — consider visiting early morning."
            elif crowd_lvl > 7:
                crowd_warning = "Very popular spot — visit early to avoid queues."

            event = {
                "place_name": found_p.get('place_name', 'Unknown Place'),
                "time": f"{format_time(start_h)} - {format_time(end_h)}",
                "slot": slot['name'],
                "icon": slot['icon'],
                "type": found_p.get('type', 'Sightseeing'),
                "city": found_p.get('city', ''),
                "state": found_p.get('state', ''),
                "rating": found_p.get('rating', 4.0),
                "best_time": found_p.get('best_visit_time', 'Anytime'),
                "visit_time": round(duration, 1),
                "crowd_level": crowd_lvl,
                "crowd_warning": crowd_warning,
                "latitude": found_p.get('latitude', 0),
                "longitude": found_p.get('longitude', 0),
                "map_link": found_p.get('map_link', ''),
                "significance": found_p.get('significance', ''),
                "travel_tip": found_p.get('travel_tip', ''),
                "must_try_food": found_p.get('must_try_food', ''),
                "avg_local_transport_cost": found_p.get(
                    'avg_local_transport_cost', 300),
                "entrance_fee": found_p.get('entrance_fee', 0),
                "packing_suggestions": found_p.get('packing_suggestions', ''),
                "short_description": found_p.get('short_description', ''),
                "recommended_transport": _get_recommended_transport(
                    found_p, transport_mode),
                "transport_mode": transport_mode,
                "transport_fares": {
                    "auto": f"{found_p.get('auto_fare_min','N/A')} min | "
                            f"{found_p.get('auto_fare_per_km','N/A')}/km",
                    "rapido_bike": f"{found_p.get('rapido_bike_min','N/A')} min | "
                                   f"{found_p.get('rapido_bike_per_km','N/A')}/km",
                    "ola_car": _fare_str(
                        found_p.get('ola_car_base','N/A'),
                        found_p.get('ola_car_per_km','N/A')),
                    "uber_car": _fare_str(
                        found_p.get('uber_car_base','N/A'),
                        found_p.get('uber_car_per_km','N/A')),
                    "city_taxi": found_p.get('city_taxi_per_km','N/A'),
                    "surge_note": found_p.get('surge_pricing_note','N/A')
                },
                "nightlife_info": {
                    "venue_type": found_p.get('venue_type','N/A'),
                    "cover_charge": found_p.get('cover_charge','N/A'),
                    "avg_drinks_price": found_p.get('avg_drinks_price','N/A'),
                    "music_genre": found_p.get('music_genre','N/A'),
                    "club_timings": found_p.get('club_timings','N/A'),
                    "dress_code": found_p.get('dress_code','N/A'),
                    "ladies_night": found_p.get('ladies_night_info','N/A'),
                    "note": found_p.get('nightlife_note','N/A')
                } if found_p.get('cover_charge','N/A') != 'N/A' else None,
                # Score transparency
                "ml_scores": {
                    "xgb": round(found_p.get('xgb_score', 0), 3),
                    "cosine": round(found_p.get('cosine_score', 0), 3),
                    "final": round(found_p.get('final_score', 0), 3)
                }
            }

            day_events.append(event)
            places_added_today += 1
            remove_from_all_pools(found_p['place_name'])

            # Refresh geo_sorted after removal
            geo_sorted = [p for p in geo_sorted
                          if p['place_name'] != found_p['place_name']]

        # Daily cost estimate
        total_entrance = sum(
            e['entrance_fee'] for e in day_events
            if isinstance(e, dict) and 'entrance_fee' in e
        )
        transport_cost = (
            max(e['avg_local_transport_cost'] for e in day_events
                if isinstance(e, dict) and 'avg_local_transport_cost' in e)
            if places_added_today > 0 else 300
        )

        # Budget-aware food estimate
        food_budget = {'low': 400, 'mid': 800, 'luxury': 1500}
        food_cost = food_budget.get(budget, 800)

        total_cost = total_entrance + transport_cost + food_cost
        day_events.append({
            "info": f"Estimated Daily Cost: ₹{int(total_cost)} "
                    f"(Entry fees + {transport_mode} transport + "
                    f"₹{food_cost} food for {group or 'traveller'})",
            "type": "cost"
        })

        itinerary[day_key] = day_events

    # Transport info
    airport, railway = "Not Available", "Not Available"
    if ranked_places:
        airports = [p.get('airport') for p in ranked_places[:10]
                    if p.get('airport') and
                    "unknown" not in str(p.get('airport')).lower()]
        railways = [p.get('railway') for p in ranked_places[:10]
                    if p.get('railway') and
                    "unknown" not in str(p.get('railway')).lower()]
        if airports: airport = " / ".join(list(set(airports)))
        if railways: railway = " / ".join(list(set(railways)))

    return {
        "transport": {"airport": airport, "railway": railway},
        "plan": itinerary
    }