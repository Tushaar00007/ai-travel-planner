import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math

def calculate_haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
        return 0
    R = 6371  # Earth radius in km
    try:
        dlat = math.radians(float(lat2) - float(lat1))
        dlon = math.radians(float(lon2) - float(lon1))
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(float(lat1))) * math.cos(math.radians(float(lat2))) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))
        return R * c
    except:
        return 0

def build_pro_itinerary(location, days, ranked_places, start_date=None):
    if not ranked_places:
        return {"transport": {"airport": "N/A", "railway": "N/A"}, "plan": {}}

    itinerary = {}
    remaining_places = ranked_places.copy()
    
    # Define Time Slots
    TIME_SLOTS = [
        {"name": "Morning", "start": 9, "icon": "🌅"},
        {"name": "Afternoon", "start": 13, "icon": "☀️"},
        {"name": "Evening", "start": 17, "icon": "🌆"},
        {"name": "Night", "start": 20, "icon": "🌙"}
    ]

    def format_time(hour_float):
        h = int(hour_float) % 24
        m = int((hour_float - int(hour_float)) * 60)
        suffix = "AM" if h < 12 else "PM"
        display_h = h if h <= 12 else h - 12
        if display_h == 0: display_h = 12
        return f"{display_h}:{m:02d} {suffix}"

    for d in range(1, days + 1):
        day_date = ""
        if start_date:
            try:
                dt = datetime.strptime(str(start_date), "%Y-%m-%d") + timedelta(days=d-1)
                day_date = f" ({dt.strftime('%b %d, %Y')})"
            except:
                pass
        
        day_key = f"Day {d}{day_date}"
        day_events = []
        
        if d == 1:
            day_events.append(f"Arrival in {location}. Check into your hotel.")

        # Find a seed for this day
        if not remaining_places:
            itinerary[day_key] = day_events
            continue
            
        seed_place = remaining_places[0]
        candidates = sorted(
            remaining_places,
            key=lambda p: calculate_haversine(seed_place.get('latitude', 0), seed_place.get('longitude', 0), p.get('latitude', 0), p.get('longitude', 0))
        )

        places_added_today = 0
        for slot in TIME_SLOTS:
            if not candidates:
                break
                
            found_p = None
            # Matching logic
            for p in candidates[:10]:
                if str(p.get('time_slot', '')).strip() == slot['name']:
                    found_p = p
                    break
            
            if not found_p:
                found_p = candidates[0]
                
            if found_p:
                # Fix: recommended_duration_min is in MINUTES, but we need HOURS
                raw_duration = found_p.get('recommended_duration_min', 120)
                duration = max(1.0, float(raw_duration) / 60.0)
                
                start_h = slot['start']
                end_h = start_h + duration
                
                event = {
                    "place_name": found_p.get('place_name', 'Unknown Place'),
                    "time": f"{format_time(start_h)} - {format_time(end_h)}",
                    "type": found_p.get('type', 'Sightseeing'),
                    "city": found_p.get('city', ''),
                    "state": found_p.get('state', ''),
                    "rating": found_p.get('rating', 4.0),
                    "best_time": found_p.get('best_visit_time', 'Anytime'),
                    "visit_time": round(duration, 1),
                    "crowd_level": found_p.get('crowd_level', 5),
                    "crowd_warning": found_p.get('crowd_warning') if found_p.get('crowd_warning') else ("High crowd expected. Visit early morning." if found_p.get('crowd_level', 0) > 7 else None),
                    "latitude": found_p.get('latitude', 0),
                    "longitude": found_p.get('longitude', 0),
                    "map_link": found_p.get('map_link', ''),
                    "significance": found_p.get('significance', ''),
                    "travel_tip": found_p.get('travel_tip', ''),
                    "must_try_food": found_p.get('must_try_food', ''),
                    "avg_local_transport_cost": found_p.get('avg_local_transport_cost', 300),
                    "entrance_fee": found_p.get('entrance_fee', 0),
                    "packing_suggestions": found_p.get('packing_suggestions', ''),
                    "short_description": found_p.get('short_description', '')
                }
                day_events.append(event)
                places_added_today += 1
                
                # Use a more reliable way to remove by comparing place_name
                name_to_remove = found_p.get('place_name')
                remaining_places = [item for item in remaining_places if item.get('place_name') != name_to_remove]
                candidates = [item for item in candidates if item.get('place_name') != name_to_remove]

        # Cost estimation
        total_entrance = sum([e['entrance_fee'] for e in day_events if isinstance(e, dict) and 'entrance_fee' in e])
        transport_cost = max([e['avg_local_transport_cost'] for e in day_events if isinstance(e, dict) and 'avg_local_transport_cost' in e]) if places_added_today > 0 else 300
        total_cost = total_entrance + transport_cost + 800
        
        day_events.append({
            "info": f"Estimated Daily Cost: ₹{int(total_cost)} (Includes entry fees, local transport, and ₹800 for food)",
            "type": "cost"
        })
        
        itinerary[day_key] = day_events

    # Transport info from top 10 ranked places
    airport = "Not Available"
    railway = "Not Available"
    if ranked_places:
        airports = [p.get('airport') for p in ranked_places[:10] if p.get('airport') and "unknown" not in str(p.get('airport')).lower()]
        railways = [p.get('railway') for p in ranked_places[:10] if p.get('railway') and "unknown" not in str(p.get('railway')).lower()]
        if airports: airport = " / ".join(list(set(airports)))
        if railways: railway = " / ".join(list(set(railways)))

    return {
        "transport": {"airport": airport, "railway": railway},
        "plan": itinerary
    }