import requests
import json
from datetime import date

def test_generate_itinerary():
    url = "http://localhost:9000/generate_itinerary"
    payload = {
        "location": "Goa, India",
        "days": 2,
        "start_date": str(date.today())
    }
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            success = data.get("success")
            print(f"Success: {success}")
            
            res_data = data.get("data", {})
            print(f"Has transport: {'transport' in res_data}")
            print(f"Has plan: {'plan' in res_data}")
            print(f"Has hotels: {'hotels' in res_data}")
            print(f"Has ranked_places: {'ranked_places' in res_data}")
            print(f"Has travel_summary: {'travel_summary' in res_data}")
            
            if 'travel_summary' in res_data:
                print(f"Summary: {res_data['travel_summary']}")
            
            # Check for detailed fields in itinerary
            plan = res_data.get("plan", {})
            if plan:
                first_day_key = list(plan.keys())[0]
                events = plan[first_day_key]
                print(f"\nChecking {first_day_key} events:")
                for event in events:
                    if isinstance(event, dict) and 'place_name' in event:
                        print(f"  Place: {event['place_name']}")
                        print(f"    Lat/Lon: {event.get('latitude')}, {event.get('longitude')}")
                        print(f"    Crowd Level: {event.get('crowd_level')}")
                        if event.get('crowd_warning'):
                            print(f"    Crowd Warning: {event.get('crowd_warning')}")
                        print(f"    Packing: {event.get('packing')}")
                        print(f"    AI Knowledge Support: {event.get('info') is not None}")
                        break
                    elif isinstance(event, dict) and event.get('type') == 'cost':
                        print(f"  Cost info: {event.get('info')}")
            
            # Check ranked_places fields
            ranked_places = res_data.get("ranked_places", [])
            if ranked_places:
                p = ranked_places[0]
                print(f"\nRanked Place Example: {p['place_name']}")
                print(f"  AI Description: {p.get('place_description_ai')[:50]}...")
                print(f"  Common Qs: {p.get('common_questions')[:50]}...")
                print(f"  Local Transport: {p.get('local_transport_options')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    test_generate_itinerary()
