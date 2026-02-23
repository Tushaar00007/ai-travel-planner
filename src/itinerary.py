from predict import get_ranked_places


def build_itinerary(city: str, days: int, hours_per_day: float = 8.0):
    """
    Build a day-wise itinerary from ranked places.

    city: city name (e.g., "Jaipur")
    days: number of days (e.g., 2)
    hours_per_day: max visiting hours per day
    """

    ranked_places = get_ranked_places(city, top_k=30)

    if not ranked_places:
        return {"error": "No places found for this city"}

    itinerary = {f"Day {i+1}": [] for i in range(days)}

    day_index = 0
    remaining_hours = hours_per_day

    for place in ranked_places:
        visit_time = place.get("visit_time", 1.5)

        # If place fits in current day
        if visit_time <= remaining_hours:
            itinerary[f"Day {day_index+1}"].append(place)
            remaining_hours -= visit_time

        else:
            # Move to next day
            day_index += 1

            if day_index >= days:
                break

            remaining_hours = hours_per_day
            itinerary[f"Day {day_index+1}"].append(place)
            remaining_hours -= visit_time

    return itinerary


if __name__ == "__main__":
    city = "Jaipur"
    days = 2

    plan = build_itinerary(city, days)

    print(f"\nItinerary for {city} ({days} days):\n")

    for day, places in plan.items():
        print(day)
        for p in places:
            # Updated to include the Google Maps link
            print(f"  - {p['place_name']} ({p['visit_time']} hrs)")
            print(f"    Map: {p.get('map_link', 'No link available')}")
        print()