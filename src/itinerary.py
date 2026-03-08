from ML_model.src.predict import get_ranked_places, load_data
from datetime import timedelta


def get_season(date_obj):

    if date_obj.month in [11,12,1,2]:
        return "peak"

    return "off"


def seasonal_multiplier(season):

    return 1.4 if season == "peak" else 1.0


def build_pro_itinerary(location, days, start_date):

    ranked_places = get_ranked_places(location)

    if not ranked_places:
        return None, "Location not found"

    df = load_data()

    location_clean = location.lower().strip()

    transport_rows = df[
        (df["city"].str.lower() == location_clean) |
        (df["state"].str.lower() == location_clean)
    ]

    if transport_rows.empty:

        transport_rows = df[
            df["city"].str.lower().str.contains(location_clean) |
            df["state"].str.lower().str.contains(location_clean)
        ]

    airport = (
        transport_rows["airport"].dropna().iloc[0]
        if not transport_rows["airport"].dropna().empty
        else "Not Available"
    )

    railway = (
        transport_rows["railway"].dropna().iloc[0]
        if not transport_rows["railway"].dropna().empty
        else "Not Available"
    )

    itinerary = {}
    remaining_places = ranked_places.copy()

    for d in range(days):

        current_date = start_date + timedelta(days=d)

        season = get_season(current_date)
        multiplier = seasonal_multiplier(season)

        day_label = f"Day {d+1} / {current_date.strftime('%d-%m-%Y')}"

        day_plan = []

        if d == 0:
            day_plan.append(f"Arrival at {airport}")

        remaining_hours = 8
        selected = []

        for place in list(remaining_places):

            visit_time = float(place.get("visit_time", 2))

            if visit_time <= remaining_hours:

                selected.append(place)
                remaining_hours -= visit_time
                remaining_places.remove(place)

        day_cost = 0

        for p in selected:

            fee = p.get("fee")

            if str(fee).isdigit():
                day_cost += int(fee)

            if p.get("adventure_available") == "Yes":

                base_price = float(p.get("adventure_price", 0))
                day_cost += int(base_price * multiplier)

        day_cost += 1500

        day_plan.extend(selected)

        day_plan.append(f"💰 Estimated Cost: ₹{int(day_cost)}")

        itinerary[day_label] = day_plan

    transport = {
        "airport": airport,
        "railway": railway
    }

    return transport, itinerary