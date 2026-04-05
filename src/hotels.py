def get_hotels(location, budget_level="Mid-Range"):
    all_hotels = [
        {"name": f"{location} Luxury Grand Resort", "price": 9000, "rating": 4.7, "budget_category": "Luxury"},
        {"name": f"{location} Comfort Stay Inn", "price": 4500, "rating": 4.3, "budget_category": "Mid-Range"},
        {"name": f"{location} Budget Traveller Lodge", "price": 2000, "rating": 4.0, "budget_category": "Low"}
    ]
    if budget_level in ["High", "Luxury"]:
        return [all_hotels[0]]
    elif budget_level == "Low":
        return [all_hotels[2]]
    return all_hotels