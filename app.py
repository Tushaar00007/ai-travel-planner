import streamlit as st
import sys
import os
from datetime import date
import pandas as pd

# --------------------------------------------------
# PROJECT STRUCTURE FIX
# --------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from itinerary import build_pro_itinerary
from hotels import get_hotels
from predict import get_ranked_places

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Travel Researcher PRO",
    page_icon="🗺️",
    layout="wide"
)

st.title("🗺️ AI Travel Researcher PRO")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.header("Trip Details")

    location = st.text_input("Destination (City / State):", "Goa")
    days = st.number_input("Trip Duration (Days):", 1, 14, 3)
    start_date = st.date_input("Trip Start Date:", date.today())

    if st.button("Generate Smart Itinerary"):

        transport, plan = build_pro_itinerary(
            location,
            days,
            start_date
        )

        if transport is None:
            st.error(plan)
            st.stop()

        st.session_state.transport = transport
        st.session_state.plan = plan
        st.session_state.hotels = get_hotels(location)
        st.session_state.ranked_places = get_ranked_places(location)

# --------------------------------------------------
# MAIN CONTENT
# --------------------------------------------------
if "transport" in st.session_state:

    tab1, tab2, tab3, tab4 = st.tabs(
        ["✈️ Arrival", "🏨 Hotels", "📅 Itinerary", "🏬 Markets"]
    )

    # --------------------------------------------------
    # ARRIVAL TAB
    # --------------------------------------------------
    with tab1:
        st.subheader("✈️ Arrival Information")

        st.success(
            f"✈️ Nearest Airport: {st.session_state.transport.get('airport')}"
        )

        st.success(
            f"🚆 Major Railway Station: {st.session_state.transport.get('railway')}"
        )
    # --------------------------------------------------
    # HOTELS TAB
    # --------------------------------------------------
    with tab2:
        st.subheader("🏨 Recommended Hotels")

        for hotel in st.session_state.hotels:
            st.markdown(f"### {hotel['name']}")
            st.write(f"⭐ Rating: {hotel['rating']}")
            st.write(f"💰 Price: ₹{hotel['price']} per night")
            st.divider()

    # --------------------------------------------------
    # ITINERARY TAB (CLEANED)
    # --------------------------------------------------
    with tab3:

        for day_label, activities in st.session_state.plan.items():

            st.markdown(f"## 📅 {day_label}")

            for item in activities:

                if isinstance(item, str):
                    if "Estimated Cost" in item:
                        st.warning(item)
                    else:
                        st.success(item)
                    continue

                with st.expander(
                    f"📍 {item.get('place_name')} ({item.get('visit_time')} hrs)"
                ):

                    st.write(f"⭐ Rating: {item.get('rating')}")
                    st.write(f"💰 Budget Level: {item.get('budget')}")
                    st.write(f"🌃 Nightlife Score: {item.get('nightlife_score')}/10")

                    if item.get("adventure_available") == "Yes":
                        st.write(
                            f"🌊 Adventure Base Price: ₹{item.get('adventure_price')}"
                        )

                    # ✅ MAP BUTTON (FROM DATASET)
                    map_link = item.get("map_link")

                    if map_link and str(map_link).strip():
                        st.link_button(
                            "🗺️ View on Google Maps",
                            map_link,
                            use_container_width=True
                        )

            st.divider()

    # --------------------------------------------------
    # MARKETS TAB (ONLY MARKETS + RESTAURANTS)
    # --------------------------------------------------
    with tab4:

        st.subheader("🏬 Famous Markets & Restaurants")

        all_markets = set()
        all_restaurants = set()

        for place in st.session_state.ranked_places:

            if place.get("famous_market"):
                for m in str(place.get("famous_market")).split(","):
                    all_markets.add(m.strip())

            if place.get("famous_restaurant"):
                for r in str(place.get("famous_restaurant")).split(","):
                    all_restaurants.add(r.strip())

        # ---- MARKETS ----
        if all_markets:
            st.markdown("### 🏬 Top Markets")
            for market in sorted(all_markets):
                st.markdown(f"- {market}")
        else:
            st.info("No market data available.")

        st.divider()

        # ---- RESTAURANTS ----
        if all_restaurants:
            st.markdown("### 🍽️ Famous Restaurants")
            for restaurant in sorted(all_restaurants):
                st.markdown(f"- {restaurant}")
        else:
            st.info("No restaurant data available.")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption("Powered by AI Travel Researcher PRO")