import streamlit as st
import pandas as pd  # <--- Added this required import!
import sys
import os

# Add the src folder to Python's path so it can find your backend scripts
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from itinerary import build_itinerary

# --- WEB APP UI ---
st.set_page_config(page_title="AI Travel Planner", page_icon="🗺️")

st.title("🗺️ AI Travel Itinerary Planner")
st.markdown("Plan your next trip to India with machine learning-powered recommendations!")

# User Inputs
col1, col2, col3 = st.columns(3)
with col1:
    city = st.text_input("Enter City:", value="Jaipur", placeholder="e.g., Mumbai, Delhi, Jaipur")
with col2:
    days = st.number_input("Number of Days:", min_value=1, max_value=14, value=2)
with col3:
    hours = st.slider("Hours per day:", min_value=4.0, max_value=12.0, value=8.0, step=0.5)

# Generate Button
if st.button("Generate My Itinerary", type="primary"):
    if not city:
        st.warning("Please enter a city name.")
    else:
        with st.spinner(f"Using AI to rank the best places in {city}..."):
            # Call your backend function
            plan = build_itinerary(city, days, hours)

            if "error" in plan:
                st.error(plan["error"])
            else:
                st.success("Itinerary Generated Successfully!")

                # Display the plan beautifully
                for day, places in plan.items():
                    st.header(day)

                    if not places:
                        st.info("Free time! Try increasing hours or choosing a city with more spots.")
                        continue

                    for p in places:
                        # Use expanders for a clean UI
                        with st.expander(f"📍 {p['place_name']} ({p['visit_time']} hrs)"):
                            st.write(f"⭐ **Rating:** {p['rating']} / 5.0")
                            st.write(f"🤖 **AI Match Score:** {round(p.get('ml_score', 0), 3)}")

                            # Clickable Map Link
                            map_link = p.get('map_link', '#')
                            if map_link != "No link available" and pd.notna(map_link):
                                st.markdown(f"[🗺️ View on Google Maps]({map_link})")
                            else:
                                st.write("🗺️ Map link unavailable")