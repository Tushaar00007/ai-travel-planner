import sys
import os
from datetime import date
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from fastapi.middleware.cors import CORSMiddleware

# Ensure the src directory is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from itinerary import build_pro_itinerary
from hotels import get_hotels
from predict import get_ranked_places
from pdf_generator import generate_itinerary_pdf

app = FastAPI(
    title="AI Travel Researcher API",
    description="API for fetching ML generated itineraries, hotels, and ranked places for a given destination.",
    version="1.1.0"
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ItineraryRequest(BaseModel):
    location: str
    days: int
    start_date: date
    preferences: Optional[dict] = None

@app.post("/ml/generate")
@app.post("/generate_itinerary")  # Keep legacy route for compatibility
def generate_itinerary(req: ItineraryRequest):
    """
    Generate an itinerary, get hotel recommendations and ranked places based on the location.
    Utilizes the 66-column dataset and upgraded ML ranking.
    """
    try:
        # 1. Get Ranked Places from ML Model
        # Returns top 25 places with detailed fields
        ranked_places = get_ranked_places(req.location, req.start_date, preferences=req.preferences)
        
        if not ranked_places:
            return {"success": False, "message": "No places found for this location."}

        # 2. Build Itinerary using the advanced engine
        # Logic: clustering, duration-based, time-slot aware, crowd-aware
        itinerary_data = build_pro_itinerary(
            req.location, 
            req.days, 
            ranked_places, 
            req.start_date
        )

        # 3. Get Hotel Recommendations
        budget_level = req.preferences.get("budget", "Mid-Range") if req.preferences else "Mid-Range"
        hotels = get_hotels(req.location, budget_level)

        # 4. Generate AI Travel Summary
        # Collect top types for a natural summary
        place_types = list(set([p['type'] for p in ranked_places[:10]]))
        summary = f"This {req.days}-day trip in {req.location} focuses on {', '.join(place_types[:3]).lower()} while covering the city's most popular attractions."
        
        # Add crowd awareness to summary
        if any(p.get('crowd_level', 0) > 7 for p in ranked_places):
            summary += " Note: High crowd levels are expected at some spots; we've optimized your schedule for early morning visits where possible."

        return {
            "success": True,
            "data": {
                "transport": itinerary_data["transport"],
                "plan": itinerary_data["plan"],
                "hotels": hotels,
                "ranked_places": ranked_places,
                "travel_summary": summary
            }
        }

    except Exception as e:
        print(f"Error in generate_itinerary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/download_pdf")
@app.post("/download_pdf")  # Keep legacy route for compatibility
async def download_pdf(req: ItineraryRequest):
    """
    Generate and download a professional itinerary PDF.
    """
    try:
        # 1. Get Data
        ranked_places = get_ranked_places(req.location, req.start_date, preferences=req.preferences)
        if not ranked_places:
            raise HTTPException(status_code=404, detail="No places found for this location.")

        itinerary_data = build_pro_itinerary(
            req.location,
            req.days,
            ranked_places,
            req.start_date
        )
        budget_level = req.preferences.get("budget", "Mid-Range") if req.preferences else "Mid-Range"
        hotels = get_hotels(req.location, budget_level)

        # 2. Generate PDF
        pdf_buffer = generate_itinerary_pdf(
            req.location,
            req.days,
            req.start_date,
            itinerary_data,
            hotels
        )

        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=itinerary_{req.location.replace(' ', '_').lower()}.pdf"
            }
        )

    except Exception as e:
        print(f"Error in download_pdf: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {
        "status": "active", 
        "message": "AI Travel Planner Backend v1.1 (66-column dataset) is running"
    }

class ChatRequest(BaseModel):
    message: str

@app.post("/chat_travel_assistant")
def chat_travel_assistant(req: ChatRequest):
    """
    AI Travel Assistant Chatbot endpoint.
    Simulates a travel knowledge expert response.
    """
    msg = req.message.lower()
    
    # Simple rule-based simulation for now
    if "night" in msg or "nightlife" in msg:
        reply = "Goa has a vibrant nightlife! You should check out Baga Beach, Tito's Lane, or Thalassa in Vagator for a great evening."
    elif "food" in msg or "eat" in msg:
        reply = "You must try the local seafood! Prawn Ghashi, Bibinca for dessert, and Cashew Feni are specialties you shouldn't miss."
    elif "hidden" in msg or "secret" in msg:
        reply = "For a quieter experience, head to Butterfly Beach or Cabo de Rama fort for stunning sunset views without the crowds."
    elif "transport" in msg or "travel" in msg:
        reply = "Renting a scooter is the best way to get around. Alternatively, you can use local taxis or the 'Goa Miles' app."
    else:
        reply = "That sounds like a great plan! Is there anything specific you'd like to know about the local attractions, food, or transport?"
        
    return {"reply": reply}

if __name__ == "__main__":
    # Start the server on port 9000
    uvicorn.run(app, host="0.0.0.0", port=9000)
