# TravelBNB — AI Travel Planner (ML Service)

Python ML service that generates personalized multi-day travel itineraries for TravelBNB. Handles destination recommendations, hotel filtering by budget, day-wise activity planning, and PDF export.

This is one of three services that make up TravelBNB:
- **Frontend** — [travel-bnb-frontend](https://github.com/Tushaar00007/travel-bnb-frontend) (React + Vite)
- **Backend** — [travelbnb-backend](https://github.com/Tushaar00007/travelbnb-backend) (FastAPI + MongoDB)
- **ML Service** (this repo) — FastAPI + Python ML

---

## Live Deployment

| Service | URL |
|---------|-----|
| ML Service (Render) | https://ai-travel-planner-txji.onrender.com |
| Backend API | https://travelbnb-backend.onrender.com |
| Frontend | https://travel-bnb-frontend.vercel.app |

---

## Project Structure

```
ai-travel-planner/
├── main.py                 # FastAPI entry (port 9000)
├── dataset/
│   └── tourism_dataset_enriched_v2.csv  # 1,400 rows, 523 cities
├── models/
│   ├── xgb_ranker.pkl      # Serialized ML model
│   └── kmeans_clusters.pkl # Clustering model
├── scripts/
│   ├── merge_datasets.py   # Combine source datasets
│   ├── enrich_dataset.py   # Add transport fares, preference tags
│   └── train_clusters.py   # Training script for clustering
├── src/                    # Core logic and ML components
│   ├── itinerary.py        # Slot-based day-wise itinerary generation
│   ├── hotels.py           # Hotel filtering by budget
│   ├── predict.py          # Inference logic
│   ├── train.py            # Model training script
│   └── pdf_generator.py    # PDF export via ReportLab
├── tests/
│   └── test_api.py
├── requirements.txt
└── verification_res.json   # Diagnostic output from validation runs
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI (Python 3.11) |
| ML Libraries | scikit-learn, pandas, numpy |
| PDF Generation | ReportLab |
| Geocoding | Google Maps Geocoding API |
| External Data | Unsplash API (place imagery) |
| Deployment | Render |

---

## Environment Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` file (see variables below), then:

```bash
uvicorn main:app --reload --port 9000
```

### Required Environment Variables

```env
GOOGLE_MAPS_API_KEY=...
UNSPLASH_ACCESS_KEY=...
ALLOWED_ORIGINS=http://localhost:8000,https://travelbnb-backend.onrender.com
```

---

## API Endpoints

### Itinerary Generation

| Method | Endpoint | Description | Body |
|--------|----------|-------------|------|
| `POST` | `/ml/generate` | Generate travel itinerary | `ItineraryRequest` |
| `POST` | `/generate_itinerary` | Legacy alias for `/ml/generate` | same |
| `POST` | `/ml/download_pdf` | Export itinerary as PDF | `{ itinerary }` |
| `POST` | `/download_pdf` | Legacy alias | same |

**ItineraryRequest:**
```json
{
  "destination": "Goa",
  "budget": 25000,
  "days": 4,
  "travelers": 2,
  "preferences": ["beach", "nightlife", "food"],
  "travelMode": "flight"
}
```

**Response:**
```json
{
  "itinerary": {
    "destination": "Goa",
    "totalCost": 23500,
    "days": [
      {
        "day": 1,
        "slots": [
          { "time": "morning", "activity": "Calangute Beach" },
          { "time": "afternoon", "activity": "Lunch at Britto's" },
          { "time": "evening", "activity": "Baga Beach Shacks" }
        ]
      }
    ],
    "hotels": []
  }
}
```

---

### Chat Assistant

| Method | Endpoint | Description | Body |
|--------|----------|-------------|------|
| `POST` | `/chat_travel_assistant` | Conversational travel Q&A | `{ message, context }` |

---

### Health Check

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service health status |

---

## How It Works

### 1. Preferences → Destination Matching
User preferences (beach, heritage, nightlife, etc.) are matched against trip preference tags in the tourism dataset. Each city has a preference vector; the model scores cities against the user's preference vector and ranks them.

### 2. Slot-Based Itinerary Generation
For each day of the trip, the itinerary is divided into three slots (morning, afternoon, evening). `itinerary.py` assigns activities to each slot based on:
- Activity type (outdoor in morning, food in afternoon/evening, nightlife at night)
- Proximity (group nearby activities within the same day)
- Opening hours and seasonal availability

### 3. Hotel Budget Filtering
`hotels.py` filters available hotels by:
- User's total budget minus estimated transport cost
- Per-night budget divided by number of nights
- Star rating preferences inferred from budget tier

### 4. PDF Export
`pdf_generator.py` uses ReportLab to generate a downloadable itinerary PDF with:
- Cover page with destination and dates
- Day-by-day breakdown with activities and timing
- Hotel details and booking links
- Budget summary

---

## Dataset

`dataset/tourism_dataset_enriched_v2.csv` contains:
- 1,400 rows across 91 columns
- 41 Indian states, 523 cities
- Transport fares (flight, train, bus)
- Trip preference tags (beach, nightlife, heritage, adventure, etc.)
- Nightlife ratings and food scene data
- Geocoordinates (1,169 rows pending geocoding)

### Dataset Pipeline

```
Source datasets (Kaggle, government tourism data)
  → scripts/merge_datasets.py    # Combine sources
  → scripts/enrich_dataset.py    # Add fares, tags, ratings
  → tourism_dataset_enriched_v2.csv
```

---

## Training

```bash
python src/train.py
```

Outputs a serialized model to `models/xgb_ranker.pkl`. `src/predict.py` loads this model for inference.

---

## Deployment (Render)

1. New Web Service → connect this repo
2. Build command: `pip install -r requirements.txt`
3. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables (Google Maps API key, Unsplash key)
5. Note the service URL — the backend's `ML_BASE_URL` env var must point here

Free tier note: Render free tier sleeps after 15 min of inactivity. First request after sleep takes 30–60 seconds. The backend should handle this with appropriate timeouts.

---

## Key Design Decisions

- **Called via backend proxy** — this service is not exposed to the frontend directly. The backend at `/api/ml/*` forwards requests here. Simpler CORS, centralized auth.
- **Legacy route aliases** — both `/ml/generate` and `/generate_itinerary` are supported for backward compatibility during the backend migration.
- **Slot-based itinerary model** — trips are divided into morning/afternoon/evening slots rather than continuous timelines. Easier to generate coherent plans and allows user edits per slot.
- **Single denormalized dataset** — while source datasets are kept separately in `scripts/`, the app consumes a single enriched CSV to avoid runtime joins.
- **PDFs generated on-demand** — not cached. Users regenerate when itineraries are updated.
