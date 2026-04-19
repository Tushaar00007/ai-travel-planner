import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Preference tag vocabulary — same keywords used in cluster_engine
STYLE_KEYWORDS = [
    'Adventure', 'Relaxation', 'Cultural', 'Nightlife',
    'Nature', 'Beach', 'Heritage', 'Foodie', 'Shopping',
    'Religious', 'Family', 'Romantic', 'Solo', 'Backpacker'
]

BUDGET_VALUES = {'low': 0.0, 'mid': 0.5, 'luxury': 1.0}

GROUP_TAG_MAP = {
    'solo':    ['Solo', 'Backpacker', 'Adventure'],
    'couple':  ['Romantic', 'Beach', 'Nature'],
    'friends': ['Nightlife', 'Adventure', 'Foodie'],
    'family':  ['Family', 'Nature', 'Cultural']
}

def _build_user_vector(preferences):
    """
    Build a numeric vector representing the user's preferences.
    Shape: (1, n_features)
    """
    vec = []

    style = preferences.get("style", "").strip()
    budget = preferences.get("budget", "mid").strip().lower()
    group = preferences.get("group", "").strip().lower()
    include_nightlife = preferences.get("includeNightlife", False)
    include_food = preferences.get("includeFood", False)
    avoid_crowds = preferences.get("avoidCrowds", False)

    # ── Style tag one-hot (14 dims) ─────────────────────────────
    style_tag_map = {
        "Adventure":  ["Adventure"],
        "Relaxation": ["Relaxation", "Beach", "Nature"],
        "Cultural":   ["Cultural", "Heritage", "Religious"],
        "Nightlife":  ["Nightlife", "Romantic"]
    }
    active_tags = set(style_tag_map.get(style, []))
    
    # Add group tags
    for tag in GROUP_TAG_MAP.get(group, []):
        active_tags.add(tag)

    # Add toggle tags
    if include_nightlife:
        active_tags.add("Nightlife")
    if include_food:
        active_tags.add("Foodie")

    for kw in STYLE_KEYWORDS:
        vec.append(1.0 if kw in active_tags else 0.0)

    # ── Budget (1 dim, normalised 0-1) ──────────────────────────
    vec.append(BUDGET_VALUES.get(budget, 0.5))

    # ── Nightlife intensity (1 dim) ─────────────────────────────
    vec.append(1.0 if include_nightlife else 0.0)

    # ── Food intensity (1 dim) ──────────────────────────────────
    vec.append(1.0 if include_food else 0.0)

    # ── Crowd preference (1 dim: 0=ok with crowds, 1=avoid) ────
    vec.append(1.0 if avoid_crowds else 0.0)

    return np.array(vec).reshape(1, -1)

def _build_place_vectors(places_df):
    """
    Build a numeric vector for each place in the dataframe.
    Must match the dimensions of _build_user_vector exactly.
    Returns numpy array of shape (n_places, n_features)
    """
    vectors = []

    for _, row in places_df.iterrows():
        vec = []
        tags_str = str(row.get('Trip_Preference_Tags', '')).lower()

        # ── Style tags (14 dims) ────────────────────────────────
        for kw in STYLE_KEYWORDS:
            vec.append(1.0 if kw.lower() in tags_str else 0.0)

        # ── Budget (1 dim) ──────────────────────────────────────
        budget_raw = str(row.get('Budget Level', 'Mid-Range')).strip()
        budget_val = {
            'Low': 0.0, 'Budget': 0.0,
            'Medium': 0.5, 'Mid-Range': 0.5,
            'High': 1.0, 'Luxury': 1.0
        }.get(budget_raw, 0.5)
        vec.append(budget_val)

        # ── Nightlife intensity (1 dim) ─────────────────────────
        nightlife_score = float(row.get('Nightlife Score (0-10)', 0) or 0)
        vec.append(nightlife_score / 10.0)

        # ── Food score (1 dim) ──────────────────────────────────
        has_food = 1.0 if (
            pd.notnull(row.get('Famous Restaurant')) and
            str(row.get('Famous Restaurant')).strip() not in ['', 'Unknown']
        ) else 0.0
        vec.append(has_food)

        # ── Crowd level (1 dim: high crowd = high value) ────────
        crowd = float(row.get('Crowd_Level', 5) or 5)
        vec.append(crowd / 10.0)

        vectors.append(vec)

    return np.array(vectors)

def compute_cosine_scores(places_df, preferences):
    """
    Main function called by predict.py.
    
    Takes the top-50 XGBoost results and re-ranks them using
    cosine similarity between the user preference vector and 
    each place's feature vector.
    
    Returns the same dataframe with a new column 'cosine_score' 
    added, values in range [0, 1].
    """
    if not preferences or places_df.empty:
        places_df = places_df.copy()
        places_df['cosine_score'] = 0.5
        return places_df

    try:
        user_vec = _build_user_vector(preferences)
        place_vecs = _build_place_vectors(places_df)

        # cosine_similarity returns shape (1, n_places)
        scores = cosine_similarity(user_vec, place_vecs)[0]

        places_df = places_df.copy()
        places_df['cosine_score'] = scores

        print(f"[preference_matcher] Cosine scores — "
              f"min: {scores.min():.3f}, "
              f"max: {scores.max():.3f}, "
              f"mean: {scores.mean():.3f}")

        return places_df

    except Exception as e:
        print(f"[preference_matcher] Error computing cosine scores: {e}")
        places_df = places_df.copy()
        places_df['cosine_score'] = 0.5
        return places_df
