import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLUSTER_MODEL_PATH = BASE_DIR / "models/kmeans_clusters.pkl"

_cluster_bundle = None

def load_cluster_bundle():
    global _cluster_bundle
    if _cluster_bundle is None and CLUSTER_MODEL_PATH.exists():
        _cluster_bundle = joblib.load(CLUSTER_MODEL_PATH)
        print(f"[cluster_engine] KMeans bundle loaded. "
              f"{_cluster_bundle['n_clusters']} clusters.")
    return _cluster_bundle

def _build_preference_vector(preferences, feature_columns, 
                              style_keywords, type_keywords):
    """
    Convert a preferences dict from the frontend into the same
    feature vector space that KMeans was trained on.
    """
    vec = {col: 0 for col in feature_columns}

    style = preferences.get("style", "").strip()
    budget = preferences.get("budget", "mid").strip().lower()
    group = preferences.get("group", "").strip()
    include_nightlife = preferences.get("includeNightlife", False)
    include_food = preferences.get("includeFood", False)
    avoid_crowds = preferences.get("avoidCrowds", False)

    # ── Style tags ──────────────────────────────────────────────
    style_tag_map = {
        "Adventure":   ["Adventure"],
        "Relaxation":  ["Relaxation", "Beach", "Nature"],
        "Cultural":    ["Cultural", "Heritage", "Religious"],
        "Nightlife":   ["Nightlife"]
    }
    tags_to_set = style_tag_map.get(style, [])
    for tag in tags_to_set:
        key = f"tag_{tag.lower()}"
        if key in vec:
            vec[key] = 1

    # ── Group type maps to tags ─────────────────────────────────
    group_tag_map = {
        "Solo":    ["solo", "backpacker"],
        "Couple":  ["romantic"],
        "Friends": ["nightlife"],
        "Family":  ["family"]
    }
    for tag in group_tag_map.get(group, []):
        key = f"tag_{tag.lower()}"
        if key in vec:
            vec[key] = 1

    # ── Nightlife toggle ────────────────────────────────────────
    if include_nightlife:
        vec['nightlife_norm'] = 0.8
        if 'tag_nightlife' in vec:
            vec['tag_nightlife'] = 1

    # ── Food toggle ─────────────────────────────────────────────
    if include_food:
        if 'tag_foodie' in vec:
            vec['tag_foodie'] = 1

    # ── Budget ──────────────────────────────────────────────────
    budget_val_map = {"low": 0, "mid": 1, "luxury": 2}
    vec['budget_level'] = budget_val_map.get(budget, 1)

    # ── Avoid crowds ────────────────────────────────────────────
    vec['crowd_norm'] = 0.2 if avoid_crowds else 0.5

    # ── Time role: default to afternoon/evening ─────────────────
    vec['time_role'] = 2

    return pd.DataFrame([vec])[feature_columns]

def get_cluster_filtered_df(df, preferences):
    """
    Stage 1 of the pipeline.
    
    Returns a filtered DataFrame containing only places from the
    best-matching cluster(s). Falls back to full df if model 
    not available or preferences are empty.
    
    Always returns at least 100 rows to give XGBoost enough 
    candidates to work with.
    """
    bundle = load_cluster_bundle()

    if bundle is None or not preferences:
        print("[cluster_engine] No bundle or preferences — skipping clustering.")
        return df

    kmeans = bundle['kmeans']
    scaler = bundle['scaler']
    feature_columns = bundle['feature_columns']
    style_keywords = bundle['style_keywords']
    type_keywords = bundle['type_keywords']
    n_clusters = bundle['n_clusters']

    # Build user preference vector
    pref_vec = _build_preference_vector(
        preferences, feature_columns, style_keywords, type_keywords
    )

    # Scale it using the same scaler used during training
    pref_vec_scaled = scaler.transform(pref_vec.fillna(0))

    # Find distances from preference vector to each cluster centre
    distances = kmeans.transform(pref_vec_scaled)[0]  # shape: (n_clusters,)

    # Pick top 2 closest clusters (not just 1) to avoid over-filtering
    top_clusters = np.argsort(distances)[:2]
    print(f"[cluster_engine] Best matching clusters: {list(top_clusters)} "
          f"(distances: {distances[top_clusters].round(2)})")

    # Filter dataframe to matching clusters
    # We need cluster_id column — compute it on the fly
    feature_cols_present = [c for c in feature_columns if c in df.columns]
    
    # Rebuild the same feature matrix used in training for this df
    place_features = _build_place_feature_matrix(df, feature_columns, 
                                                   style_keywords, type_keywords)
    place_features_scaled = scaler.transform(place_features.fillna(0))
    place_cluster_ids = kmeans.predict(place_features_scaled)
    
    df = df.copy()
    df['_cluster_id'] = place_cluster_ids
    
    cluster_mask = df['_cluster_id'].isin(top_clusters)
    filtered = df[cluster_mask].copy()

    # Safety net: if fewer than 80 places in matching clusters,
    # expand to top 3 clusters to give XGBoost enough candidates
    if len(filtered) < 80:
        top_clusters = np.argsort(distances)[:3]
        cluster_mask = df['_cluster_id'].isin(top_clusters)
        filtered = df[cluster_mask].copy()
        print(f"[cluster_engine] Expanded to 3 clusters — {len(filtered)} places")
    
    print(f"[cluster_engine] Returning {len(filtered)} places "
          f"from clusters {list(top_clusters)}")
    
    return filtered

def _build_place_feature_matrix(df, feature_columns, 
                                  style_keywords, type_keywords):
    """
    Rebuild the same feature matrix used in train_clusters.py
    but for a given dataframe subset (used for on-the-fly cluster assignment).
    """
    features = pd.DataFrame(index=df.index)

    for kw in style_keywords:
        features[f'tag_{kw.lower()}'] = df['Trip_Preference_Tags'].apply(
            lambda x: 1 if kw.lower() in str(x).lower() else 0
        )

    budget_map = {'Low': 0, 'Budget': 0, 'Medium': 1,
                  'Mid-Range': 1, 'High': 2, 'Luxury': 2}
    features['budget_level'] = df['Budget Level'].map(budget_map).fillna(1)
    features['nightlife_norm'] = df['Nightlife Score (0-10)'].fillna(0) / 10
    features['crowd_norm'] = df['Crowd_Level'].fillna(5) / 10
    features['tourism_norm'] = df['Tourism Score (1-10)'].fillna(5) / 10

    role_map = {
        'Morning': 0, 'Afternoon': 1, 'Evening': 2,
        'Night': 3, 'Any Time': 2, 'N/A': 2
    }
    features['time_role'] = df['Itinerary_Role'].map(role_map).fillna(2)

    for kw in type_keywords:
        features[f'type_{kw.lower()}'] = df['Type'].apply(
            lambda x: 1 if kw.lower() in str(x).lower() else 0
        )

    features['has_adventure'] = df['Adventure_Available'].apply(
        lambda x: 1 if str(x).lower() not in ['unknown', 'no', 'nan', ''] else 0
    )
    features['is_beach'] = df['Is_Beach'].apply(
        lambda x: 1 if str(x).lower() == 'yes' else 0
    )
    features['is_shopping'] = df['Is_Shopping'].apply(
        lambda x: 1 if str(x).lower() == 'yes' else 0
    )

    # Ensure all expected columns are present in correct order
    for col in feature_columns:
        if col not in features.columns:
            features[col] = 0

    return features[feature_columns]
