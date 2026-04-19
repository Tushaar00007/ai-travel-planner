import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset/tourism_dataset_enriched_v4.csv"
MODEL_PATH = BASE_DIR / "models/kmeans_clusters.pkl"

def train_kmeans():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    
    print(f"Loaded {len(df)} places")

    # ── Feature matrix for clustering ──────────────────────────
    # We cluster places by their CHARACTER, not quality.
    # Each place becomes a vector of what kind of experience it is.

    features = pd.DataFrame()

    # 1. Style tags — one-hot encode Trip_Preference_Tags
    style_keywords = [
        'Adventure', 'Relaxation', 'Cultural', 'Nightlife',
        'Nature', 'Beach', 'Heritage', 'Foodie', 'Shopping',
        'Religious', 'Family', 'Romantic', 'Solo', 'Backpacker'
    ]
    for kw in style_keywords:
        features[f'tag_{kw.lower()}'] = df['Trip_Preference_Tags'].apply(
            lambda x: 1 if kw.lower() in str(x).lower() else 0
        )

    # 2. Budget level — encode as 0/1/2
    budget_map = {'Low': 0, 'Budget': 0, 'Medium': 1, 
                  'Mid-Range': 1, 'High': 2, 'Luxury': 2}
    features['budget_level'] = df['Budget Level'].map(budget_map).fillna(1)

    # 3. Nightlife score (normalised)
    features['nightlife_norm'] = df['Nightlife Score (0-10)'].fillna(0) / 10

    # 4. Crowd level (normalised)
    features['crowd_norm'] = df['Crowd_Level'].fillna(5) / 10

    # 5. Tourism score (normalised)
    features['tourism_norm'] = df['Tourism Score (1-10)'].fillna(5) / 10

    # 6. Itinerary role — encode time of day
    role_map = {
        'Morning': 0, 'Afternoon': 1, 'Evening': 2,
        'Night': 3, 'Any Time': 2, 'N/A': 2
    }
    features['time_role'] = df['Itinerary_Role'].map(role_map).fillna(2)

    # 7. Place type — encode top types
    type_keywords = [
        'Beach', 'Temple', 'Museum', 'Park', 'Market',
        'Fort', 'Restaurant', 'Club', 'Hilltop', 'Waterfall'
    ]
    for kw in type_keywords:
        features[f'type_{kw.lower()}'] = df['Type'].apply(
            lambda x: 1 if kw.lower() in str(x).lower() else 0
        )

    # 8. Adventure available
    features['has_adventure'] = df['Adventure_Available'].apply(
        lambda x: 1 if str(x).lower() not in ['unknown', 'no', 'nan', ''] else 0
    )

    # 9. Is beach
    features['is_beach'] = df['Is_Beach'].apply(
        lambda x: 1 if str(x).lower() == 'yes' else 0
    )

    # 10. Is shopping
    features['is_shopping'] = df['Is_Shopping'].apply(
        lambda x: 1 if str(x).lower() == 'yes' else 0
    )

    print(f"Feature matrix shape: {features.shape}")

    # ── Scale features ──────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.fillna(0))

    # ── Train KMeans ────────────────────────────────────────────
    # 8 clusters: Adventure, Beach/Relaxation, Cultural/Heritage,
    # Nightlife/Party, Family/Nature, Food/Market, 
    # Religious/Spiritual, Mixed/General
    N_CLUSTERS = 8

    print(f"Training KMeans with {N_CLUSTERS} clusters...")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=20,        # more initialisations = more stable clusters
        max_iter=500
    )
    cluster_labels = kmeans.fit_predict(X_scaled)

    # ── Assign cluster IDs back to dataframe for inspection ─────
    df['cluster_id'] = cluster_labels

    # Print cluster composition for verification
    print("\n── Cluster Composition ──")
    for cid in range(N_CLUSTERS):
        cluster_df = df[df['cluster_id'] == cid]
        top_tags = cluster_df['Trip_Preference_Tags'].str.split(',').explode().str.strip().value_counts().head(3)
        print(f"Cluster {cid} ({len(cluster_df)} places): {list(top_tags.index)}")

    # ── Save bundle ─────────────────────────────────────────────
    bundle = {
        'kmeans': kmeans,
        'scaler': scaler,
        'feature_columns': list(features.columns),
        'style_keywords': style_keywords,
        'type_keywords': type_keywords,
        'n_clusters': N_CLUSTERS
    }

    import os
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nKMeans bundle saved to {MODEL_PATH}")
    print(f"Feature columns: {list(features.columns)}")

if __name__ == "__main__":
    train_kmeans()
