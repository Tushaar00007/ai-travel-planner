import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "dataset/tourism_dataset_enriched_v3.csv"
MODEL_PATH = BASE_DIR / "models/xgb_ranker.pkl"

def train_model():
    if not DATA_PATH.exists():
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    # Load the new dataset
    df = pd.read_csv(DATA_PATH)
    
    # Strip any whitespace from column names just in case
    df.columns = [c.strip() for c in df.columns]

    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    # 1. Feature Engineering (Requirement #2)
    # log_rating = log1p(Google review rating)
    df['log_rating'] = np.log1p(df['Google review rating'].fillna(0))
    
    # popularity_norm = Popularity Index / 100
    df['popularity_norm'] = df['Popularity Index (0-100)'].fillna(0) / 100
    
    # tourism_score_norm = Tourism Score / 10
    df['tourism_score_norm'] = df['Tourism Score (1-10)'].fillna(0) / 10
    
    # crowd_norm = Crowd_Level / 10
    df['crowd_norm'] = df['Crowd_Level'].fillna(0) / 10
    
    # market_score = number of famous markets (assuming semicolon separated)
    df['market_score'] = df['Famous Market'].apply(lambda x: len(str(x).split(';')) if pd.notnull(x) and str(x).strip() != "" else 0)
    
    # restaurant_score = number of famous restaurants (assuming semicolon separated)
    df['restaurant_score'] = df['Famous Restaurant'].apply(lambda x: len(str(x).split(';')) if pd.notnull(x) and str(x).strip() != "" else 0)

    # preference_count = number of comma-separated values in the Trip_Preference_Tags column
    df['preference_count'] = df['Trip_Preference_Tags'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 1)

    # 2. Update Target Score Logic (Requirement #3)
    # target_score = (Google rating * 0.25) + (popularity_norm * 0.20) + (tourism_score_norm * 0.20) + 
    #                (nightlife_score * 0.10) + (market_score * 0.10) + (crowd_norm * -0.05) + (log_rating * 0.10)
    
    df['rating'] = df['Google review rating'].fillna(0)
    df['nightlife_score'] = df['Nightlife Score (0-10)'].fillna(0)
    
    df['target_score'] = (
        (df['rating'] * 0.25) +
        (df['popularity_norm'] * 0.20) +
        (df['tourism_score_norm'] * 0.20) +
        (df['nightlife_score'] * 0.10) +
        (df['market_score'] * 0.10) +
        (df['crowd_norm'] * -0.05) +
        (df['log_rating'] * 0.10)
    )

    # 3. Encoding categorical features
    le_city = LabelEncoder()
    le_state = LabelEncoder()
    le_type = LabelEncoder()
    le_budget = LabelEncoder()

    df['city_encoded'] = le_city.fit_transform(df['City'].fillna('Unknown'))
    df['state_encoded'] = le_state.fit_transform(df['State'].fillna('Unknown'))
    df['type_encoded'] = le_type.fit_transform(df['Type'].fillna('Unknown'))
    df['budget_encoded'] = le_budget.fit_transform(df['Budget Level'].fillna('Budget'))

    # 4. Define ML Features (Requirement #4)
    features = [
        'city_encoded', 'state_encoded', 'type_encoded', 'budget_encoded',
        'rating', 'time needed to visit in hrs', 'nightlife_score',
        'popularity_norm', 'tourism_score_norm', 'crowd_norm',
        'market_score', 'restaurant_score', 'preference_count', 'Recommended_Duration_Min',
        'Avg_Local_Transport_Cost'
    ]
    
    # Fill remaining missing values in features
    df['time needed to visit in hrs'] = df['time needed to visit in hrs'].fillna(2.0)
    df['Recommended_Duration_Min'] = df['Recommended_Duration_Min'].fillna(1.0)
    # Convert Avg_Local_Transport_Cost to numeric
    def get_cost(val):
        try:
            if pd.isna(val): return 300
            import re
            nums = re.findall(r'\d+', str(val).replace(',', ''))
            return int(nums[0]) if nums else 300
        except:
            return 300

    df['Avg_Local_Transport_Cost'] = df['Avg_Local_Transport_Cost'].apply(get_cost)

    X = df[features]
    y = df['target_score']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost regression model
    model = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    print("Training model...")
    model.fit(X_train, y_train)

    # Save model and encoders
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    bundle = {
        'model': model,
        'le_city': le_city,
        'le_state': le_state,
        'le_type': le_type,
        'le_budget': le_budget,
        'features': features
    }
    joblib.dump(bundle, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Final training R^2 score: {model.score(X_train, y_train):.4f}")

if __name__ == "__main__":
    train_model()