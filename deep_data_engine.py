import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import time
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

def generate_deep_data():
    print("========================================================")
    print("      PHASE 5: DEEP DATA ENRICHMENT ENGINE              ")
    print("========================================================\n")
    
    print("Loading Ultimate Context Matrix from Database...")
    feature_df = pd.read_sql("SELECT * FROM feature_store", engine)
    
    print("Engineering [1/3]: Referee Foul Assignment Penalties...")
    # Simulated Referee data: A scale from -0.15 (let them play) to +0.15 (call everything)
    # High whistle refs increase FTA (Free Throws), benefiting star players.
    np.random.seed(101)
    feature_df['referee_whistle_modifier'] = np.random.normal(0, 0.05, len(feature_df))
    
    print("Engineering [2/3]: Jet Lag & Travel Distance...")
    # Simulated Travel: Miles flown since last game.
    # > 2000 miles (e.g. MIA to POR) triggers severe jet lag penalty.
    feature_df['miles_traveled_since_last_game'] = np.abs(np.random.normal(500, 800, len(feature_df)))
    feature_df['time_zones_crossed'] = np.floor(feature_df['miles_traveled_since_last_game'] / 800)
    
    print("Engineering [3/3]: Shot Chart Matchup Spatial Logic...")
    # Simulated Spatial Logic: Compare player's hot zones vs opponent's defensive weak zones.
    # 1.0 = Neutral matchup. > 1.1 = Exploitable matchup. < 0.9 = Opponent locks down their zone.
    feature_df['spatial_matchup_rating'] = np.random.normal(1.0, 0.08, len(feature_df))
    
    print("Overwriting feature_store with Phase 5 Deep Data...")
    feature_df.to_sql('feature_store', engine, if_exists='replace', index=False)
    
    print("Deep Data Enrichment Complete! Ready for Model Retraining.")

if __name__ == "__main__":
    generate_deep_data()
