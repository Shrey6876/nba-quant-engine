import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

def load_raw_data():
    """Loads raw game logs and games from SQLite."""
    print("Loading raw data from database...")
    query = """
    SELECT 
        p.id as log_id, p.player_id, p.game_id, p.minutes, p.points, p.rebounds, p.assists, p.threes_made,
        g.game_date
    FROM player_game_logs p
    JOIN games g ON p.game_id = g.id
    ORDER BY p.player_id, g.game_date ASC
    """
    df = pd.read_sql(query, engine)
    df['game_date'] = pd.to_datetime(df['game_date'])
    return df

def calculate_complex_features(df):
    """
    Builds the 'Heavy Matrix' of features.
    """
    print("Building The Heavy Matrix (Complex Feature Engineering)...")
    
    df = df.sort_values(by=['player_id', 'game_date'])
    
    # 1. Time-Series Dynamics (EMA)
    df['pts_ema_3'] = df.groupby('player_id')['points'].transform(lambda x: x.ewm(span=3, adjust=False).mean().shift(1))
    df['pts_ema_7'] = df.groupby('player_id')['points'].transform(lambda x: x.ewm(span=7, adjust=False).mean().shift(1))
    df['pts_ema_15'] = df.groupby('player_id')['points'].transform(lambda x: x.ewm(span=15, adjust=False).mean().shift(1))
    
    df['reb_ema_7'] = df.groupby('player_id')['rebounds'].transform(lambda x: x.ewm(span=7, adjust=False).mean().shift(1))
    df['ast_ema_7'] = df.groupby('player_id')['assists'].transform(lambda x: x.ewm(span=7, adjust=False).mean().shift(1))
    df['min_ema_7'] = df.groupby('player_id')['minutes'].transform(lambda x: x.ewm(span=7, adjust=False).mean().shift(1))
    
    # 2. Volatility/Variance Tracking
    df['pts_volatility_10'] = df.groupby('player_id')['points'].transform(lambda x: x.rolling(window=10).std().shift(1))
    
    # 3. Contextual Level (Rest Advantage)
    df['days_rest'] = df.groupby('player_id')['game_date'].diff().dt.days
    df['days_rest'] = df['days_rest'].clip(upper=7).fillna(7)
    
    # 4. Advanced: Career Baseline (True Talent Level)
    # The EMA gives recent form, but a 100-game moving average gives their true baseline talent
    df['pts_career_baseline'] = df.groupby('player_id')['points'].transform(lambda x: x.rolling(window=82, min_periods=10).mean().shift(1))
    
    # 5. Advanced: Minutes projection
    # If their EMA minutes is < 15, we severely discount their confidence.
    df['expected_minutes_ratio'] = df['min_ema_7'] / 36.0 # normalized against starter minutes
    
    df_clean = df.dropna().copy()
    
    print(f"Feature Engineering complete. Matrix contains {len(df_clean)} rows and {len(df_clean.columns)} columns.")
    return df_clean

def save_features(df):
    print("Saving feature matrix to database...")
    df.to_sql('feature_store', engine, if_exists='replace', index=False)
    print("Feature store updated successfully.")

if __name__ == "__main__":
    raw_df = load_raw_data()
    feature_df = calculate_complex_features(raw_df)
    save_features(feature_df)
