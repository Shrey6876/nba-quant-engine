import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import time
from nba_api.stats.endpoints import leaguegamelog
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

def pull_raw_matchups():
    """
    Pulls raw data directly from NBA API for the 3 seasons to get MATCHUP strings.
    """
    seasons = ["2023-24", "2024-25", "2025-26"]
    dfs = []
    
    for season in seasons:
        print(f"Fetching raw matchup data from nba_api for {season}...")
        try:
            logs = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation='P')
            df = logs.get_data_frames()[0]
            dfs.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching {season}: {e}")
            
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

def calculate_opponent_defense(raw_df):
    """
    Parses MATCHUP to get the Opponent, then calculates Opponent Defensive Rating (Points allowed).
    """
    print("Parsing 79,000+ Matchup Strings via Regex...")
    
    # MATCHUP looks like "BOS @ NYK" or "BOS vs. NYK"
    # The last 3 characters are always the opponent abbreviation.
    raw_df['Opponent'] = raw_df['MATCHUP'].str[-3:]
    raw_df['GAME_DATE'] = pd.to_datetime(raw_df['GAME_DATE'])
    
    # Sort chronologically
    raw_df = raw_df.sort_values(by=['GAME_DATE'])
    
    # We want to know how many points the Opponent gave up.
    # To do this accurately, we group by Game_ID and Opponent to see total points scored against them.
    # Or simply: how many points did players score against this Opponent?
    
    # We will compute a rolling EMA of points scored by ANY player against this Opponent
    print("Calculating Rolling Defensive Rating for all 30 teams...")
    # Group by Opponent and Game Date, sum the points scored against them in that game
    team_points_allowed = raw_df.groupby(['Opponent', 'GAME_ID', 'GAME_DATE'])['PTS'].sum().reset_index()
    team_points_allowed = team_points_allowed.sort_values(by=['Opponent', 'GAME_DATE'])
    
    # Calculate a 10-game EMA of points allowed
    team_points_allowed['opp_def_rating_10'] = team_points_allowed.groupby('Opponent')['PTS'].transform(
        lambda x: x.ewm(span=10, adjust=False).mean().shift(1)
    )
    
    # Merge this defensive rating back into the raw_df
    # Note: raw_df represents the offensive player's stats. We join on GAME_ID and Opponent.
    merged = pd.merge(raw_df, team_points_allowed[['GAME_ID', 'Opponent', 'opp_def_rating_10']], 
                      on=['GAME_ID', 'Opponent'], how='left')
    
    # Clean up column names to match our database (player_id, game_id)
    merged = merged.rename(columns={'PLAYER_ID': 'player_id', 'GAME_ID': 'game_id'})
    merged['player_id'] = merged['player_id'].astype(int)
    merged['game_id'] = merged['game_id'].astype(str)
    
    return merged[['player_id', 'game_id', 'Opponent', 'opp_def_rating_10']]

def update_feature_store():
    print("Loading existing feature store...")
    feature_df = pd.read_sql("SELECT * FROM feature_store", engine)
    feature_df['game_date'] = pd.to_datetime(feature_df['game_date'])
    
    raw_df = pull_raw_matchups()
    opp_def_df = calculate_opponent_defense(raw_df)
    
    print("Merging Opponent Defense into the Heavy Matrix...")
    # Merge on player_id and game_id
    # feature_store has player_id and game_id
    final_df = pd.merge(feature_df, opp_def_df, on=['player_id', 'game_id'], how='left')
    
    # Drop rows where we couldn't calculate def rating (first few games of season)
    final_df = final_df.dropna(subset=['opp_def_rating_10'])
    
    print("Overwriting feature_store with new Opponent context...")
    final_df.to_sql('feature_store', engine, if_exists='replace', index=False)
    print("Done! The Matrix is now complete.")

if __name__ == "__main__":
    update_feature_store()
