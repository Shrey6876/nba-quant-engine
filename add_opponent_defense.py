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

import sys
import datetime

def get_current_season() -> str:
    """Auto-detect the current NBA season string (e.g. '2025-26')."""
    today = datetime.date.today()
    year = today.year if today.month >= 10 else today.year - 1
    return f"{year}-{str(year + 1)[-2:]}"


def pull_raw_matchups(incremental: bool = False):
    """
    Pulls raw data directly from NBA API for matchup strings.
    incremental=True: Only pulls the current season (fast, for daily CI).
    incremental=False: Pulls last 3 seasons (for full rebuild).
    """
    today = datetime.date.today()
    current_season = get_current_season()

    if incremental:
        season_year = int(current_season.split("-")[0])
        seasons = [current_season]
        # During playoffs (Apr-Jun), query both types; otherwise just Regular Season
        season_types = ["Regular Season", "Playoffs"] if 4 <= today.month <= 6 else ["Regular Season"]
    else:
        season_year = int(current_season.split("-")[0])
        seasons = [
            f"{season_year - 2}-{str(season_year - 1)[-2:]}",
            f"{season_year - 1}-{str(season_year)[-2:]}",
            current_season,
        ]
        season_types = ["Regular Season", "Playoffs"]

    print(f"  Seasons: {seasons} | Types: {season_types}")
    dfs = []

    for season in seasons:
      for season_type in season_types:
        label = f"{season} ({season_type})"
        print(f"Fetching raw matchup data from nba_api for {label}...")
        try:
            logs = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation='P',
                season_type_all_star=season_type
            )
            df = logs.get_data_frames()[0]
            if not df.empty:
                dfs.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching {label}: {e}")

    if not dfs:
        return pd.DataFrame()
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

def update_feature_store(incremental: bool = False):
    print("Loading existing feature store...")
    feature_df = pd.read_sql("SELECT * FROM feature_store", engine)
    feature_df['game_date'] = pd.to_datetime(feature_df['game_date'])

    raw_df = pull_raw_matchups(incremental=incremental)
    if raw_df.empty:
        print("  ⚠️  No matchup data fetched. Skipping opponent defense update.")
        return
    opp_def_df = calculate_opponent_defense(raw_df)

    print("Merging Opponent Defense into the Heavy Matrix...")
    final_df = pd.merge(feature_df, opp_def_df, on=['player_id', 'game_id'], how='left')

    # Only drop rows missing opp_def_rating when it's a full rebuild.
    # In incremental mode, existing rows already have it — only new rows may lack it.
    if not incremental:
        final_df = final_df.dropna(subset=['opp_def_rating_10'])
    else:
        # Fill NaN for new rows with league average (110)
        final_df['opp_def_rating_10'] = final_df['opp_def_rating_10'].fillna(110.0)

    print("Overwriting feature_store with new Opponent context...")
    final_df.to_sql('feature_store', engine, if_exists='replace', index=False)
    print("Done! The Matrix is now complete.")


if __name__ == "__main__":
    incremental_mode = "--full" not in sys.argv
    update_feature_store(incremental=incremental_mode)
