#!/usr/bin/env python3
"""
deep_data_engine.py
───────────────────
Phase 5: Deep Data Enrichment — REAL travel distance and contextual features.
All previous np.random features replaced with deterministic calculations.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import math
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

# NBA arena coordinates (lat, lon) for distance calculation
NBA_ARENAS = {
    'ATL': (33.757, -84.396), 'BOS': (42.366, -71.062), 'BKN': (40.683, -73.975),
    'CHA': (35.225, -80.839), 'CHI': (41.881, -87.674), 'CLE': (41.496, -81.688),
    'DAL': (32.790, -96.810), 'DEN': (39.749, -105.007), 'DET': (42.341, -83.055),
    'GSW': (37.768, -122.388), 'HOU': (29.751, -95.362), 'IND': (39.764, -86.155),
    'LAC': (34.043, -118.267), 'LAL': (34.043, -118.267), 'MEM': (35.138, -90.051),
    'MIA': (25.781, -80.187), 'MIL': (43.045, -87.917), 'MIN': (44.980, -93.276),
    'NOP': (29.949, -90.082), 'NYK': (40.751, -73.994), 'OKC': (35.463, -97.515),
    'ORL': (28.539, -81.384), 'PHI': (39.901, -75.172), 'PHX': (33.446, -112.071),
    'POR': (45.532, -122.667), 'SAC': (38.580, -121.500), 'SAS': (29.427, -98.438),
    'TOR': (43.643, -79.379), 'UTA': (40.768, -111.901), 'WAS': (38.898, -77.021),
}

def haversine_miles(lat1, lon1, lat2, lon2):
    """Calculate distance in miles between two coordinates."""
    R = 3959  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

# Timezone offsets for time zone crossing calculation (Eastern = 0 base)
NBA_TIMEZONES = {
    'ATL': 0, 'BOS': 0, 'BKN': 0, 'CHA': 0, 'CHI': -1, 'CLE': 0,
    'DAL': -1, 'DEN': -2, 'DET': 0, 'GSW': -3, 'HOU': -1, 'IND': 0,
    'LAC': -3, 'LAL': -3, 'MEM': -1, 'MIA': 0, 'MIL': -1, 'MIN': -1,
    'NOP': -1, 'NYK': 0, 'OKC': -1, 'ORL': 0, 'PHI': 0, 'PHX': -2,
    'POR': -3, 'SAC': -3, 'SAS': -1, 'TOR': 0, 'UTA': -2, 'WAS': 0,
}


def calculate_travel_features(feature_df):
    """
    Calculate real travel distance and time zone crossings 
    from game-to-game schedule data.
    """
    print("  Computing travel distances from NBA arena coordinates...")
    
    # We need to know WHERE each game was played.
    # Parse from the Opponent column + home_away indicator
    # If home_away == 1 (home), the game is at the player's team arena.
    # If home_away == 0 (away), the game is at the Opponent's arena.
    
    # First, get the player's team from the game log
    team_query = """
    SELECT player_id, team_abbr, game_id
    FROM player_game_logs
    WHERE team_abbr IS NOT NULL
    """
    team_df = pd.read_sql(team_query, engine)
    
    if team_df.empty or 'Opponent' not in feature_df.columns:
        print("    ⚠️  Insufficient data for travel calculation. Using distance = 0.")
        feature_df['miles_traveled_since_last_game'] = 0.0
        feature_df['time_zones_crossed'] = 0.0
        return feature_df
    
    # Merge team info
    feature_df = pd.merge(feature_df, team_df[['player_id', 'game_id', 'team_abbr']], 
                          on=['player_id', 'game_id'], how='left', suffixes=('', '_lookup'))
    
    # Determine the city of EACH game
    def get_game_city(row):
        if row.get('home_away', 0) == 1:
            return row.get('team_abbr', '')
        else:
            return row.get('Opponent', '')
    
    feature_df['game_city'] = feature_df.apply(get_game_city, axis=1)
    
    # Sort by player and date, then calculate distances between consecutive games
    feature_df = feature_df.sort_values(['player_id', 'game_date'])
    feature_df['prev_city'] = feature_df.groupby('player_id')['game_city'].shift(1)
    
    def calc_distance(row):
        city = str(row.get('game_city', ''))
        prev = str(row.get('prev_city', ''))
        if not city or not prev or city not in NBA_ARENAS or prev not in NBA_ARENAS:
            return 0.0
        if city == prev:
            return 0.0
        lat1, lon1 = NBA_ARENAS[prev]
        lat2, lon2 = NBA_ARENAS[city]
        return haversine_miles(lat1, lon1, lat2, lon2)
    
    def calc_tz(row):
        city = str(row.get('game_city', ''))
        prev = str(row.get('prev_city', ''))
        if not city or not prev or city not in NBA_TIMEZONES or prev not in NBA_TIMEZONES:
            return 0.0
        return abs(NBA_TIMEZONES.get(city, 0) - NBA_TIMEZONES.get(prev, 0))
    
    feature_df['miles_traveled_since_last_game'] = feature_df.apply(calc_distance, axis=1)
    feature_df['time_zones_crossed'] = feature_df.apply(calc_tz, axis=1)
    
    # Clean up temp columns
    feature_df = feature_df.drop(columns=['game_city', 'prev_city', 'team_abbr_lookup'], errors='ignore')
    
    avg_miles = feature_df['miles_traveled_since_last_game'].mean()
    max_miles = feature_df['miles_traveled_since_last_game'].max()
    print(f"    Avg travel: {avg_miles:.0f} mi | Max: {max_miles:.0f} mi")
    
    return feature_df


def generate_deep_data():
    print("=" * 60)
    print("  PHASE 5: DEEP DATA ENRICHMENT (REAL DATA)")
    print("=" * 60)
    
    print("\nLoading feature store...")
    feature_df = pd.read_sql("SELECT * FROM feature_store", engine)
    feature_df['game_date'] = pd.to_datetime(feature_df['game_date'])
    
    # ── 1. Real Travel Distance ──────────────────────────────────────────
    # Drop old random columns if they exist
    feature_df = feature_df.drop(
        columns=['miles_traveled_since_last_game', 'time_zones_crossed', 
                 'referee_whistle_modifier', 'spatial_matchup_rating'],
        errors='ignore'
    )
    feature_df = calculate_travel_features(feature_df)
    
    # ── 2. Drop referee_whistle_modifier and spatial_matchup_rating ──────
    # These were random noise. We don't replace them — we DROP them.
    # home_away (from feature_engineering.py) is a much better contextual signal.
    
    # ── Save ─────────────────────────────────────────────────────────────
    print("\n  Saving enriched feature store...")
    feature_df.to_sql('feature_store', engine, if_exists='replace', index=False)
    print("  ✅ Deep Data Enrichment Complete (all real data).")

if __name__ == "__main__":
    generate_deep_data()
