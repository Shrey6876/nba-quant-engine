#!/usr/bin/env python3
"""
feature_engineering.py
─────────────────────
Phase 1: Builds the Heavy Matrix from raw game logs.
Now includes USG%, FGA/FTA EMAs, home/away, and expanded stat features.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

def load_raw_data():
    """Loads raw game logs with expanded fields from SQLite."""
    print("Loading raw data from database (with expanded box score fields)...")
    query = """
    SELECT 
        p.id as log_id, p.player_id, p.game_id, p.minutes, p.points, p.rebounds, 
        p.assists, p.threes_made,
        p.field_goals_attempted, p.free_throws_attempted, p.turnovers,
        p.plus_minus, p.fg_pct, p.ft_pct,
        p.matchup, p.team_abbr,
        g.game_date
    FROM player_game_logs p
    JOIN games g ON p.game_id = g.id
    ORDER BY p.player_id, g.game_date ASC
    """
    df = pd.read_sql(query, engine)
    df['game_date'] = pd.to_datetime(df['game_date'])
    print(f"  Loaded {len(df)} game logs.")
    return df

def calculate_complex_features(df):
    """
    Builds the Heavy Matrix — all features calculated from real data.
    """
    print("Building The Heavy Matrix (Complex Feature Engineering)...")
    
    df = df.sort_values(by=['player_id', 'game_date'])
    
    # ── 1. Time-Series Dynamics (EMA) ────────────────────────────────────
    print("  [1/8] EMA features (PTS, REB, AST, MIN)...")
    df['pts_ema_3'] = df.groupby('player_id')['points'].transform(lambda x: x.ewm(span=3, adjust=False).mean().shift(1))
    df['pts_ema_7'] = df.groupby('player_id')['points'].transform(lambda x: x.ewm(span=7, adjust=False).mean().shift(1))
    df['pts_ema_15'] = df.groupby('player_id')['points'].transform(lambda x: x.ewm(span=15, adjust=False).mean().shift(1))
    
    df['reb_ema_7'] = df.groupby('player_id')['rebounds'].transform(lambda x: x.ewm(span=7, adjust=False).mean().shift(1))
    df['ast_ema_7'] = df.groupby('player_id')['assists'].transform(lambda x: x.ewm(span=7, adjust=False).mean().shift(1))
    df['min_ema_7'] = df.groupby('player_id')['minutes'].transform(lambda x: x.ewm(span=7, adjust=False).mean().shift(1))
    
    # ── 2. Volatility/Variance Tracking ──────────────────────────────────
    print("  [2/8] Volatility features...")
    df['pts_volatility_10'] = df.groupby('player_id')['points'].transform(lambda x: x.rolling(window=10).std().shift(1))
    
    # ── 3. Rest Days ─────────────────────────────────────────────────────
    print("  [3/8] Rest days...")
    df['days_rest'] = df.groupby('player_id')['game_date'].diff().dt.days
    df['days_rest'] = df['days_rest'].clip(upper=7).fillna(7)
    
    # ── 4. Career Baseline ───────────────────────────────────────────────
    print("  [4/8] Career baseline (82-game rolling)...")
    df['pts_career_baseline'] = df.groupby('player_id')['points'].transform(lambda x: x.rolling(window=82, min_periods=10).mean().shift(1))
    
    # ── 5. Minutes Projection ────────────────────────────────────────────
    df['expected_minutes_ratio'] = df['min_ema_7'] / 36.0
    
    # ── 6. NEW: Usage Rate (USG%) ────────────────────────────────────────
    print("  [5/8] Usage Rate (USG%) features...")
    # USG% proxy from individual box score:  
    # USG% ≈ (FGA + 0.44*FTA + TOV) / MINUTES * 48
    # This is a per-minute opportunity rate, not true team-adjusted USG%,
    # but it's a strong signal that doesn't require team aggregate data.
    
    has_advanced = df['field_goals_attempted'].notna()
    df.loc[has_advanced, 'raw_usage'] = (
        df.loc[has_advanced, 'field_goals_attempted'] + 
        0.44 * df.loc[has_advanced, 'free_throws_attempted'].fillna(0) + 
        df.loc[has_advanced, 'turnovers'].fillna(0)
    ) / df.loc[has_advanced, 'minutes'].clip(lower=1) * 48
    
    df['usg_rate_10'] = df.groupby('player_id')['raw_usage'].transform(
        lambda x: x.ewm(span=10, adjust=False).mean().shift(1)
    )
    df['usg_delta_5'] = df.groupby('player_id')['raw_usage'].transform(
        lambda x: x.diff(5)
    )
    
    # ── 7. NEW: FGA/FTA/PLUS_MINUS EMAs ──────────────────────────────────
    print("  [6/8] FGA, FTA, +/- EMAs...")
    df['fga_ema_7'] = df.groupby('player_id')['field_goals_attempted'].transform(
        lambda x: x.ewm(span=7, adjust=False).mean().shift(1)
    )
    df['fta_ema_7'] = df.groupby('player_id')['free_throws_attempted'].transform(
        lambda x: x.ewm(span=7, adjust=False).mean().shift(1)
    )
    df['plus_minus_ema_7'] = df.groupby('player_id')['plus_minus'].transform(
        lambda x: x.ewm(span=7, adjust=False).mean().shift(1)
    )
    
    # ── 8. NEW: Home/Away ────────────────────────────────────────────────
    print("  [7/8] Home/away indicator...")
    # MATCHUP: "BOS vs. NYK" = home, "BOS @ NYK" = away
    df['home_away'] = 0.0
    if 'matchup' in df.columns and df['matchup'].notna().any():
        df.loc[df['matchup'].str.contains('vs.', na=False), 'home_away'] = 1.0
        df.loc[df['matchup'].str.contains('@', na=False), 'home_away'] = 0.0
    
    # ── 9. NEW: 3-Pointers EMA ───────────────────────────────────────────
    print("  [8/8] 3PM EMA...")
    df['threes_ema_7'] = df.groupby('player_id')['threes_made'].transform(
        lambda x: x.ewm(span=7, adjust=False).mean().shift(1)
    )
    
    # ── Drop intermediate columns ────────────────────────────────────────
    # NOTE: team_abbr is intentionally kept — predict_tomorrow.py uses it for game filtering.
    # matchup is dropped (we've extracted home_away from it already).
    drop_cols = ['raw_usage', 'matchup', 'field_goals_attempted',
                 'free_throws_attempted', 'turnovers', 'plus_minus', 'fg_pct', 'ft_pct']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    df_clean = df.dropna(subset=[
        'pts_ema_3', 'pts_ema_7', 'pts_ema_15', 'reb_ema_7', 'ast_ema_7', 'min_ema_7',
        'pts_volatility_10', 'pts_career_baseline'
    ]).copy()
    
    # Fill remaining NaN in new features with 0 (for records before backfill)
    for col in ['usg_rate_10', 'usg_delta_5', 'fga_ema_7', 'fta_ema_7', 
                'plus_minus_ema_7', 'threes_ema_7']:
        df_clean[col] = df_clean[col].fillna(0)
    
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
