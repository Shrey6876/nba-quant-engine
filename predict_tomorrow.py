#!/usr/bin/env python3
"""
predict_tomorrow.py
───────────────────
Generates player-prop projections for tomorrow's NBA games.

Usage:
    python predict_tomorrow.py              # prints to stdout
    python predict_tomorrow.py > picks.txt  # saves to file

The script:
  1. Loads trained models (XGBoost for Points / Rebounds / Assists).
  2. Pulls the most recent feature row per active rotation player.
  3. Applies live modifiers (injuries → usage shifts).
  4. Runs 10,000 Monte Carlo simulations per player per stat.
  5. Outputs a formatted cheat-sheet with PRA projections & edge flags.
"""

import datetime
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

# ── Constants ────────────────────────────────────────────────────
MONTE_CARLO_SIMS = 10_000
EDGE_THRESHOLD   = 2.0   # minimum points gap to flag a bet

FEATURE_COLS = [
    'pts_ema_3', 'pts_ema_7', 'pts_ema_15',
    'reb_ema_7', 'ast_ema_7', 'min_ema_7',
    'pts_volatility_10', 'days_rest',
    'pts_career_baseline', 'expected_minutes_ratio',
    'opp_def_rating_10', 'pos_def_rating_10',
    'opp_pace_10', 'opp_fta_rate_10',
    'referee_whistle_modifier', 'miles_traveled_since_last_game',
    'time_zones_crossed', 'spatial_matchup_rating'
]


def load_models():
    """Load the trained model files."""
    try:
        pts = joblib.load('models/xgb_points_model.joblib')
        reb = joblib.load('models/xgb_rebounds_model.joblib')
        ast = joblib.load('models/xgb_assists_model.joblib')
        return pts, reb, ast
    except FileNotFoundError:
        print("❌ Models not found. Run 'python train_model.py' first.")
        sys.exit(1)


def load_injuries():
    """Read the injuries.json config file."""
    try:
        with open('injuries.json', 'r') as f:
            data = json.load(f)
            return data.get('injuries', [])
    except FileNotFoundError:
        return []


def monte_carlo(model, features: dict, variance: float) -> float:
    """Run N Monte Carlo simulations and return the mean projection."""
    input_df = pd.DataFrame([features])
    predicted_mean = model.predict(input_df)[0]
    sims = np.random.normal(loc=predicted_mean, scale=variance, size=MONTE_CARLO_SIMS)
    sims = np.maximum(sims, 0)
    return float(np.mean(sims))


def main():
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    print("=" * 62)
    print(f"  🏀 NBA QUANT ENGINE — PROJECTIONS FOR {tomorrow.isoformat()}")
    print("=" * 62)
    print()

    model_pts, model_reb, model_ast = load_models()
    injuries = load_injuries()

    # ── Pull the latest feature row per rotation player ──────
    # First try last 7 days; if off-season / no games, fall back to
    # the most recent game date available in the database.
    recent_cutoff = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()

    query = f"""
        SELECT fs.*, p.full_name
        FROM feature_store fs
        JOIN players p ON fs.player_id = p.id
        WHERE fs.game_date >= '{recent_cutoff}'
          AND fs.expected_minutes_ratio > 0.7
        ORDER BY fs.game_date DESC
    """
    df = pd.read_sql(query, engine)

    # Smart fallback: if no recent data, use the latest game date in the DB
    if df.empty:
        max_date_row = pd.read_sql("SELECT MAX(game_date) as md FROM feature_store", engine)
        max_date = max_date_row.iloc[0]['md']
        if max_date is None:
            print("  ⚠️  feature_store is empty. Run the data pipeline first.")
            return
        print(f"  ℹ️  No games in last 7 days. Using latest available data ({str(max_date)[:10]}).\n")
        query = f"""
            SELECT fs.*, p.full_name
            FROM feature_store fs
            JOIN players p ON fs.player_id = p.id
            WHERE fs.game_date = '{max_date}'
              AND fs.expected_minutes_ratio > 0.7
            ORDER BY p.full_name
        """
        df = pd.read_sql(query, engine)

    # Keep only the MOST RECENT row per player
    df = df.drop_duplicates(subset='player_id', keep='first')
    df = df.dropna(subset=FEATURE_COLS)
    df = df.sort_values('full_name')

    print(f"  📊 {len(df)} rotation players loaded\n")

    bet_count = 0

    for _, row in df.iterrows():
        player_name = row['full_name']
        features = row[FEATURE_COLS].to_dict()

        # ── Apply injury usage shifts ────────────────────────
        usage_tag = ""
        for inj in injuries:
            if (inj.get('status') == 'OUT'
                    and player_name in inj.get('usage_beneficiaries', [])):
                features['pts_ema_3']  *= 1.15
                features['pts_ema_7']  *= 1.15
                features['reb_ema_7']  *= 1.10
                features['ast_ema_7']  *= 1.20
                usage_tag = f"  ⬆️ ({inj['player']} OUT)"

        # ── Monte Carlo projections ──────────────────────────
        proj_pts = monte_carlo(model_pts, features, variance=5.5)
        proj_reb = monte_carlo(model_reb, features, variance=2.5)
        proj_ast = monte_carlo(model_ast, features, variance=2.0)
        proj_pra = proj_pts + proj_reb + proj_ast

        # ── Mock Vegas line & edge detection ─────────────────
        vegas_line = round(features['pts_ema_7'] * 2) / 2
        edge = proj_pts - vegas_line
        edge_flag = ""
        if abs(edge) >= EDGE_THRESHOLD:
            direction = "OVER" if edge > 0 else "UNDER"
            edge_flag = f"  🎯 {direction} ({abs(edge):.1f}pt edge)"
            bet_count += 1

        print(f"  {player_name:<25} PTS {proj_pts:5.1f} | REB {proj_reb:4.1f} | AST {proj_ast:4.1f} | PRA {proj_pra:5.1f}{usage_tag}{edge_flag}")

    print(f"\n{'─' * 62}")
    print(f"  🎯 Actionable edges found: {bet_count}")
    print(f"  ⚙️  Edge threshold: ≥ {EDGE_THRESHOLD} pts  |  Sims: {MONTE_CARLO_SIMS:,}")
    print(f"  🏥 Injury config: injuries.json ({len(injuries)} entries)")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
