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
  3. Applies live modifiers (injuries → usage shifts, QUESTIONABLE → minutes penalty).
  4. Runs 10,000 Monte Carlo simulations per player per stat.
  5. Outputs a formatted cheat-sheet with OVER/UNDER verdicts for PTS, REB, AST, PRA.
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
EDGE_THRESHOLD   = 2.0   # minimum pts gap to flag a strong bet
CONFIDENCE_THRESHOLD = 0.5  # minimum gap to call a lean

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


def verdict(proj: float, line: float, strong_threshold: float = 2.0) -> str:
    """
    Return a clear OVER / UNDER / PUSH verdict with confidence tier.

    🟢 OVER  / 🔴 UNDER  → strong edge (≥ threshold)
    🟡 lean OVER / lean UNDER → slight lean (≥ 0.5 but < threshold)
    ⚪ PUSH → model matches the line
    """
    edge = proj - line
    if abs(edge) < CONFIDENCE_THRESHOLD:
        return "⚪ PUSH  "
    elif edge >= strong_threshold:
        return f"🟢 OVER  "
    elif edge <= -strong_threshold:
        return f"🔴 UNDER "
    elif edge > 0:
        return f"🟡 o     "
    else:
        return f"🟡 u     "


def edge_str(proj: float, line: float) -> str:
    """Return a signed edge string like +3.2 or -1.5."""
    e = proj - line
    sign = "+" if e >= 0 else ""
    return f"{sign}{e:.1f}"


def main():
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    print("=" * 90)
    print(f"  🏀 NBA QUANT ENGINE — PROJECTIONS FOR {tomorrow.isoformat()}")
    print("=" * 90)
    print()

    model_pts, model_reb, model_ast = load_models()
    injuries = load_injuries()

    # ── Pull the latest feature row per rotation player ──────
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

    # ── Column header ────────────────────────────────────────
    hdr = f"  {'PLAYER':<25} {'PTS':>5}  {'LINE':>5}  {'CALL':^9} {'EDGE':>5} │ {'REB':>4}  {'CALL':^9} │ {'AST':>4}  {'CALL':^9} │ {'PRA':>5}"
    print(hdr)
    print(f"  {'─' * 86}")

    strong_bets = []   # collect actionable edges

    for _, row in df.iterrows():
        player_name = row['full_name']
        features = row[FEATURE_COLS].to_dict()

        # ── Apply injury modifiers ───────────────────────────
        inj_tag = ""
        for inj in injuries:
            # Usage redistribution for OUT players
            if (inj.get('status') == 'OUT'
                    and player_name in inj.get('usage_beneficiaries', [])):
                features['pts_ema_3']  *= 1.15
                features['pts_ema_7']  *= 1.15
                features['reb_ema_7']  *= 1.10
                features['ast_ema_7']  *= 1.20
                inj_tag = f" ⬆️"
            # Minutes penalty for QUESTIONABLE players
            if (inj.get('status') == 'QUESTIONABLE'
                    and inj.get('player') == player_name):
                features['expected_minutes_ratio'] *= 0.90
                inj_tag = " ⚠️"

        # ── Monte Carlo projections ──────────────────────────
        proj_pts = monte_carlo(model_pts, features, variance=5.5)
        proj_reb = monte_carlo(model_reb, features, variance=2.5)
        proj_ast = monte_carlo(model_ast, features, variance=2.0)
        proj_pra = proj_pts + proj_reb + proj_ast

        # ── Derive implied lines from rolling averages ───────
        # (proxy for Vegas: nearest 0.5 of the 7-game EMA)
        pts_line = round(features['pts_ema_7'] * 2) / 2
        reb_line = round(features['reb_ema_7'] * 2) / 2
        ast_line = round(features['ast_ema_7'] * 2) / 2
        pra_line = pts_line + reb_line + ast_line

        # ── Verdicts ─────────────────────────────────────────
        pts_v = verdict(proj_pts, pts_line)
        reb_v = verdict(proj_reb, reb_line, strong_threshold=1.5)
        ast_v = verdict(proj_ast, ast_line, strong_threshold=1.5)

        pts_e = edge_str(proj_pts, pts_line)

        name_display = f"{player_name}{inj_tag}"

        print(
            f"  {name_display:<27}"
            f" {proj_pts:5.1f}  {pts_line:5.1f}  {pts_v} {pts_e:>5}"
            f" │ {proj_reb:4.1f}  {reb_v}"
            f" │ {proj_ast:4.1f}  {ast_v}"
            f" │ {proj_pra:5.1f}"
        )

        # Collect strong bets
        pts_edge = proj_pts - pts_line
        if abs(pts_edge) >= EDGE_THRESHOLD:
            direction = "OVER" if pts_edge > 0 else "UNDER"
            strong_bets.append({
                'player': player_name,
                'stat': 'PTS',
                'line': pts_line,
                'proj': proj_pts,
                'edge': pts_edge,
                'direction': direction
            })
        reb_edge = proj_reb - reb_line
        if abs(reb_edge) >= 1.5:
            direction = "OVER" if reb_edge > 0 else "UNDER"
            strong_bets.append({
                'player': player_name,
                'stat': 'REB',
                'line': reb_line,
                'proj': proj_reb,
                'edge': reb_edge,
                'direction': direction
            })
        ast_edge = proj_ast - ast_line
        if abs(ast_edge) >= 1.5:
            direction = "OVER" if ast_edge > 0 else "UNDER"
            strong_bets.append({
                'player': player_name,
                'stat': 'AST',
                'line': ast_line,
                'proj': proj_ast,
                'edge': ast_edge,
                'direction': direction
            })

    # ── Bet Sheet: Actionable Edges ──────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  🎯 ACTIONABLE EDGES ({len(strong_bets)} found)")
    print(f"{'=' * 90}")

    if strong_bets:
        # Sort by absolute edge size (strongest first)
        strong_bets.sort(key=lambda x: abs(x['edge']), reverse=True)

        print(f"  {'PLAYER':<25} {'STAT':>4}  {'LINE':>5}  {'PROJ':>5}  {'EDGE':>6}  {'VERDICT':^12}")
        print(f"  {'─' * 72}")

        for b in strong_bets:
            icon = "🟢" if b['direction'] == "OVER" else "🔴"
            sign = "+" if b['edge'] > 0 else ""
            print(
                f"  {b['player']:<25}"
                f" {b['stat']:>4}"
                f"  {b['line']:5.1f}"
                f"  {b['proj']:5.1f}"
                f"  {sign}{b['edge']:5.1f}"
                f"  {icon} {b['direction']}"
            )
    else:
        print("  No strong edges found today. All lines look fairly priced.")

    # ── Legend ────────────────────────────────────────────────
    print(f"\n{'─' * 90}")
    print("  LEGEND:  🟢 OVER (strong)  🔴 UNDER (strong)  🟡 o/u (lean)  ⚪ PUSH (no edge)")
    print(f"           ⬆️ = usage boost (teammate OUT)  ⚠️ = player QUESTIONABLE")
    print(f"  CONFIG:  Sims: {MONTE_CARLO_SIMS:,}  |  PTS edge ≥ {EDGE_THRESHOLD}pt  |  REB/AST edge ≥ 1.5pt")
    print(f"  INJURY:  injuries.json ({len(injuries)} entries)")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
