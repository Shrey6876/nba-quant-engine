#!/usr/bin/env python3
"""
predict_tomorrow.py
───────────────────
Generates player-prop projections for upcoming NBA games.

Usage:
    python predict_tomorrow.py              # prints to stdout
    python predict_tomorrow.py > picks.txt  # saves to file

The script:
  1. Pulls the actual upcoming game schedule from The Odds API.
  2. Loads trained models (XGBoost for Points / Rebounds / Assists).
  3. Pulls the most recent feature row per active rotation player.
  4. Filters to ONLY players on teams scheduled to play.
  5. Applies live modifiers (injuries → usage shifts, QUESTIONABLE → minutes penalty).
  6. Runs 10,000 Monte Carlo simulations per player per stat.
  7. Outputs a formatted cheat-sheet grouped by game with OVER/UNDER verdicts.
"""

import datetime
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import requests
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
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

# ── NBA team name → abbreviation mapping ─────────────────────────
TEAM_ABBR = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
    'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
    'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
    'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
    'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS',
}


def fetch_upcoming_games():
    """Fetch the actual upcoming games from The Odds API."""
    if not ODDS_API_KEY:
        print("  ⚠️  No ODDS_API_KEY set. Cannot fetch live schedule.")
        return []

    url = (
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
        f"?regions=us&markets=h2h,spreads&oddsFormat=american&apiKey={ODDS_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        games = resp.json()

        upcoming = []
        for g in games:
            home = g['home_team']
            away = g['away_team']
            commence = g.get('commence_time', '')

            # Parse spread from DraftKings/FanDuel
            spread = 0.0
            for bk in g.get('bookmakers', []):
                if bk['key'] in ('draftkings', 'fanduel'):
                    for mkt in bk.get('markets', []):
                        if mkt['key'] == 'spreads':
                            for oc in mkt['outcomes']:
                                if oc['name'] == home:
                                    spread = oc.get('point', 0.0)
                                    break
                    if spread != 0.0:
                        break

            upcoming.append({
                'home_team': home,
                'away_team': away,
                'home_abbr': TEAM_ABBR.get(home, home[:3].upper()),
                'away_abbr': TEAM_ABBR.get(away, away[:3].upper()),
                'commence_time': commence,
                'spread': spread,
                'blowout_risk': abs(spread) > 12.0,
            })

        return upcoming
    except Exception as e:
        print(f"  ⚠️  Odds API error: {e}")
        return []


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


def get_team_players(df, team_abbrs):
    """
    Filter the feature store DataFrame to only include players from
    the specified team abbreviations using their most recent game matchup.
    """
    # We need to figure out which team each player belongs to.
    # Use the player_game_logs + games tables to find each player's team from their most recent game.
    team_query = """
        SELECT DISTINCT pgl.player_id, p.full_name
        FROM player_game_logs pgl
        JOIN players p ON pgl.player_id = p.id
        JOIN games g ON pgl.game_id = g.id
        WHERE g.game_date = (SELECT MAX(g2.game_date) FROM player_game_logs pgl2 JOIN games g2 ON pgl2.game_id = g2.id WHERE pgl2.player_id = pgl.player_id)
    """
    # This is expensive. Instead, let's use a simpler approach:
    # Get each player's latest game and match against the feature_store
    return df


def main():
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    # ── Fetch real upcoming games ────────────────────────────
    upcoming_games = fetch_upcoming_games()

    print("=" * 90)
    print(f"  🏀 NBA QUANT ENGINE — PROJECTIONS FOR {tomorrow.isoformat()}")
    print("=" * 90)

    if upcoming_games:
        print(f"\n  📅 SCHEDULED GAMES ({len(upcoming_games)}):")
        for g in upcoming_games:
            spread_str = f"({g['away_abbr']} {-g['spread']:+.1f})" if g['spread'] != 0 else ""
            blowout = " ⚠️ BLOWOUT RISK" if g['blowout_risk'] else ""
            print(f"     • {g['away_abbr']} @ {g['home_abbr']}  {spread_str}{blowout}")
    else:
        print("\n  ⚠️  No upcoming games found from Odds API.")

    print()

    model_pts, model_reb, model_ast = load_models()
    injuries = load_injuries()

    # ── Determine which team abbreviations are playing ────────
    playing_abbrs = set()
    game_lookup = {}  # team_abbr -> game info
    for g in upcoming_games:
        playing_abbrs.add(g['home_abbr'])
        playing_abbrs.add(g['away_abbr'])
        game_lookup[g['home_abbr']] = g
        game_lookup[g['away_abbr']] = g

    # ── Pull the latest feature row per rotation player ──────
    # First try the last 14 days to account for playoff rest days
    recent_cutoff = (datetime.date.today() - datetime.timedelta(days=14)).isoformat()
    max_date_row = pd.read_sql("SELECT MAX(game_date) as md FROM feature_store", engine)
    max_date = max_date_row.iloc[0]['md']

    if max_date is None:
        print("  ⚠️  feature_store is empty. Run the data pipeline first.")
        return

    # Show data freshness
    max_date_str = str(max_date)[:10]
    days_stale = (datetime.date.today() - datetime.date.fromisoformat(max_date_str)).days

    if days_stale > 3:
        print(f"  ⚠️  DATA WARNING: Feature store last updated {max_date_str} ({days_stale} days ago)")
        print(f"     Run the full pipeline to refresh: python ingest_nba_api.py && python feature_engineering.py")
        print()
    else:
        print(f"  ✅ Data fresh as of {max_date_str}\n")

    query = f"""
        SELECT fs.*, p.full_name
        FROM feature_store fs
        JOIN players p ON fs.player_id = p.id
        WHERE fs.game_date >= '{recent_cutoff}'
          AND fs.expected_minutes_ratio > 0.7
        ORDER BY fs.game_date DESC
    """
    df = pd.read_sql(query, engine)

    # Fallback if no recent data
    if df.empty:
        print(f"  ℹ️  No games in last 14 days. Using latest available data ({max_date_str}).\n")
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

    # ── Filter to only players on teams that are actually playing ─
    if playing_abbrs:
        # Match players to teams using their most recent game matchup from the DB
        team_map_query = """
            SELECT pgl.player_id, 
                   CASE 
                       WHEN INSTR(m.matchup, 'vs.') > 0 THEN SUBSTR(m.matchup, 1, 3)
                       WHEN INSTR(m.matchup, '@') > 0 THEN SUBSTR(m.matchup, 1, 3)
                       ELSE 'UNK'
                   END as team_abbr
            FROM (
                SELECT pgl2.player_id, pgl2.game_id,
                       ROW_NUMBER() OVER (PARTITION BY pgl2.player_id ORDER BY g2.game_date DESC) as rn
                FROM player_game_logs pgl2
                JOIN games g2 ON pgl2.game_id = g2.id
            ) pgl
            JOIN (
                SELECT GAME_ID, PLAYER_ID, MATCHUP
                FROM (VALUES ('placeholder', 0, 'placeholder'))
            ) m ON 1=0
            WHERE pgl.rn = 1
        """
        # SQLite doesn't store MATCHUP in our schema, so we need another approach.
        # Let's use the nba_api to get team rosters for the playing teams.
        # Simpler approach: fetch the latest season game logs and map player->team
        try:
            from nba_api.stats.endpoints import leaguegamelog
            import time
            
            # Pull current season playoff data to get player-team mapping
            logs = leaguegamelog.LeagueGameLog(
                season='2025-26',
                player_or_team_abbreviation='P',
                season_type_all_star='Playoffs'
            )
            roster_df = logs.get_data_frames()[0]
            
            # Get each player's most recent team
            roster_df = roster_df.sort_values('GAME_DATE', ascending=False)
            player_team = roster_df.drop_duplicates(subset='PLAYER_ID', keep='first')[['PLAYER_ID', 'TEAM_ABBREVIATION']]
            player_team = player_team.rename(columns={'PLAYER_ID': 'player_id', 'TEAM_ABBREVIATION': 'team_abbr'})
            player_team['player_id'] = player_team['player_id'].astype(int)
            
            # Merge team info into our feature df
            df = df.merge(player_team, on='player_id', how='left')
            
            # Filter to only teams playing
            df_playing = df[df['team_abbr'].isin(playing_abbrs)].copy()
            
            if len(df_playing) > 0:
                df = df_playing
                print(f"  📊 {len(df)} players loaded from {len(playing_abbrs)} teams\n")
            else:
                print(f"  ⚠️  Could not match players to playing teams. Showing all {len(df)} rotation players.\n")
                df['team_abbr'] = 'UNK'
        except Exception as e:
            print(f"  ⚠️  Team filtering unavailable ({e}). Showing all {len(df)} rotation players.\n")
            df['team_abbr'] = 'UNK'
    else:
        print(f"  📊 {len(df)} rotation players loaded (no live schedule available)\n")
        df['team_abbr'] = 'UNK'

    strong_bets = []   # collect actionable edges

    # ── Group output by game ────────────────────────────────
    if upcoming_games and 'team_abbr' in df.columns:
        printed_games = set()
        for game in upcoming_games:
            game_key = f"{game['away_abbr']}@{game['home_abbr']}"
            if game_key in printed_games:
                continue
            printed_games.add(game_key)

            spread_str = f" | Spread: {game['home_abbr']} {game['spread']:+.1f}" if game['spread'] != 0 else ""
            blowout_str = " | ⚠️ BLOWOUT RISK" if game['blowout_risk'] else ""

            print(f"  ┌─────────────────────────────────────────────────────────────────────────────────────┐")
            print(f"  │  🏀 {game['away_abbr']} @ {game['home_abbr']}{spread_str}{blowout_str}")
            print(f"  └─────────────────────────────────────────────────────────────────────────────────────┘")

            game_teams = {game['home_abbr'], game['away_abbr']}
            game_df = df[df['team_abbr'].isin(game_teams)].sort_values('full_name')

            if game_df.empty:
                print(f"  (No rotation players found for this game)\n")
                continue

            # Column header
            hdr = f"  {'PLAYER':<25} {'PTS':>5}  {'LINE':>5}  {'CALL':^9} {'EDGE':>5} │ {'REB':>4}  {'CALL':^9} │ {'AST':>4}  {'CALL':^9} │ {'PRA':>5}"
            print(hdr)
            print(f"  {'─' * 86}")

            for _, row in game_df.iterrows():
                player_name = row['full_name']
                team = row['team_abbr']
                features = row[FEATURE_COLS].to_dict()

                # ── Apply injury modifiers ───────────────────────────
                inj_tag = ""
                for inj in injuries:
                    if (inj.get('status') == 'OUT'
                            and player_name in inj.get('usage_beneficiaries', [])):
                        features['pts_ema_3']  *= 1.15
                        features['pts_ema_7']  *= 1.15
                        features['reb_ema_7']  *= 1.10
                        features['ast_ema_7']  *= 1.20
                        inj_tag = f" ⬆️"
                    if (inj.get('status') == 'QUESTIONABLE'
                            and inj.get('player') == player_name):
                        features['expected_minutes_ratio'] *= 0.90
                        inj_tag = " ⚠️"

                # ── Apply blowout penalty for this game ──────────────
                blowout_tag = ""
                if game['blowout_risk']:
                    features['min_ema_7'] *= 0.85
                    features['expected_minutes_ratio'] = features['min_ema_7'] / 36.0
                    blowout_tag = " 📉"

                # ── Monte Carlo projections ──────────────────────────
                proj_pts = monte_carlo(model_pts, features, variance=5.5)
                proj_reb = monte_carlo(model_reb, features, variance=2.5)
                proj_ast = monte_carlo(model_ast, features, variance=2.0)
                proj_pra = proj_pts + proj_reb + proj_ast

                # ── Derive implied lines from rolling averages ───────
                pts_line = round(features['pts_ema_7'] * 2) / 2
                reb_line = round(features['reb_ema_7'] * 2) / 2
                ast_line = round(features['ast_ema_7'] * 2) / 2

                # ── Verdicts ─────────────────────────────────────────
                pts_v = verdict(proj_pts, pts_line)
                reb_v = verdict(proj_reb, reb_line, strong_threshold=1.5)
                ast_v = verdict(proj_ast, ast_line, strong_threshold=1.5)

                pts_e = edge_str(proj_pts, pts_line)
                name_display = f"{player_name} ({team}){inj_tag}{blowout_tag}"

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
                        'player': player_name, 'team': team, 'game': game_key,
                        'stat': 'PTS', 'line': pts_line, 'proj': proj_pts,
                        'edge': pts_edge, 'direction': direction
                    })
                reb_edge = proj_reb - reb_line
                if abs(reb_edge) >= 1.5:
                    direction = "OVER" if reb_edge > 0 else "UNDER"
                    strong_bets.append({
                        'player': player_name, 'team': team, 'game': game_key,
                        'stat': 'REB', 'line': reb_line, 'proj': proj_reb,
                        'edge': reb_edge, 'direction': direction
                    })
                ast_edge = proj_ast - ast_line
                if abs(ast_edge) >= 1.5:
                    direction = "OVER" if ast_edge > 0 else "UNDER"
                    strong_bets.append({
                        'player': player_name, 'team': team, 'game': game_key,
                        'stat': 'AST', 'line': ast_line, 'proj': proj_ast,
                        'edge': ast_edge, 'direction': direction
                    })

            print()  # spacing between games

    else:
        # Fallback: no games or no team mapping — show all
        hdr = f"  {'PLAYER':<25} {'PTS':>5}  {'LINE':>5}  {'CALL':^9} {'EDGE':>5} │ {'REB':>4}  {'CALL':^9} │ {'AST':>4}  {'CALL':^9} │ {'PRA':>5}"
        print(hdr)
        print(f"  {'─' * 86}")

        for _, row in df.iterrows():
            player_name = row['full_name']
            features = row[FEATURE_COLS].to_dict()

            inj_tag = ""
            for inj in injuries:
                if (inj.get('status') == 'OUT'
                        and player_name in inj.get('usage_beneficiaries', [])):
                    features['pts_ema_3']  *= 1.15
                    features['pts_ema_7']  *= 1.15
                    features['reb_ema_7']  *= 1.10
                    features['ast_ema_7']  *= 1.20
                    inj_tag = f" ⬆️"
                if (inj.get('status') == 'QUESTIONABLE'
                        and inj.get('player') == player_name):
                    features['expected_minutes_ratio'] *= 0.90
                    inj_tag = " ⚠️"

            proj_pts = monte_carlo(model_pts, features, variance=5.5)
            proj_reb = monte_carlo(model_reb, features, variance=2.5)
            proj_ast = monte_carlo(model_ast, features, variance=2.0)
            proj_pra = proj_pts + proj_reb + proj_ast

            pts_line = round(features['pts_ema_7'] * 2) / 2
            reb_line = round(features['reb_ema_7'] * 2) / 2
            ast_line = round(features['ast_ema_7'] * 2) / 2

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

    # ── Bet Sheet: Actionable Edges ──────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  🎯 ACTIONABLE EDGES ({len(strong_bets)} found)")
    print(f"{'=' * 90}")

    if strong_bets:
        strong_bets.sort(key=lambda x: abs(x['edge']), reverse=True)

        print(f"  {'PLAYER':<25} {'STAT':>4}  {'LINE':>5}  {'PROJ':>5}  {'EDGE':>6}  {'VERDICT':^12}")
        print(f"  {'─' * 72}")

        for b in strong_bets:
            icon = "🟢" if b['direction'] == "OVER" else "🔴"
            sign = "+" if b['edge'] > 0 else ""
            game_tag = f" [{b.get('game', '')}]" if b.get('game') else ""
            print(
                f"  {b['player']:<25}"
                f" {b['stat']:>4}"
                f"  {b['line']:5.1f}"
                f"  {b['proj']:5.1f}"
                f"  {sign}{b['edge']:5.1f}"
                f"  {icon} {b['direction']}{game_tag}"
            )
    else:
        print("  No strong edges found today. All lines look fairly priced.")

    # ── Legend ────────────────────────────────────────────────
    print(f"\n{'─' * 90}")
    print("  LEGEND:  🟢 OVER (strong)  🔴 UNDER (strong)  🟡 o/u (lean)  ⚪ PUSH (no edge)")
    print(f"           ⬆️ = usage boost (teammate OUT)  ⚠️ = player QUESTIONABLE  📉 = blowout penalty")
    print(f"  CONFIG:  Sims: {MONTE_CARLO_SIMS:,}  |  PTS edge ≥ {EDGE_THRESHOLD}pt  |  REB/AST edge ≥ 1.5pt")
    print(f"  DATA:    Feature store through {max_date_str}  |  Injuries: {len(injuries)} entries")
    print(f"  GAMES:   {len(upcoming_games)} scheduled")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
