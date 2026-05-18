#!/usr/bin/env python3
"""
predict_tomorrow.py
───────────────────
Generates player-prop projections for upcoming NBA games.
Now uses REAL sportsbook lines, Expected Value, and Kelly Criterion sizing.
"""

import datetime, json, os, sys
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

MONTE_CARLO_SIMS = 10_000
EV_THRESHOLD = 3.0        # minimum EV% to flag as actionable
BANKROLL = 1000.0          # default bankroll for Kelly sizing

FEATURE_COLS = [
    'pts_ema_3', 'pts_ema_7', 'pts_ema_15',
    'reb_ema_7', 'ast_ema_7', 'min_ema_7',
    'pts_volatility_10', 'days_rest',
    'pts_career_baseline', 'expected_minutes_ratio',
    'opp_def_rating_10', 'pos_def_rating_10',
    'opp_pace_10', 'opp_fta_rate_10',
    'usg_rate_10', 'usg_delta_5',
    'fga_ema_7', 'fta_ema_7', 'plus_minus_ema_7',
    'home_away',
    'miles_traveled_since_last_game', 'time_zones_crossed',
    'threes_ema_7',
]

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

# ── Odds math ────────────────────────────────────────────────────────────

def american_to_implied(odds):
    if odds < 0: return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)

def american_to_decimal(odds):
    if odds < 0: return 1 + (100 / abs(odds))
    return 1 + (odds / 100)

def devig(over_odds, under_odds):
    p_o = american_to_implied(over_odds)
    p_u = american_to_implied(under_odds)
    t = p_o + p_u
    return (p_o / t, p_u / t) if t > 0 else (0.5, 0.5)

def kelly_criterion(model_prob, odds, fraction=0.25):
    """Quarter Kelly for conservative sizing."""
    b = american_to_decimal(odds) - 1
    if b <= 0: return 0
    q = 1 - model_prob
    k = (b * model_prob - q) / b
    return max(k * fraction, 0)

def ev_pct(model_prob, odds):
    """Expected value as a percentage."""
    decimal = american_to_decimal(odds)
    return (model_prob * decimal - 1) * 100

# ── Data loaders ─────────────────────────────────────────────────────────

def fetch_upcoming_games():
    if not ODDS_API_KEY: return []
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?regions=us&markets=h2h,spreads&oddsFormat=american&apiKey={ODDS_API_KEY}"
    try:
        resp = requests.get(url, timeout=15); resp.raise_for_status()
        games = resp.json()
        upcoming = []
        for g in games:
            home, away = g['home_team'], g['away_team']
            spread = 0.0
            for bk in g.get('bookmakers', []):
                if bk['key'] in ('draftkings', 'fanduel'):
                    for mkt in bk.get('markets', []):
                        if mkt['key'] == 'spreads':
                            for oc in mkt['outcomes']:
                                if oc['name'] == home:
                                    spread = oc.get('point', 0.0); break
                    if spread != 0.0: break
            upcoming.append({
                'home_team': home, 'away_team': away,
                'home_abbr': TEAM_ABBR.get(home, home[:3].upper()),
                'away_abbr': TEAM_ABBR.get(away, away[:3].upper()),
                'spread': spread, 'blowout_risk': abs(spread) > 12.0,
            })
        return upcoming
    except Exception as e:
        print(f"  ⚠️  Odds API error: {e}"); return []

def load_real_lines():
    """Load real sportsbook prop lines from the database."""
    today = datetime.date.today().isoformat()
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    try:
        df = pd.read_sql(f"""
            SELECT player_name, stat_type, line, over_odds, under_odds,
                   fair_prob_over, fair_prob_under, book_name
            FROM live_prop_lines
            WHERE game_date >= '{today}' AND game_date <= '{tomorrow}'
        """, engine)
        if not df.empty:
            # Keep best line per player per stat (highest over odds = best for bettor)
            best = df.sort_values('over_odds', ascending=False).drop_duplicates(
                subset=['player_name', 'stat_type'], keep='first'
            )
            return best
    except: pass
    return pd.DataFrame()

def load_models():
    try:
        pts = joblib.load('models/xgb_points_model.joblib')
        reb = joblib.load('models/xgb_rebounds_model.joblib')
        ast = joblib.load('models/xgb_assists_model.joblib')
        return pts, reb, ast
    except:
        print("❌ Models not found. Run 'python train_model.py' first."); sys.exit(1)

def load_injuries():
    try:
        with open('injuries.json', 'r') as f: return json.load(f).get('injuries', [])
    except: return []

# ── Simulation ───────────────────────────────────────────────────────────

def monte_carlo_full(model, features, variance, line):
    """Run MC sims and return (mean, prob_over, prob_under)."""
    input_df = pd.DataFrame([features])
    predicted_mean = model.predict(input_df)[0]
    sims = np.maximum(np.random.normal(loc=predicted_mean, scale=variance, size=MONTE_CARLO_SIMS), 0)
    mean = float(np.mean(sims))
    prob_over = float(np.mean(sims > line))
    prob_under = float(np.mean(sims < line))
    return mean, prob_over, prob_under

# ── Display helpers ──────────────────────────────────────────────────────

def verdict_str(ev, direction):
    if abs(ev) < 2.0: return "⚪ PUSH  "
    if direction == 'OVER':
        return "🟢 OVER  " if ev >= EV_THRESHOLD else "🟡 o     "
    else:
        return "🔴 UNDER " if ev >= EV_THRESHOLD else "🟡 u     "

# ── Main ─────────────────────────────────────────────────────────────────

def main():
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    upcoming_games = fetch_upcoming_games()
    real_lines = load_real_lines()
    has_real_lines = not real_lines.empty

    print("=" * 94)
    print(f"  🏀 NBA QUANT ENGINE v2.0 — PROJECTIONS FOR {tomorrow.isoformat()}")
    print("=" * 94)

    if upcoming_games:
        print(f"\n  📅 SCHEDULED GAMES ({len(upcoming_games)}):")
        for g in upcoming_games:
            s = f"({g['away_abbr']} {-g['spread']:+.1f})" if g['spread'] != 0 else ""
            print(f"     • {g['away_abbr']} @ {g['home_abbr']}  {s}")

    if has_real_lines:
        n_players = real_lines['player_name'].nunique()
        n_books = real_lines['book_name'].nunique() if 'book_name' in real_lines.columns else 0
        print(f"\n  📊 REAL LINES: {len(real_lines)} props for {n_players} players from {n_books} books")
    else:
        print(f"\n  ⚠️  No real sportsbook lines found. Using EMA-derived lines (fallback).")
        print(f"     Run 'python ingest_player_props.py' to fetch real lines.")

    model_pts, model_reb, model_ast = load_models()
    injuries = load_injuries()

    # Determine playing teams
    playing_abbrs = set()
    game_lookup = {}
    for g in upcoming_games:
        playing_abbrs.update([g['home_abbr'], g['away_abbr']])
        game_lookup[g['home_abbr']] = g
        game_lookup[g['away_abbr']] = g

    # Load feature store
    max_date_row = pd.read_sql("SELECT MAX(game_date) as md FROM feature_store", engine)
    max_date = str(max_date_row.iloc[0]['md'])[:10]
    days_stale = (datetime.date.today() - datetime.date.fromisoformat(max_date)).days

    if days_stale > 3:
        print(f"\n  ⚠️  DATA WARNING: Feature store last updated {max_date} ({days_stale} days ago)")
    else:
        print(f"\n  ✅ Data fresh as of {max_date}")

    recent_cutoff = (datetime.date.today() - datetime.timedelta(days=14)).isoformat()
    df = pd.read_sql(f"""
        SELECT fs.*, p.full_name FROM feature_store fs
        JOIN players p ON fs.player_id = p.id
        WHERE fs.game_date >= '{recent_cutoff}' AND fs.expected_minutes_ratio > 0.7
        ORDER BY fs.game_date DESC
    """, engine)

    if df.empty:
        df = pd.read_sql(f"""
            SELECT fs.*, p.full_name FROM feature_store fs
            JOIN players p ON fs.player_id = p.id
            WHERE fs.game_date = '{max_date}' AND fs.expected_minutes_ratio > 0.7
        """, engine)

    df = df.drop_duplicates(subset='player_id', keep='first')
    
    # Use available features only
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=[c for c in available_features if c in ['pts_ema_3', 'pts_ema_7', 'reb_ema_7', 'ast_ema_7', 'min_ema_7']])
    # Fill NaN in new features with 0
    for c in available_features:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Use team_abbr from the feature store (preserved by feature_engineering.py).
    # The query above orders by game_date DESC + drop_duplicates(player_id) so each
    # player's row reflects their MOST RECENT team assignment — no API call needed.
    if 'team_abbr' in df.columns and playing_abbrs:
        df_playing = df[df['team_abbr'].isin(playing_abbrs)].copy()
        if len(df_playing) > 0:
            df = df_playing
            print(f"  📊 {len(df)} players matched to today's {len(playing_abbrs)} playing teams\n")
        else:
            print(f"  ⚠️  No players matched playing teams {playing_abbrs}.")
            print(f"      Feature store max_date={max_date} ({days_stale}d ago). Run ingest to refresh.")
            df['team_abbr'] = 'UNK'
    else:
        if 'team_abbr' not in df.columns:
            print("  ⚠️  team_abbr not in feature store. Re-run feature_engineering.py to rebuild.")
        df['team_abbr'] = df.get('team_abbr', 'UNK')

    strong_bets = []

    # ── Output by game ───────────────────────────────────────────────
    for game in (upcoming_games or [{'home_abbr': 'ALL', 'away_abbr': '', 'spread': 0, 'blowout_risk': False}]):
        game_key = f"{game['away_abbr']}@{game['home_abbr']}"
        spread_str = f" | Spread: {game['home_abbr']} {game['spread']:+.1f}" if game['spread'] != 0 else ""

        print(f"  ┌{'─'*91}┐")
        print(f"  │  🏀 {game['away_abbr']} @ {game['home_abbr']}{spread_str}")
        print(f"  └{'─'*91}┘")

        game_teams = {game['home_abbr'], game['away_abbr']}
        game_df = df[df['team_abbr'].isin(game_teams)].sort_values('full_name') if 'team_abbr' in df.columns else df

        if game_df.empty:
            print(f"  (No rotation players found)\n"); continue

        if has_real_lines:
            print(f"  {'PLAYER':<27} {'PTS':>5} {'LINE':>5} {'ODDS':>6} {'MODEL%':>6} {'MKT%':>5} {'EV%':>5} {'CALL':^9} {'KELLY':>6} │ {'REB':>4} {'CALL':^7} │ {'AST':>4} {'CALL':^7}")
        else:
            print(f"  {'PLAYER':<27} {'PTS':>5} {'LINE':>5} {'CALL':^9} {'EDGE':>5} │ {'REB':>4} {'CALL':^7} │ {'AST':>4} {'CALL':^7} │ {'PRA':>5}")
        print(f"  {'─'*91}")

        for _, row in game_df.iterrows():
            player_name = row['full_name']
            team = row.get('team_abbr', 'UNK')
            features = {c: row[c] for c in available_features if c in row.index}

            # Injury modifiers
            inj_tag = ""
            for inj in injuries:
                if inj.get('status') == 'OUT' and player_name in inj.get('usage_beneficiaries', []):
                    for k in ['pts_ema_3', 'pts_ema_7']: features[k] = features.get(k, 0) * 1.15
                    features['reb_ema_7'] = features.get('reb_ema_7', 0) * 1.10
                    features['ast_ema_7'] = features.get('ast_ema_7', 0) * 1.20
                    inj_tag = " ⬆️"
                if inj.get('status') == 'QUESTIONABLE' and inj.get('player') == player_name:
                    features['expected_minutes_ratio'] = features.get('expected_minutes_ratio', 1) * 0.90
                    inj_tag = " ⚠️"

            if game.get('blowout_risk'):
                features['min_ema_7'] = features.get('min_ema_7', 30) * 0.85
                features['expected_minutes_ratio'] = features.get('min_ema_7', 30) / 36.0

            # Get real line or fallback
            def get_real_line(stat_type):
                if has_real_lines:
                    match = real_lines[(real_lines['player_name'] == player_name) & (real_lines['stat_type'] == stat_type)]
                    if not match.empty:
                        r = match.iloc[0]
                        return float(r['line']), int(r['over_odds']), int(r['under_odds'])
                return None, None, None

            pts_line_real, pts_over_odds, pts_under_odds = get_real_line('player_points')
            reb_line_real, reb_over_odds, reb_under_odds = get_real_line('player_rebounds')
            ast_line_real, ast_over_odds, ast_under_odds = get_real_line('player_assists')

            pts_line = pts_line_real if pts_line_real else round(features.get('pts_ema_7', 0) * 2) / 2
            reb_line = reb_line_real if reb_line_real else round(features.get('reb_ema_7', 0) * 2) / 2
            ast_line = ast_line_real if ast_line_real else round(features.get('ast_ema_7', 0) * 2) / 2

            # Monte Carlo
            proj_pts, prob_over_pts, _ = monte_carlo_full(model_pts, features, 5.5, pts_line)
            proj_reb, prob_over_reb, _ = monte_carlo_full(model_reb, features, 2.5, reb_line)
            proj_ast, prob_over_ast, _ = monte_carlo_full(model_ast, features, 2.0, ast_line)
            proj_pra = proj_pts + proj_reb + proj_ast

            name_display = f"{player_name} ({team}){inj_tag}"

            if has_real_lines and pts_over_odds is not None:
                # Real lines mode: show EV% and Kelly
                market_prob = american_to_implied(pts_over_odds)
                pts_ev = ev_pct(prob_over_pts, pts_over_odds)
                pts_ev_under = ev_pct(1 - prob_over_pts, pts_under_odds) if pts_under_odds else 0

                # Pick the better side
                if pts_ev >= pts_ev_under:
                    best_ev, best_dir = pts_ev, 'OVER'
                    best_prob, best_odds = prob_over_pts, pts_over_odds
                else:
                    best_ev, best_dir = pts_ev_under, 'UNDER'
                    best_prob, best_odds = 1 - prob_over_pts, pts_under_odds

                kelly = kelly_criterion(best_prob, best_odds) if best_ev > 0 else 0
                v = verdict_str(best_ev, best_dir)

                reb_edge = proj_reb - reb_line
                ast_edge = proj_ast - ast_line
                reb_v = "🟢 O " if reb_edge >= 1.5 else ("🔴 U " if reb_edge <= -1.5 else "⚪   ")
                ast_v = "🟢 O " if ast_edge >= 1.5 else ("🔴 U " if ast_edge <= -1.5 else "⚪   ")

                print(
                    f"  {name_display:<27}"
                    f" {proj_pts:5.1f} {pts_line:5.1f} {pts_over_odds:>+5d}"
                    f" {prob_over_pts*100:5.1f}% {market_prob*100:4.1f}%"
                    f" {best_ev:+5.1f} {v} {kelly*100:5.2f}%"
                    f" │ {proj_reb:4.1f} {reb_v}"
                    f" │ {proj_ast:4.1f} {ast_v}"
                )

                if best_ev >= EV_THRESHOLD:
                    strong_bets.append({
                        'player': player_name, 'team': team, 'game': game_key,
                        'stat': 'PTS', 'line': pts_line, 'proj': proj_pts,
                        'ev': best_ev, 'direction': best_dir, 'odds': best_odds,
                        'model_prob': best_prob, 'kelly': kelly,
                        'wager': round(kelly * BANKROLL, 2),
                    })
            else:
                # Fallback mode: EMA-derived lines
                pts_edge = proj_pts - pts_line
                reb_edge = proj_reb - reb_line
                ast_edge = proj_ast - ast_line
                
                pts_v = "🟢 OVER  " if pts_edge >= 2.0 else ("🔴 UNDER " if pts_edge <= -2.0 else ("🟡 o     " if pts_edge > 0.5 else ("🟡 u     " if pts_edge < -0.5 else "⚪ PUSH  ")))
                reb_v = "🟢 O " if reb_edge >= 1.5 else ("🔴 U " if reb_edge <= -1.5 else "⚪   ")
                ast_v = "🟢 O " if ast_edge >= 1.5 else ("🔴 U " if ast_edge <= -1.5 else "⚪   ")

                print(
                    f"  {name_display:<27}"
                    f" {proj_pts:5.1f} {pts_line:5.1f} {pts_v} {pts_edge:+5.1f}"
                    f" │ {proj_reb:4.1f} {reb_v}"
                    f" │ {proj_ast:4.1f} {ast_v}"
                    f" │ {proj_pra:5.1f}"
                )

                if abs(pts_edge) >= 2.0:
                    strong_bets.append({
                        'player': player_name, 'team': team, 'game': game_key,
                        'stat': 'PTS', 'line': pts_line, 'proj': proj_pts,
                        'ev': pts_edge, 'direction': 'OVER' if pts_edge > 0 else 'UNDER',
                        'odds': -110, 'model_prob': prob_over_pts if pts_edge > 0 else 1-prob_over_pts,
                        'kelly': 0, 'wager': 0,
                    })
        print()

    # ── Actionable Edges ─────────────────────────────────────────────
    print(f"{'='*94}")
    total_wager = sum(b.get('wager', 0) for b in strong_bets)
    bankroll_str = f"  |  BANKROLL: ${BANKROLL:,.0f}  |  EXPOSURE: ${total_wager:,.2f}" if has_real_lines else ""
    print(f"  🎯 ACTIONABLE EDGES ({len(strong_bets)} found){bankroll_str}")
    print(f"{'='*94}")

    if strong_bets:
        strong_bets.sort(key=lambda x: abs(x.get('ev', 0)), reverse=True)
        if has_real_lines:
            print(f"  {'PLAYER':<25} {'STAT':>4} {'LINE':>5} {'PROJ':>5} {'ODDS':>6} {'MODEL%':>6} {'EV%':>6} {'KELLY':>6} {'WAGER':>7} {'CALL'}")
        else:
            print(f"  {'PLAYER':<25} {'STAT':>4} {'LINE':>5} {'PROJ':>5} {'EDGE':>6} {'CALL'}")
        print(f"  {'─'*88}")

        for b in strong_bets:
            icon = "🟢" if b['direction'] == "OVER" else "🔴"
            game_tag = f" [{b.get('game', '')}]"
            if has_real_lines:
                print(
                    f"  {b['player']:<25} {b['stat']:>4} {b['line']:5.1f} {b['proj']:5.1f}"
                    f" {b['odds']:>+5d} {b['model_prob']*100:5.1f}%"
                    f" {b['ev']:>+5.1f}% {b['kelly']*100:5.2f}%"
                    f" ${b['wager']:6.2f}"
                    f"  {icon} {b['direction']}{game_tag}"
                )
            else:
                print(
                    f"  {b['player']:<25} {b['stat']:>4} {b['line']:5.1f} {b['proj']:5.1f}"
                    f" {b['ev']:>+5.1f}"
                    f"  {icon} {b['direction']}{game_tag}"
                )

    # ── Legend ────────────────────────────────────────────────────────
    print(f"\n{'─'*94}")
    lines_source = "REAL sportsbook lines" if has_real_lines else "EMA-derived lines (fallback)"
    print(f"  LINES:   {lines_source}")
    print(f"  CONFIG:  Sims: {MONTE_CARLO_SIMS:,}  |  EV threshold: {EV_THRESHOLD}%  |  Kelly: Quarter (÷4)")
    print(f"  DATA:    Feature store through {max_date}  |  Injuries: {len(injuries)} entries  |  Games: {len(upcoming_games)}")
    print(f"{'='*94}")


if __name__ == "__main__":
    main()
