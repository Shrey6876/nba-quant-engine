import os
import json
import requests
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

def get_live_game_context():
    """Pulls live odds to determine spreads, and infers today's schedule."""
    print("Fetching Live Odds & Today's Schedule from Vegas...")
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/?regions=us&markets=h2h,spreads&oddsFormat=american&apiKey={ODDS_API_KEY}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        games = response.json()
        
        game_contexts = {}
        for game in games:
            home_team = game['home_team']
            away_team = game['away_team']
            
            spread = 0
            for bookmaker in game.get('bookmakers', []):
                if bookmaker['key'] in ['draftkings', 'fanduel']:
                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'spreads':
                            for outcome in market['outcomes']:
                                if outcome['name'] == home_team:
                                    spread = outcome.get('point', 0)
                                    break
            
            blowout_risk = abs(spread) > 12.0
            
            game_contexts[home_team] = {'spread': spread, 'blowout_risk': blowout_risk, 'opponent': away_team}
            game_contexts[away_team] = {'spread': -spread, 'blowout_risk': blowout_risk, 'opponent': home_team}
            
        return game_contexts
    except Exception as e:
        print(f"Error fetching Odds API: {e}")
        return {}

def load_injuries():
    try:
        with open('injuries.json', 'r') as f:
            data = json.load(f)
            return data.get('injuries', [])
    except FileNotFoundError:
        return []

def run_monte_carlo_simulation(model, features_dict, variance, n_simulations=10000):
    input_df = pd.DataFrame([features_dict])
    predicted_mean = model.predict(input_df)[0]
    simulations = np.random.normal(loc=predicted_mean, scale=variance, size=n_simulations)
    simulations = np.maximum(simulations, 0)
    return predicted_mean, simulations

def generate_automated_cheat_sheet():
    print("\nInitializing Phase 3.5: FULL AUTOMATION ENGINE...")
    
    live_context = get_live_game_context()
    if not live_context:
        print("No live games found. Using mocked schedule.")
        live_context = {
            'Los Angeles Lakers': {'spread': -5.5, 'blowout_risk': False, 'opponent': 'Denver Nuggets'},
            'Boston Celtics': {'spread': -14.5, 'blowout_risk': True, 'opponent': 'Washington Wizards'}
        }
        
    injuries = load_injuries()
    
    # Load Models
    try:
        model_pts = joblib.load('models/xgb_points_model.joblib')
        model_reb = joblib.load('models/xgb_rebounds_model.joblib')
        model_ast = joblib.load('models/xgb_assists_model.joblib')
    except Exception as e:
        print("Models not found. Ensure train_model.py finished training P/R/A.")
        return

    print("\n========================================================")
    print("      LIVE MULTI-STAT CHEAT SHEET (PRA)                 ")
    print("========================================================\n")
    
    # We will simulate the top players for the games found in live_context
    # Since we don't have a real-time roster API, we'll use predefined star features to demonstrate the logic.
    
    players_to_simulate = [
        {
            'name': 'Jayson Tatum', 'team': 'Boston Celtics',
            'features': {
                'pts_ema_3': 28.5, 'pts_ema_7': 27.2, 'pts_ema_15': 26.5,
                'reb_ema_7': 8.1, 'ast_ema_7': 4.5, 'min_ema_7': 36.0,
                'pts_volatility_10': 5.5, 'days_rest': 2,
                'pts_career_baseline': 26.9, 'expected_minutes_ratio': 1.0,
                'opp_def_rating_10': 118.0,
                'pos_def_rating_10': 119.5,
                'opp_pace_10': 102.5,
                'opp_fta_rate_10': 22.1,
                'referee_whistle_modifier': 0.12,
                'miles_traveled_since_last_game': 150.0,
                'time_zones_crossed': 0.0,
                'spatial_matchup_rating': 1.15
            }
        },
        {
            'name': 'Anthony Davis', 'team': 'Los Angeles Lakers',
            'features': {
                'pts_ema_3': 24.5, 'pts_ema_7': 25.1, 'pts_ema_15': 24.8,
                'reb_ema_7': 12.1, 'ast_ema_7': 3.5, 'min_ema_7': 35.0,
                'pts_volatility_10': 4.8, 'days_rest': 1,
                'pts_career_baseline': 24.0, 'expected_minutes_ratio': 0.97,
                'opp_def_rating_10': 110.0,
                'pos_def_rating_10': 108.5,
                'opp_pace_10': 98.5,
                'opp_fta_rate_10': 28.5,
                'referee_whistle_modifier': -0.05,
                'miles_traveled_since_last_game': 2500.0,
                'time_zones_crossed': 3.0,
                'spatial_matchup_rating': 0.95
            }
        }
    ]
    
    for p in players_to_simulate:
        name = p['name']
        team = p['team']
        features = p['features']
        
        # 1. Check for Blowout Penalty from Live Schedule
        context = live_context.get(team, {})
        blowout_msg = ""
        if context.get('blowout_risk'):
            features['min_ema_7'] *= 0.85
            features['expected_minutes_ratio'] = features['min_ema_7'] / 36.0
            blowout_msg = f" | ⚠️ BLOWOUT PENALTY: Spread {context.get('spread')}"
            
        # 2. Check for Injury Usage Shifts
        usage_msg = ""
        for inj in injuries:
            if inj['team'] == team and inj['status'] == 'OUT' and name in inj['usage_beneficiaries']:
                features['pts_ema_3'] *= 1.15
                features['pts_ema_7'] *= 1.15
                features['reb_ema_7'] *= 1.10
                features['ast_ema_7'] *= 1.20 # Huge boost if PG is out
                usage_msg = f" | ⬆️ USAGE SHIFT: {inj['player']} is OUT"
                
        # 3. Simulate PRA
        mean_pts, _ = run_monte_carlo_simulation(model_pts, features, variance=5.5)
        mean_reb, _ = run_monte_carlo_simulation(model_reb, features, variance=2.5)
        mean_ast, _ = run_monte_carlo_simulation(model_ast, features, variance=2.0)
        
        total_pra = mean_pts + mean_reb + mean_ast
        
        print(f"[{name}] - {team}{blowout_msg}{usage_msg}")
        print(f"  Projected Points:   {mean_pts:.1f}")
        print(f"  Projected Rebounds: {mean_reb:.1f}")
        print(f"  Projected Assists:  {mean_ast:.1f}")
        print(f"  Projected PRA:      {total_pra:.1f}\n")

if __name__ == "__main__":
    generate_automated_cheat_sheet()
