import os
import requests
from database import SessionLocal
import schema

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

def fetch_live_player_props():
    """
    Scaffold function to fetch live player props from The Odds API.
    """
    if not ODDS_API_KEY:
        print("WARNING: ODDS_API_KEY not found in environment.")
        return None
        
    print("Fetching live player props...")
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "player_points,player_rebounds,player_assists",
        "oddsFormat": "american"
    }
    
    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        print(f"Fetched {len(data)} games with odds.")
        return data
    except Exception as e:
        print(f"Error fetching odds data: {e}")
        return None

def save_odds_to_db(odds_data):
    """
    Scaffold function to save live lines to schema.LivePropLine
    """
    db = SessionLocal()
    # Map json data to LivePropLine objects
    print("Saving live odds to database...")
    db.close()

if __name__ == "__main__":
    print("--- Live Odds Ingestion Pipeline ---")
    data = fetch_live_player_props()
    if data:
        save_odds_to_db(data)
