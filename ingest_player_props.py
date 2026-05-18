#!/usr/bin/env python3
"""
ingest_player_props.py
──────────────────────
Fetches real player prop lines from The Odds API for upcoming NBA games.
Stores lines from multiple bookmakers with implied & no-vig fair probabilities.

Usage:
    python ingest_player_props.py
"""

import os
import datetime
import requests
import pandas as pd
from sqlalchemy import create_engine
from database import SessionLocal
import schema
from dotenv import load_dotenv

load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_points_rebounds_assists",
    "player_threes",
]

# Preferred books in priority order for "best line"
PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars", "bovada"]


def american_to_implied(odds: int) -> float:
    """Convert American odds to implied probability (0-1)."""
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    return 100 / (odds + 100)


def devig(over_odds: int, under_odds: int) -> tuple:
    """Remove vig to get fair (no-vig) probabilities."""
    p_over = american_to_implied(over_odds)
    p_under = american_to_implied(under_odds)
    total = p_over + p_under
    if total == 0:
        return 0.5, 0.5
    return p_over / total, p_under / total


def fetch_events() -> list:
    """Fetch upcoming NBA events (games) from The Odds API."""
    if not ODDS_API_KEY:
        print("  ⚠️  ODDS_API_KEY not set.")
        return []

    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events?apiKey={ODDS_API_KEY}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        events = resp.json()
        print(f"  📅 Found {len(events)} upcoming events")
        return events
    except Exception as e:
        print(f"  ⚠️  Error fetching events: {e}")
        return []


def fetch_player_props(event_id: str, markets: list) -> dict:
    """Fetch player prop odds for a specific event."""
    markets_str = ",".join(markets)
    url = (
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
        f"?apiKey={ODDS_API_KEY}&regions=us&markets={markets_str}&oddsFormat=american"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    ⚠️  Error fetching props for event {event_id}: {e}")
        return {}


def parse_and_store_props(event_data: dict, game_date):
    """Parse prop data from The Odds API and store in the database."""
    db = SessionLocal()
    home_team = event_data.get("home_team", "")
    away_team = event_data.get("away_team", "")
    event_id = event_data.get("id", "")
    
    props_stored = 0
    
    for bookmaker in event_data.get("bookmakers", []):
        book_key = bookmaker.get("key", "")
        
        for market in bookmaker.get("markets", []):
            market_key = market.get("key", "")  # e.g. "player_points"
            outcomes = market.get("outcomes", [])
            
            # Group outcomes by player description (the line/point value pairs)
            # Each player has an "Over" and "Under" outcome
            player_outcomes = {}
            for outcome in outcomes:
                player_name = outcome.get("description", "")
                direction = outcome.get("name", "")   # "Over" or "Under"
                odds = outcome.get("price", 0)
                point = outcome.get("point", 0)
                
                if player_name not in player_outcomes:
                    player_outcomes[player_name] = {"line": point}
                
                if direction == "Over":
                    player_outcomes[player_name]["over_odds"] = odds
                elif direction == "Under":
                    player_outcomes[player_name]["under_odds"] = odds
            
            # Store each player's prop line
            for player_name, data in player_outcomes.items():
                over_odds = data.get("over_odds")
                under_odds = data.get("under_odds")
                line = data.get("line", 0)
                
                if over_odds is None or under_odds is None:
                    continue
                
                implied_over = american_to_implied(over_odds)
                implied_under = american_to_implied(under_odds)
                fair_over, fair_under = devig(over_odds, under_odds)
                
                prop_line = schema.LivePropLine(
                    player_name=player_name,
                    game_date=game_date,
                    stat_type=market_key,
                    line=line,
                    over_odds=over_odds,
                    under_odds=under_odds,
                    implied_prob_over=round(implied_over, 4),
                    implied_prob_under=round(implied_under, 4),
                    fair_prob_over=round(fair_over, 4),
                    fair_prob_under=round(fair_under, 4),
                    book_name=book_key,
                    event_id=event_id,
                    home_team=home_team,
                    away_team=away_team,
                )
                db.add(prop_line)
                props_stored += 1
    
    db.commit()
    db.close()
    return props_stored


def get_best_lines() -> pd.DataFrame:
    """
    Query stored props and return the best available line per player per stat.
    'Best' = highest fair probability edge (lowest vig) from preferred books.
    """
    today = datetime.date.today().isoformat()
    tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
    
    query = f"""
        SELECT player_name, stat_type, line, over_odds, under_odds,
               implied_prob_over, implied_prob_under,
               fair_prob_over, fair_prob_under,
               book_name, home_team, away_team
        FROM live_prop_lines
        WHERE game_date >= '{today}' AND game_date <= '{tomorrow}'
        ORDER BY player_name, stat_type, book_name
    """
    df = pd.read_sql(query, engine)
    
    if df.empty:
        return df
    
    # For each player × stat, pick the best over odds (most favorable to bettor)
    best_lines = []
    for (player, stat), group in df.groupby(['player_name', 'stat_type']):
        # Best over = highest over odds (least negative or most positive)
        best_over_row = group.loc[group['over_odds'].idxmax()]
        # Best under = highest under odds
        best_under_row = group.loc[group['under_odds'].idxmax()]
        
        # Use the consensus line (most common line value)
        consensus_line = group['line'].mode().iloc[0] if not group['line'].mode().empty else group['line'].iloc[0]
        
        best_lines.append({
            'player_name': player,
            'stat_type': stat,
            'consensus_line': consensus_line,
            'best_over_odds': int(best_over_row['over_odds']),
            'best_over_book': best_over_row['book_name'],
            'best_under_odds': int(best_under_row['under_odds']),
            'best_under_book': best_under_row['book_name'],
            'fair_prob_over': best_over_row['fair_prob_over'],
            'fair_prob_under': best_over_row['fair_prob_under'],
            'home_team': best_over_row['home_team'],
            'away_team': best_over_row['away_team'],
            'num_books': len(group['book_name'].unique()),
        })
    
    return pd.DataFrame(best_lines)


def main():
    print("=" * 70)
    print("  📊 PLAYER PROP LINE INGESTION")
    print("=" * 70)
    
    if not ODDS_API_KEY:
        print("\n  ❌ ODDS_API_KEY not found. Set it in your .env file.")
        return
    
    # Step 1: Fetch events
    print("\n  Step 1: Fetching upcoming NBA events...")
    events = fetch_events()
    
    if not events:
        print("  No upcoming events found.")
        return
    
    # Step 2: Clear old prop lines for today/tomorrow
    db = SessionLocal()
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    db.query(schema.LivePropLine).filter(
        schema.LivePropLine.game_date >= today,
        schema.LivePropLine.game_date <= tomorrow
    ).delete()
    db.commit()
    db.close()
    print(f"  Cleared old prop lines for {today} - {tomorrow}")
    
    # Step 3: Fetch & store props for each event
    total_props = 0
    for event in events:
        event_id = event.get("id", "")
        home = event.get("home_team", "")
        away = event.get("away_team", "")
        commence = event.get("commence_time", "")
        
        # Parse game date from commence_time
        try:
            game_date = datetime.datetime.fromisoformat(
                commence.replace("Z", "+00:00")
            ).date()
        except:
            game_date = tomorrow
        
        print(f"\n  📥 {away} @ {home} ({game_date})")
        
        # Fetch props
        prop_data = fetch_player_props(event_id, PROP_MARKETS)
        if not prop_data:
            continue
        
        count = parse_and_store_props(prop_data, game_date)
        total_props += count
        print(f"    Stored {count} prop lines")
        
        import time
        time.sleep(0.5)
    
    # Step 4: Show summary
    print(f"\n{'=' * 70}")
    print(f"  ✅ Ingestion complete: {total_props} total prop lines stored")
    print(f"{'=' * 70}")
    
    # Step 5: Show best lines summary
    best = get_best_lines()
    if not best.empty:
        pts_lines = best[best['stat_type'] == 'player_points'].head(10)
        if not pts_lines.empty:
            print(f"\n  📋 Sample PTS lines (best available):")
            print(f"  {'PLAYER':<25} {'LINE':>5} {'OVER':>6} {'BOOK':<12} {'UNDER':>6} {'BOOK':<12}")
            print(f"  {'─' * 70}")
            for _, r in pts_lines.iterrows():
                print(
                    f"  {r['player_name']:<25}"
                    f" {r['consensus_line']:5.1f}"
                    f" {r['best_over_odds']:>+6d}"
                    f" {r['best_over_book']:<12}"
                    f" {r['best_under_odds']:>+6d}"
                    f" {r['best_under_book']:<12}"
                )


if __name__ == "__main__":
    main()
