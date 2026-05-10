import time
from nba_api.stats.endpoints import leaguegamelog
from database import SessionLocal
import schema
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert

def fetch_and_store_historical_data(seasons=["2023-24", "2024-25", "2025-26"]):
    # Pull both Regular Season and Playoffs to capture full season data
    season_types = ["Regular Season", "Playoffs"]
    
    for season in seasons:
      for season_type in season_types:
        label = f"{season} ({season_type})"
        print(f"Fetching game logs for {label}...")
        try:
            # Fetch player game logs for the entire league
            logs = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation='P',
                season_type_all_star=season_type
            )
            df = logs.get_data_frames()[0]
            if df.empty:
                print(f"  No data for {label}. Skipping.")
                time.sleep(1)
                continue
            print(f"Successfully fetched {len(df)} game logs from nba_api for {label}.")
            
            db: Session = SessionLocal()
            
            # We need to extract unique players and games first
            unique_players = df[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates()
            unique_games = df[['GAME_ID', 'GAME_DATE']].drop_duplicates()
            
            print(f"Upserting Players for {label}...")
            for _, row in unique_players.iterrows():
                player_id = int(row['PLAYER_ID'])
                player = db.query(schema.Player).filter(schema.Player.id == player_id).first()
                if not player:
                    new_player = schema.Player(id=player_id, full_name=row['PLAYER_NAME'], is_active=True)
                    db.add(new_player)
                    
            print(f"Upserting Games for {label}...")
            for _, row in unique_games.iterrows():
                game_id = str(row['GAME_ID'])
                game = db.query(schema.Game).filter(schema.Game.id == game_id).first()
                if not game:
                    from datetime import datetime
                    g_date = datetime.strptime(row['GAME_DATE'], '%Y-%m-%d').date()
                    new_game = schema.Game(id=game_id, game_date=g_date)
                    db.add(new_game)
                    
            db.commit()
            
            print(f"Upserting Player Game Logs for {label}...")
            for _, row in df.iterrows():
                player_id = int(row['PLAYER_ID'])
                game_id = str(row['GAME_ID'])
                
                existing = db.query(schema.PlayerGameLog).filter_by(player_id=player_id, game_id=game_id).first()
                if not existing:
                    log = schema.PlayerGameLog(
                        player_id=player_id,
                        game_id=game_id,
                        minutes=float(row['MIN']),
                        points=int(row['PTS']),
                        rebounds=int(row['REB']),
                        assists=int(row['AST']),
                        threes_made=int(row['FG3M'])
                    )
                    db.add(log)
                    
            db.commit()
            db.close()
            print(f"ETL for {label} Complete.\n")
            
            # Sleep to prevent rate limiting from stats.nba.com
            time.sleep(2)
            
        except Exception as e:
            print(f"Error in ETL pipeline for {label}: {e}")

if __name__ == "__main__":
    fetch_and_store_historical_data(["2023-24", "2024-25", "2025-26"])
