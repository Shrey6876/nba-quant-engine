import time
from nba_api.stats.endpoints import shotchartdetail, boxscoreofficialsv3
import pandas as pd

def research_shot_charts():
    print("--- Researching Shot Charts ---")
    try:
        # Pull shot chart for a specific player (e.g., LeBron James: 2544) for a specific season
        # team_id = 0 for all teams, context_measure_simple = 'FGA'
        sc = shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=2544,
            season_nullable='2023-24',
            context_measure_simple='FGA'
        )
        df = sc.get_data_frames()[0]
        print("Shot Chart Columns:", df.columns.tolist())
        print("Sample Data:\n", df[['SHOT_TYPE', 'SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_DISTANCE']].head(3))
        print(f"Total shots pulled: {len(df)}")
    except Exception as e:
        print("Error pulling shot charts:", e)

def research_officials():
    print("\n--- Researching Boxscore Officials ---")
    try:
        # Provide a valid Game ID (e.g., Game 1 of the 2023-24 season)
        game_id = '0022300061' # Random valid game ID from 23-24
        officials = boxscoreofficialsv3.BoxScoreOfficialsV3(game_id=game_id)
        df = officials.get_data_frames()[0]
        print("Officials Columns:", df.columns.tolist())
        print("Sample Data:\n", df.head())
    except Exception as e:
        print("Error pulling officials:", e)

if __name__ == "__main__":
    research_shot_charts()
    time.sleep(1)
    research_officials()
