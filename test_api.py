from nba_api.stats.endpoints import commonplayerinfo, leaguedashteamstats
import pandas as pd
import time

def test_player_position():
    try:
        # Test pulling Lebron's position
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=2544)
        df = player_info.get_data_frames()[0]
        print("Player Info Columns:", df.columns.tolist())
        print("Position:", df['POSITION'].iloc[0])
    except Exception as e:
        print("Error pulling player info:", e)

def test_team_pace():
    try:
        # Test pulling team pace
        stats = leaguedashteamstats.LeagueDashTeamStats(season='2023-24', measure_type_detailed_defense='Advanced')
        df = stats.get_data_frames()[0]
        print("Team Stats Columns:", df.columns.tolist())
        print("Pace for top team:", df[['TEAM_NAME', 'PACE']].head(1))
    except Exception as e:
        print("Error pulling team stats:", e)

if __name__ == "__main__":
    test_player_position()
    time.sleep(1)
    test_team_pace()
