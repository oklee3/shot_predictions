import pandas as pd
from tqdm import tqdm
import time
from nba_api.stats.endpoints import PlayerDashboardByYearOverYear

def get_player_stats(player_id, season='2024-25'):
    try:
        dash = PlayerDashboardByYearOverYear(player_id=player_id, season=season)
        df = dash.get_data_frames()[1]
        latest_season = df[df['GROUP_VALUE'] == season]
        stats = latest_season[['FG_PCT', 'FG3_PCT', 'MIN']].copy()
        stats['PLAYER_ID'] = player_id
        return stats
    except Exception as e:
        print(f"Error for player_id {player_id}: {e}")
        return None

original_df = pd.read_csv("./raw_data/NBA_2025_Shots.csv")
unique_ids = original_df['PLAYER_ID'].unique()

all_stats = []
for pid in tqdm(unique_ids, desc="Fetching Player Stats"):
    stats_df = get_player_stats(pid)
    if stats_df is not None:
        all_stats.append(stats_df)
        time.sleep(0.5)

if all_stats:
    stats_combined = pd.concat(all_stats, ignore_index=True)
    merged_df = original_df.merge(stats_combined, on='PLAYER_ID', how='left')
    merged_df.to_csv("./merged_data/24_25_allstats.csv", index=False)
    print("Saved merged stats to '24_25_allstats.csv'")
else:
    print("No stats retrieved.")
