import pandas as pd
from tqdm import tqdm
import time
from nba_api.stats.endpoints import PlayerDashboardByYearOverYear

import pandas as pd
from tqdm import tqdm
import time
from nba_api.stats.endpoints import PlayerDashboardByYearOverYear

def get_player_stats(player_id, season='2024-25'):
    try:
        dash = PlayerDashboardByYearOverYear(player_id=player_id, season=season)
        df = dash.get_data_frames()[1]
        latest_season = df[df['GROUP_VALUE'] == season]

        if latest_season.empty:
            return None

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

    # Use weighted average by MIN to combine multi-team stints
    stats_combined['FG_PCT_x_MIN'] = stats_combined['FG_PCT'] * stats_combined['MIN']
    stats_combined['FG3_PCT_x_MIN'] = stats_combined['FG3_PCT'] * stats_combined['MIN']

    weighted = stats_combined.groupby('PLAYER_ID').agg({
        'FG_PCT_x_MIN': 'sum',
        'FG3_PCT_x_MIN': 'sum',
        'MIN': 'sum'
    }).reset_index()

    weighted['FG_PCT'] = weighted['FG_PCT_x_MIN'] / weighted['MIN']
    weighted['FG3_PCT'] = weighted['FG3_PCT_x_MIN'] / weighted['MIN']
    stats_aggregated = weighted[['PLAYER_ID', 'FG_PCT', 'FG3_PCT', 'MIN']]

    merged_df = original_df.merge(stats_aggregated, on='PLAYER_ID', how='left')
    merged_df.to_csv("./merged_data/24_25_allstats.csv", index=False)
    print("Saved merged stats to '24_25_allstats.csv'")
else:
    print("No stats retrieved. Check connection or player list.")
