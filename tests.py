import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and metadata
# this model is trained on 24-25 season data only
model_data = joblib.load('./models/random_forest_24_25.joblib')
model = model_data['model']
trained_features = model_data['feature_names']

# select stats to test on
# currently using model trained on 2025 season to predict 2024 season shots
df = pd.read_csv('./raw_data/NBA_2024_Shots.csv')
df = df.drop(columns=['SEASON_2', 'GAME_ID', 'ZONE_ABB', 'EVENT_TYPE', 'GAME_DATE',
                     'PLAYER_ID', 'TEAM_ID', 'TEAM_NAME'])

# predicted vs expected

X = df.drop(columns=['SHOT_MADE', 'PLAYER_NAME'])
X_encoded = pd.get_dummies(X)
X_encoded = X_encoded.reindex(columns=trained_features, fill_value=0)

df['PREDICTED_PROB'] = model.predict_proba(X_encoded)[:, 1]
df['PREDICTED_MADE'] = model.predict(X_encoded)

point_map = {
    '2PT Field Goal': 2,
    '3PT Field Goal': 3
}

df['SHOT_VALUE'] = df['SHOT_TYPE'].map(point_map)
df['ACTUAL_POINTS'] = df['SHOT_MADE'] * df['SHOT_VALUE']
df['EXPECTED_POINTS'] = df['PREDICTED_PROB'] * df['SHOT_VALUE']

player_summary = df.groupby('PLAYER_NAME').agg(
    total_shots=('SHOT_MADE', 'count'),
    actual_points=('ACTUAL_POINTS', 'sum'),
    expected_points=('EXPECTED_POINTS', 'sum'),
)

player_summary['points_above_expected'] = player_summary['actual_points'] - player_summary['expected_points']

# plot graph

plt.figure(figsize=(10, 7))
plt.scatter(
    player_summary['expected_points'],
    player_summary['points_above_expected'],
    alpha=0.7
)

plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Expected Points (sum of predicted probabilities)')
plt.ylabel('Points Above Expected (actual - expected)')
plt.title('Player Expected Points vs. Points Above Expected')

for player in player_summary.index:
    if abs(player_summary.loc[player, 'expected_points']) > 500 and abs(player_summary.loc[player, 'points_above_expected']) > 10:
        plt.text(
            player_summary.loc[player, 'expected_points'],
            player_summary.loc[player, 'points_above_expected'],
            player,
            fontsize=8,
            alpha=0.7
        )

plt.tight_layout()
plt.show()
