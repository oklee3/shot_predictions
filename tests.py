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
df = pd.read_csv('./raw_data/NBA_2015_Shots.csv')
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

# by player

player_name = "LeBron James"

player_rows = df[df['PLAYER_NAME'] == player_name].copy()
player_y = player_rows['SHOT_MADE'].astype(int)

player_X = player_rows.drop(columns=['SHOT_MADE', 'PLAYER_NAME'])
player_X_encoded = pd.get_dummies(player_X)
player_X_encoded = player_X_encoded.reindex(columns=trained_features, fill_value=0)

predictions = model.predict(player_X_encoded)
probabilities = model.predict_proba(player_X_encoded)[:, 1]

player_rows = player_rows.assign(
    PREDICTED_MADE=predictions,
    PREDICTED_PROB=probabilities
)

correct = np.sum(player_y.values == predictions)
total = len(player_y)
print(f"Correct Predictions: {correct} / {total}")
print(f"Accuracy for {player_name}: {correct / total}")
print(player_summary.loc[player_name])

# find examples that were incorrectly classified
mismatches = player_rows[player_rows['SHOT_MADE'] != player_rows['PREDICTED_MADE']]

print("\nMismatched Predictions (SHOT_MADE != PREDICTED_MADE):")
print(mismatches[[
    'PLAYER_NAME', 
    'ACTION_TYPE', 
    'SHOT_TYPE', 
    'SHOT_DISTANCE', 
    'ZONE_NAME',
    'SHOT_MADE', 
    'PREDICTED_MADE', 
    'PREDICTED_PROB'
]].head(5).to_string())