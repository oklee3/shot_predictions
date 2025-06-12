import pandas as pd
import numpy as np
import joblib

# Load model and metadata
# this model is trained on 24-25 season data only
model_data = joblib.load('./models/random_forest_24_25.joblib')
model = model_data['model']
trained_features = model_data['feature_names']

# select stats to test on
# currently using model trained on 2025 season to predict 2024 season shots
df = pd.read_csv('./merged_data/23_24_allstats.csv')
df = df.drop(columns=['SEASON_2', 'GAME_ID', 'ZONE_ABB', 'EVENT_TYPE', 'GAME_DATE',
                     'PLAYER_ID', 'TEAM_ID', 'TEAM_NAME'])

player_name = "Dyson Daniels"

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

# find examples that were incorrectly classified
mismatches = player_rows[player_rows['SHOT_MADE'] != player_rows['PREDICTED_MADE']]

print("\nMismatched Predictions (SHOT_MADE != PREDICTED_MADE):")
print(mismatches[[
    'PLAYER_NAME', 
    'FG_PCT', 
    'FG3_PCT', 
    'ACTION_TYPE', 
    'SHOT_TYPE', 
    'SHOT_DISTANCE', 
    'ZONE_NAME',
    'SHOT_MADE', 
    'PREDICTED_MADE', 
    'PREDICTED_PROB'
]].to_string())