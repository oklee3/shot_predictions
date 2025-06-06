import pandas as pd
import numpy as np
import joblib

# Load model and metadata
model_data = joblib.load('./models/random_forest_24_25.joblib')
model = model_data['model']
trained_features = model_data['feature_names']

df = pd.read_csv('./merged_data/24_25_allstats.csv')
df = df.drop(columns=['SEASON_2', 'GAME_ID', 'ZONE_ABB', 'EVENT_TYPE', 'GAME_DATE',
                     'PLAYER_ID', 'TEAM_ID', 'TEAM_NAME'])

# choose player
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

print("\nPlayer Shot Predictions:")
print(player_rows[[
    'PLAYER_NAME', 
    'FG_PCT', 
    'FG3_PCT', 
    'ACTION_TYPE', 
    'SHOT_TYPE', 
    'SHOT_DISTANCE', 
    'LOC_Y', 
    'LOC_X', 
    'SHOT_MADE', 
    'PREDICTED_MADE', 
    'PREDICTED_PROB'
]].to_string())