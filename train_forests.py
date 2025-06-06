import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv('./merged_data/24_25_allstats.csv')
df = df.drop(columns=['SEASON_2', 'GAME_ID', 'ZONE_ABB', 'EVENT_TYPE', 'GAME_DATE',
                     'PLAYER_ID', 'TEAM_ID', 'TEAM_NAME'])

X = df.drop(columns=['SHOT_MADE', 'PLAYER_NAME'])
y = df['SHOT_MADE'].astype(int)
X_encoded = pd.get_dummies(X)
X_encoded['PLAYER_NAME'] = df['PLAYER_NAME']

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded.drop(columns=['PLAYER_NAME']),  # Remove PLAYER_ID for training
    y, 
    test_size=0.2, 
    stratify=y,
    random_state=42
)

test_player_ids = X_encoded.iloc[X_test.index]['PLAYER_NAME']

model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1)
model.fit(X_train, y_train)

joblib.dump({
    'model': model,
    'test_player_ids': test_player_ids,
    'feature_names': X_train.columns
}, './models/random_forest_24_25.joblib')

importances = model.feature_importances_
important_features = X_train.columns[np.argsort(importances)[::-1][:10]]
print("Top features by importance:", list(important_features))

y_pred = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
