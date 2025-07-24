import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df_2025 = pd.read_csv('./raw_data/NBA_2025_Shots.csv')
df_2024 = pd.read_csv('./raw_data/NBA_2024_Shots.csv')
df_2023 = pd.read_csv('./raw_data/NBA_2023_Shots.csv')

# Combine and preprocess
df_combined = pd.concat([df_2025, df_2024, df_2023])
df_combined = df_combined.drop(columns=['SEASON_2', 'GAME_ID', 'ZONE_ABB', 'EVENT_TYPE', 'GAME_DATE',
                                      'PLAYER_ID', 'TEAM_ID', 'TEAM_NAME'])

# Rebuild model with combined data
X = df_combined.drop(columns=['SHOT_MADE', 'PLAYER_NAME'])
y = df_combined['SHOT_MADE'].astype(int)
X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)  # Increased trees
model.fit(X_train, y_train)

joblib.dump({
    'model': model,
    'feature_names': X_train.columns
}, './models/random_forest_23_25.joblib')

importances = model.feature_importances_
important_features = X_train.columns[np.argsort(importances)[::-1][:10]]
print("Top features by importance:", list(important_features))

y_pred = model.predict_proba(X_test)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, y_pred))
