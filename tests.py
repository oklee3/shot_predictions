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
df = pd.read_csv('./merged_data/23_24_allstats.csv')
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

# Create probability range indicators
df['PROB_0_25'] = df['PREDICTED_PROB'] < 0.25
df['PROB_25_50'] = (df['PREDICTED_PROB'] >= 0.25) & (df['PREDICTED_PROB'] < 0.5)
df['PROB_50_75'] = (df['PREDICTED_PROB'] >= 0.5) & (df['PREDICTED_PROB'] < 0.75)
df['PROB_75_100'] = df['PREDICTED_PROB'] >= 0.75

# Create more granular probability ranges for detailed analysis
df['PROB_0_10'] = df['PREDICTED_PROB'] < 0.1
df['PROB_10_20'] = (df['PREDICTED_PROB'] >= 0.1) & (df['PREDICTED_PROB'] < 0.2)
df['PROB_20_30'] = (df['PREDICTED_PROB'] >= 0.2) & (df['PREDICTED_PROB'] < 0.3)
df['PROB_30_40'] = (df['PREDICTED_PROB'] >= 0.3) & (df['PREDICTED_PROB'] < 0.4)
df['PROB_40_50'] = (df['PREDICTED_PROB'] >= 0.4) & (df['PREDICTED_PROB'] < 0.5)
df['PROB_50_60'] = (df['PREDICTED_PROB'] >= 0.5) & (df['PREDICTED_PROB'] < 0.6)
df['PROB_60_70'] = (df['PREDICTED_PROB'] >= 0.6) & (df['PREDICTED_PROB'] < 0.7)
df['PROB_70_80'] = (df['PREDICTED_PROB'] >= 0.7) & (df['PREDICTED_PROB'] < 0.8)
df['PROB_80_90'] = (df['PREDICTED_PROB'] >= 0.8) & (df['PREDICTED_PROB'] < 0.9)
df['PROB_90_100'] = df['PREDICTED_PROB'] >= 0.9

player_summary = df.groupby('PLAYER_NAME').agg(
    total_shots=('SHOT_MADE', 'count'),
    actual_points=('ACTUAL_POINTS', 'sum'),
    expected_points=('EXPECTED_POINTS', 'sum'),
    
    # Basic probability range proportions
    prob_0_25=('PROB_0_25', 'mean'),
    prob_25_50=('PROB_25_50', 'mean'),
    prob_50_75=('PROB_50_75', 'mean'),
    prob_75_100=('PROB_75_100', 'mean'),
    
    # Detailed probability range proportions
    prob_0_10=('PROB_0_10', 'mean'),
    prob_10_20=('PROB_10_20', 'mean'),
    prob_20_30=('PROB_20_30', 'mean'),
    prob_30_40=('PROB_30_40', 'mean'),
    prob_40_50=('PROB_40_50', 'mean'),
    prob_50_60=('PROB_50_60', 'mean'),
    prob_60_70=('PROB_60_70', 'mean'),
    prob_70_80=('PROB_70_80', 'mean'),
    prob_80_90=('PROB_80_90', 'mean'),
    prob_90_100=('PROB_90_100', 'mean'),
    
    # Additional metrics
    avg_predicted_prob=('PREDICTED_PROB', 'mean'),
    std_predicted_prob=('PREDICTED_PROB', 'std'),
    min_predicted_prob=('PREDICTED_PROB', 'min'),
    max_predicted_prob=('PREDICTED_PROB', 'max'),
    
    # Shot difficulty metrics
    high_difficulty_shots=('PROB_0_25', 'sum'),  # Count of shots with <25% predicted probability
    low_difficulty_shots=('PROB_75_100', 'sum'),  # Count of shots with >75% predicted probability
    medium_difficulty_shots=('PROB_25_50', 'sum'),  # Count of shots with 25-50% predicted probability
)

player_summary['points_above_expected'] = player_summary['actual_points'] - player_summary['expected_points']

# Calculate shot difficulty ratios
player_summary['high_difficulty_ratio'] = player_summary['high_difficulty_shots'] / player_summary['total_shots']
player_summary['low_difficulty_ratio'] = player_summary['low_difficulty_shots'] / player_summary['total_shots']
player_summary['medium_difficulty_ratio'] = player_summary['medium_difficulty_shots'] / player_summary['total_shots']

# Calculate shot selection efficiency (how well they perform vs expected)
player_summary['shot_efficiency'] = player_summary['points_above_expected'] / player_summary['total_shots']

# Display summary statistics
print("=== PREDICTION PROBABILITY DISTRIBUTION METRICS ===")
print(f"Total players analyzed: {len(player_summary)}")
print(f"Average predicted probability across all shots: {df['PREDICTED_PROB'].mean():.3f}")
print(f"Standard deviation of predicted probabilities: {df['PREDICTED_PROB'].std():.3f}")

# Show players with interesting probability distributions
print("\n=== PLAYERS WITH HIGH DIFFICULTY SHOT SELECTION (>30% low probability shots) ===")
high_difficulty_players = player_summary[player_summary['prob_0_25'] > 0.3].sort_values('prob_0_25', ascending=False)
print(high_difficulty_players[['total_shots', 'prob_0_25', 'prob_25_50', 'prob_50_75', 'prob_75_100', 'shot_efficiency']].head(10))

print("\n=== PLAYERS WITH CONSERVATIVE SHOT SELECTION (>50% high probability shots) ===")
conservative_players = player_summary[player_summary['prob_75_100'] > 0.5].sort_values('prob_75_100', ascending=False)
print(conservative_players[['total_shots', 'prob_0_25', 'prob_25_50', 'prob_50_75', 'prob_75_100', 'shot_efficiency']].head(10))

print("\n=== PLAYERS WITH BALANCED SHOT SELECTION (25-75% in medium ranges) ===")
balanced_players = player_summary[
    (player_summary['prob_25_50'] + player_summary['prob_50_75'] > 0.5) & 
    (player_summary['total_shots'] > 100)
].sort_values('shot_efficiency', ascending=False)
print(balanced_players[['total_shots', 'prob_0_25', 'prob_25_50', 'prob_50_75', 'prob_75_100', 'shot_efficiency']].head(10))

# Create visualizations
plt.figure(figsize=(15, 10))

# Plot 1: Probability distribution for top players
plt.subplot(2, 2, 1)
top_players = player_summary.nlargest(10, 'total_shots')
prob_cols = ['prob_0_25', 'prob_25_50', 'prob_50_75', 'prob_75_100']
top_players[prob_cols].plot(kind='bar', stacked=True, ax=plt.gca())
plt.title('Shot Probability Distribution - Top 10 Players by Volume')
plt.ylabel('Proportion of Shots')
plt.xticks(rotation=45)
plt.legend(['0-25%', '25-50%', '50-75%', '75-100%'])

# Plot 2: Shot efficiency vs difficulty
plt.subplot(2, 2, 2)
plt.scatter(player_summary['high_difficulty_ratio'], player_summary['shot_efficiency'], alpha=0.6)
plt.xlabel('Proportion of High Difficulty Shots (<25% predicted)')
plt.ylabel('Shot Efficiency (Points Above Expected per Shot)')
plt.title('Shot Difficulty vs Efficiency')

# Plot 3: Average predicted probability distribution
plt.subplot(2, 2, 3)
plt.hist(player_summary['avg_predicted_prob'], bins=30, alpha=0.7, edgecolor='black')
plt.xlabel('Average Predicted Probability')
plt.ylabel('Number of Players')
plt.title('Distribution of Average Predicted Probabilities')

# Plot 4: Probability range comparison
plt.subplot(2, 2, 4)
prob_ranges = ['prob_0_25', 'prob_25_50', 'prob_50_75', 'prob_75_100']
avg_proportions = [player_summary[col].mean() for col in prob_ranges]
plt.bar(['0-25%', '25-50%', '50-75%', '75-100%'], avg_proportions)
plt.ylabel('Average Proportion of Shots')
plt.title('Average Shot Probability Distribution Across All Players')

plt.tight_layout()
plt.show()

# Save detailed results
player_summary.to_csv('./analysis_results/player_probability_analysis.csv')
print(f"\nDetailed results saved to './analysis_results/player_probability_analysis.csv'")
