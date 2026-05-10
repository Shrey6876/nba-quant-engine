import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine
import joblib
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

def load_feature_store():
    print("Loading feature matrix from database...")
    df = pd.read_sql("SELECT * FROM feature_store", engine)
    df['game_date'] = pd.to_datetime(df['game_date'])
    return df

def prepare_data(df, target_col='points'):
    # Define our feature columns (the Heavy Matrix)
    feature_cols = [
        'pts_ema_3', 'pts_ema_7', 'pts_ema_15',
        'reb_ema_7', 'ast_ema_7', 'min_ema_7',
        'pts_volatility_10', 'days_rest',
        'pts_career_baseline', 'expected_minutes_ratio',
        'opp_def_rating_10', 'pos_def_rating_10',
        'opp_pace_10', 'opp_fta_rate_10',
        'referee_whistle_modifier', 'miles_traveled_since_last_game',
        'time_zones_crossed', 'spatial_matchup_rating'
    ]
    
    # We drop any rows with NaN in features or target
    ml_df = df.dropna(subset=feature_cols + [target_col]).copy()
    
    # Chronological Split to prevent lookahead bias (Data Leakage)
    # Train: 2023-24 and 2024-25 seasons
    # Test: 2025-26 season
    # Note: For simplicity in this script, we split by date.
    split_date = pd.to_datetime('2025-09-01')
    
    train_df = ml_df[ml_df['game_date'] < split_date]
    test_df = ml_df[ml_df['game_date'] >= split_date]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Training Set: {len(X_train)} games")
    print(f"Testing Set: {len(X_test)} games")
    
    return X_train, y_train, X_test, y_test, feature_cols

def train_xgboost(X_train, y_train, X_test, y_test, target_col='points'):
    print(f"\nTraining XGBoost Regressor for {target_col}...")
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"Model Evaluation (Out-of-Sample Test Data):")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} {target_col}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} {target_col}")
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/xgb_{target_col}_model.joblib')
    print(f"Model saved to models/xgb_{target_col}_model.joblib")
    return model

def run_monte_carlo_simulation(model, features_dict, variance, n_simulations=10000):
    """
    Runs a Monte Carlo simulation based on the model's predicted mean and historical variance.
    """
    # 1. Get the predicted mean from XGBoost
    input_df = pd.DataFrame([features_dict])
    predicted_mean = model.predict(input_df)[0]
    
    # 2. Simulate outcomes drawing from a normal distribution
    # In reality, points are Poisson or Negative Binomial, but a bounded normal is a good proxy for high scorers
    simulations = np.random.normal(loc=predicted_mean, scale=variance, size=n_simulations)
    
    # Points cannot be negative
    simulations = np.maximum(simulations, 0)
    
    return predicted_mean, simulations

if __name__ == "__main__":
    df = load_feature_store()
    
    targets = ['points', 'rebounds', 'assists']
    models = {}
    
    for target in targets:
        print(f"\n======================================")
        print(f" TRAINING MODEL FOR: {target.upper()}")
        print(f"======================================")
        X_train, y_train, X_test, y_test, feature_cols = prepare_data(df, target_col=target)
        models[target] = train_xgboost(X_train, y_train, X_test, y_test, target_col=target)
    
    print("\n--- Example Simulation: Jokic/Luka Archetype ---")
    # Simulate a game for a high-volume scorer
    example_features = {
        'pts_ema_3': 32.5, 'pts_ema_7': 28.4, 'pts_ema_15': 27.1,
        'reb_ema_7': 11.2, 'ast_ema_7': 8.5, 'min_ema_7': 36.5,
        'pts_volatility_10': 6.2, 'days_rest': 2,
        'pts_career_baseline': 26.8, 'expected_minutes_ratio': 1.05,
        'opp_def_rating_10': 118.5, # Playing a terrible defense
        'pos_def_rating_10': 120.5, # Terrible against this position
        'opp_pace_10': 105.2, # Very fast pace
        'opp_fta_rate_10': 20.1, # Low foul risk
        'referee_whistle_modifier': 0.12, # High foul calling ref
        'miles_traveled_since_last_game': 150.0, # Short flight
        'time_zones_crossed': 0.0,
        'spatial_matchup_rating': 1.15 # Excellent shot chart matchup
    }
    
    # Test points model
    mean_proj_pts, sim_results_pts = run_monte_carlo_simulation(models['points'], example_features, variance=6.2)
    print(f"XGBoost Predicted Mean Points: {mean_proj_pts:.2f}")
    prob_over_pts = np.mean(sim_results_pts > 28.5) * 100
    print(f"Monte Carlo Probability of Scoring OVER 28.5 points: {prob_over_pts:.1f}%")
    
    # Test rebounds model
    mean_proj_reb, sim_results_reb = run_monte_carlo_simulation(models['rebounds'], example_features, variance=2.5)
    print(f"XGBoost Predicted Mean Rebounds: {mean_proj_reb:.2f}")
    prob_over_reb = np.mean(sim_results_reb > 10.5) * 100
    print(f"Monte Carlo Probability of Rebounding OVER 10.5: {prob_over_reb:.1f}%")
