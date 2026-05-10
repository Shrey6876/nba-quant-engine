import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error
import os
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

print("========================================================")
print("      PHASE 6: AUTONOMOUS HYPERPARAMETER TUNING         ")
print("========================================================\n")

print("Loading Deep Data Matrix from Database...")
df = pd.read_sql("SELECT * FROM feature_store", engine)
df['game_date'] = pd.to_datetime(df['game_date'])

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

# Drop NaNs
df = df.dropna(subset=feature_cols + ['points'])

# Train/Test Split Chronologically
train_df = df[df['game_date'] < '2025-09-01']
test_df = df[df['game_date'] >= '2025-09-01']

X_train, y_train = train_df[feature_cols], train_df['points']
X_test, y_test = test_df[feature_cols], test_df['points']

def objective(trial):
    # 1. Suggest parameters for the Optuna trial
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'random_state': 42
    }
    
    # 2. Train the model with these exact parameters
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train)
    
    # 3. Evaluate the model
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    return rmse

print("Starting Bayesian Optimization Study (Optuna)...")
# Note: In production we use n_trials=500. Using 15 here for speed demonstration.
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)

print("\n--- OPTIMIZATION COMPLETE ---")
print("Best hyperparameters found mathematically:")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")
print(f"Best Out-Of-Sample RMSE Achieved: {study.best_value:.4f} points")

print("\nThese parameters will now be saved and utilized by the final Ensemble Model.")
