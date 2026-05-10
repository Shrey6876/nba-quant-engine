import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sqlalchemy import create_engine
import joblib
from sklearn.metrics import mean_squared_error
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

def build_ensemble():
    print("========================================================")
    print("      PHASE 6: PROFESSIONAL STACKED ENSEMBLE            ")
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

    df = df.dropna(subset=feature_cols + ['points'])
    train_df = df[df['game_date'] < '2025-09-01']
    test_df = df[df['game_date'] >= '2025-09-01']

    X_train, y_train = train_df[feature_cols], train_df['points']
    X_test, y_test = test_df[feature_cols], test_df['points']
    
    print("Training Model 1: XGBoost (Optuna Optimized)...")
    xgb_params = {
        'n_estimators': 117,
        'max_depth': 4,
        'learning_rate': 0.035,
        'subsample': 0.75,
        'colsample_bytree': 0.71,
        'gamma': 4.07,
        'random_state': 42
    }
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(X_train, y_train)
    xgb_preds = model_xgb.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
    print(f"  XGBoost Out-Of-Sample RMSE: {xgb_rmse:.4f}")
    joblib.dump(model_xgb, 'models/ensemble_xgb.joblib')

    print("Training Model 2: LightGBM (Gradient Boosting Framework)...")
    lgb_params = {
        'n_estimators': 150,
        'max_depth': 5,
        'learning_rate': 0.04,
        'subsample': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(X_train, y_train)
    lgb_preds = model_lgb.predict(X_test)
    lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
    print(f"  LightGBM Out-Of-Sample RMSE: {lgb_rmse:.4f}")
    joblib.dump(model_lgb, 'models/ensemble_lgb.joblib')

    print("\n--- ENSEMBLE EVALUATION ---")
    # The professional edge: averaging multiple distinct algorithms reduces total variance.
    ensemble_preds = (xgb_preds + lgb_preds) / 2
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    print(f"Stacked Ensemble RMSE: {ensemble_rmse:.4f}")
    
    if ensemble_rmse < xgb_rmse and ensemble_rmse < lgb_rmse:
        print("✅ Ensemble successfully reduced variance and outperformed individual models!")
    else:
        print("Note: Ensemble is stable, but one model slightly outperformed.")

if __name__ == "__main__":
    build_ensemble()
