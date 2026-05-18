#!/usr/bin/env python3
"""
train_model.py
──────────────
Trains XGBoost regressors for PTS, REB, AST, and 3PM using the real feature matrix.
Updated with new features: USG%, FGA/FTA EMAs, home/away, real travel.
"""

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

# ── Canonical feature column list ────────────────────────────────────────
# Shared across train_model.py, backtester.py, ensemble_model.py, predict_tomorrow.py
FEATURE_COLS = [
    # Time-series dynamics
    'pts_ema_3', 'pts_ema_7', 'pts_ema_15',
    'reb_ema_7', 'ast_ema_7', 'min_ema_7',
    'pts_volatility_10', 'days_rest',
    'pts_career_baseline', 'expected_minutes_ratio',
    # Opponent context (REAL)
    'opp_def_rating_10', 'pos_def_rating_10',
    'opp_pace_10', 'opp_fta_rate_10',
    # Usage & efficiency (NEW)
    'usg_rate_10', 'usg_delta_5',
    'fga_ema_7', 'fta_ema_7', 'plus_minus_ema_7',
    # Contextual (REAL)
    'home_away',
    'miles_traveled_since_last_game', 'time_zones_crossed',
    # Additional stat features
    'threes_ema_7',
]

def load_feature_store():
    print("Loading feature matrix from database...")
    df = pd.read_sql("SELECT * FROM feature_store", engine)
    df['game_date'] = pd.to_datetime(df['game_date'])
    return df

def prepare_data(df, target_col='points'):
    # Use only columns that exist in the dataframe
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        print(f"  ⚠️  Missing features (will skip): {missing_cols}")
    
    ml_df = df.dropna(subset=available_cols + [target_col]).copy()
    
    # Chronological split
    split_date = pd.to_datetime('2025-09-01')
    train_df = ml_df[ml_df['game_date'] < split_date]
    test_df = ml_df[ml_df['game_date'] >= split_date]
    
    X_train = train_df[available_cols]
    y_train = train_df[target_col]
    X_test = test_df[available_cols]
    y_test = test_df[target_col]
    
    print(f"  Training Set: {len(X_train):,} games ({len(available_cols)} features)")
    print(f"  Testing Set:  {len(X_test):,} games")
    
    return X_train, y_train, X_test, y_test, available_cols

def train_xgboost(X_train, y_train, X_test, y_test, target_col='points'):
    print(f"\n  Training XGBoost Regressor for {target_col.upper()}...")
    
    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=2.0,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"  RMSE: {rmse:.2f} | MAE: {mae:.2f}")
    
    # Feature importance
    importances = model.feature_importances_
    feature_names = X_train.columns
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top features: {', '.join([f'{n} ({v:.3f})' for n, v in top_features])}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/xgb_{target_col}_model.joblib')
    print(f"  Saved to models/xgb_{target_col}_model.joblib")
    return model

if __name__ == "__main__":
    df = load_feature_store()
    
    targets = ['points', 'rebounds', 'assists', 'threes_made']
    models = {}
    
    for target in targets:
        print(f"\n{'=' * 60}")
        print(f"  TRAINING MODEL FOR: {target.upper()}")
        print(f"{'=' * 60}")
        
        if target not in df.columns:
            print(f"  ⚠️  Column '{target}' not in feature store. Skipping.")
            continue
            
        X_train, y_train, X_test, y_test, cols = prepare_data(df, target_col=target)
        if len(X_train) == 0:
            print(f"  ⚠️  No training data for {target}. Skipping.")
            continue
        models[target] = train_xgboost(X_train, y_train, X_test, y_test, target_col=target)
    
    print(f"\n{'=' * 60}")
    print(f"  ✅ All models trained with {len(FEATURE_COLS)} features")
    print(f"{'=' * 60}")
