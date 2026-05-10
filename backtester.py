import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import joblib
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error
import os
from dotenv import load_dotenv

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nba_quant.db")
engine = create_engine(DATABASE_URL)

def run_multi_season_backtest():
    print("========================================================")
    print("  PHASE 4 v2: MULTI-SEASON WALK-FORWARD BACKTESTER      ")
    print("========================================================\n")
    
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
    df = df[df['expected_minutes_ratio'] > 0.6]
    
    EDGE_THRESHOLD = 2.0
    
    # Walk-Forward Seasons:
    # Season 1: Train on 2023-24, Test on 2024-25
    # Season 2: Train on 2023-24 + 2024-25, Test on 2025-26
    seasons = [
        {
            'name': '2024-25 Season',
            'train_end': '2024-09-01',
            'test_start': '2024-09-01',
            'test_end': '2025-09-01'
        },
        {
            'name': '2025-26 Season',
            'train_end': '2025-09-01',
            'test_start': '2025-09-01',
            'test_end': '2026-09-01'
        }
    ]
    
    all_bets = []
    
    for season in seasons:
        print(f"\n{'='*56}")
        print(f"  BACKTESTING: {season['name']}")
        print(f"{'='*56}")
        
        train_df = df[df['game_date'] < season['train_end']]
        test_df = df[(df['game_date'] >= season['test_start']) & (df['game_date'] < season['test_end'])].copy()
        
        X_train, y_train = train_df[feature_cols], train_df['points']
        X_test, y_test = test_df[feature_cols], test_df['points']
        
        print(f"  Training Set: {len(train_df):,} games")
        print(f"  Test Set:     {len(test_df):,} games")
        
        # --- Train XGBoost (Optuna-Optimized) ---
        xgb_model = xgb.XGBRegressor(
            n_estimators=117, max_depth=4, learning_rate=0.035,
            subsample=0.75, colsample_bytree=0.71, gamma=4.07, random_state=42
        )
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
        
        # --- Train LightGBM ---
        lgb_model = lgb.LGBMRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.04,
            subsample=0.8, random_state=42, verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        lgb_preds = lgb_model.predict(X_test)
        lgb_rmse = np.sqrt(mean_squared_error(y_test, lgb_preds))
        
        # --- Stacked Ensemble ---
        ensemble_preds = (xgb_preds + lgb_preds) / 2
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
        
        print(f"\n  Model Performance (RMSE):")
        print(f"    XGBoost:  {xgb_rmse:.4f}")
        print(f"    LightGBM: {lgb_rmse:.4f}")
        print(f"    Ensemble: {ensemble_rmse:.4f}")
        
        test_df['model_projection'] = ensemble_preds
        
        # Mock Vegas Line
        test_df['mock_vegas_line'] = np.round(test_df['pts_ema_7'] * 2) / 2
        
        # Bet Logic
        test_df['bet_placed'] = 'NONE'
        test_df.loc[(test_df['model_projection'] - test_df['mock_vegas_line']) >= EDGE_THRESHOLD, 'bet_placed'] = 'OVER'
        test_df.loc[(test_df['mock_vegas_line'] - test_df['model_projection']) >= EDGE_THRESHOLD, 'bet_placed'] = 'UNDER'
        
        bets_df = test_df[test_df['bet_placed'] != 'NONE'].copy()
        bets_df['win'] = False
        bets_df.loc[(bets_df['bet_placed'] == 'OVER') & (bets_df['points'] > bets_df['mock_vegas_line']), 'win'] = True
        bets_df.loc[(bets_df['bet_placed'] == 'UNDER') & (bets_df['points'] < bets_df['mock_vegas_line']), 'win'] = True
        
        total = len(bets_df)
        wins = bets_df['win'].sum()
        losses = total - wins
        wr = (wins / total * 100) if total > 0 else 0
        roi = ((wins * 0.909) - losses) / total * 100 if total > 0 else 0
        
        print(f"\n  --- {season['name']} RESULTS ---")
        print(f"  Actionable Bets: {total}")
        print(f"  Wins: {wins}  |  Losses: {losses}")
        print(f"  Win Rate: {wr:.2f}%")
        print(f"  ROI: {roi:.2f}%")
        
        if wr > 52.38:
            print(f"  ✅ PROFITABLE")
        else:
            print(f"  ❌ UNPROFITABLE")
        
        all_bets.append(bets_df)
    
    # --- COMBINED RESULTS ---
    combined = pd.concat(all_bets)
    total = len(combined)
    wins = combined['win'].sum()
    losses = total - wins
    wr = (wins / total * 100) if total > 0 else 0
    roi = ((wins * 0.909) - losses) / total * 100 if total > 0 else 0
    
    print(f"\n{'='*56}")
    print(f"  COMBINED 2-SEASON BACKTEST RESULTS")
    print(f"{'='*56}")
    print(f"  Total Actionable Bets: {total:,}")
    print(f"  Wins: {wins:,}  |  Losses: {losses:,}")
    print(f"  Overall Win Rate: {wr:.2f}%")
    print(f"  Overall ROI: {roi:.2f}%")
    print(f"  Breakeven Threshold: 52.38%")
    
    if wr > 52.38:
        print(f"\n  ✅ ENGINE IS PROFITABLE ACROSS BOTH SEASONS.")
    else:
        print(f"\n  ❌ ENGINE IS UNPROFITABLE ACROSS BOTH SEASONS.")
    
    # --- BET TYPE BREAKDOWN ---
    over_bets = combined[combined['bet_placed'] == 'OVER']
    under_bets = combined[combined['bet_placed'] == 'UNDER']
    
    over_wr = (over_bets['win'].sum() / len(over_bets) * 100) if len(over_bets) > 0 else 0
    under_wr = (under_bets['win'].sum() / len(under_bets) * 100) if len(under_bets) > 0 else 0
    
    print(f"\n  --- BET TYPE BREAKDOWN ---")
    print(f"  OVER Bets:  {len(over_bets):,} placed | Win Rate: {over_wr:.2f}%")
    print(f"  UNDER Bets: {len(under_bets):,} placed | Win Rate: {under_wr:.2f}%")

if __name__ == "__main__":
    run_multi_season_backtest()
