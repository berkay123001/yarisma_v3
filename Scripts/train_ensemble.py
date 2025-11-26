import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data():
    print("Loading data...")
    train = pd.read_csv('Processed/clean_train.csv')
    test = pd.read_csv('Processed/clean_test.csv')
    return train, test

def feature_engineering(df):
    print("Engineering features...")
    # 1. employment_credit_ratio
    df['employment_credit_ratio'] = df['employment_status'] / (df['credit_score'] + 1e-6)
    
    # 2. employment_grade_interaction
    df['employment_grade_interaction'] = df['employment_status'] * df['grade_subgrade']
    
    # 3. employment_dti_interaction
    df['employment_dti_interaction'] = df['employment_status'] * df['debt_to_income_ratio']
    
    return df

def train_ensemble(train, test):
    print("Starting Ensemble Training...")
    
    X = train.drop('loan_paid_back', axis=1)
    y = train['loan_paid_back']
    X_test = test.drop('id', axis=1)
    
    # Stratified K-Fold (Same seed as baseline for consistency, though we retrain everything)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Placeholders for OOF and Test predictions
    oof_lgb = np.zeros(len(X))
    test_lgb = np.zeros(len(X_test))
    
    oof_xgb = np.zeros(len(X))
    test_xgb = np.zeros(len(X_test))
    
    oof_cat = np.zeros(len(X))
    test_cat = np.zeros(len(X_test))
    
    # --- Model Parameters ---
    # LightGBM
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'max_depth': 4,
        'colsample_bytree': 0.8,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': -1
    }
    
    # XGBoost
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist', # CPU optimized
        'max_depth': 4,
        'colsample_bytree': 0.8,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }
    
    # CatBoost
    cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'task_type': 'CPU',
        'depth': 4,
        'learning_rate': 0.05,
        'iterations': 1000,
        'random_seed': 42,
        'verbose': 0,
        'allow_writing_files': False
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 1. LightGBM
        print("Training LightGBM...")
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        callbacks_lgb = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=callbacks_lgb)
        
        val_pred_lgb = model_lgb.predict_proba(X_val)[:, 1]
        oof_lgb[val_idx] = val_pred_lgb
        test_lgb += model_lgb.predict_proba(X_test)[:, 1] / skf.get_n_splits()
        print(f"LGB AUC: {roc_auc_score(y_val, val_pred_lgb):.5f}")

        # 2. XGBoost
        print("Training XGBoost...")
        model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_pred_xgb = model_xgb.predict_proba(X_val)[:, 1]
        oof_xgb[val_idx] = val_pred_xgb
        test_xgb += model_xgb.predict_proba(X_test)[:, 1] / skf.get_n_splits()
        print(f"XGB AUC: {roc_auc_score(y_val, val_pred_xgb):.5f}")

        # 3. CatBoost
        print("Training CatBoost...")
        model_cat = CatBoostClassifier(**cat_params)
        model_cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)
        
        val_pred_cat = model_cat.predict_proba(X_val)[:, 1]
        oof_cat[val_idx] = val_pred_cat
        test_cat += model_cat.predict_proba(X_test)[:, 1] / skf.get_n_splits()
        print(f"Cat AUC: {roc_auc_score(y_val, val_pred_cat):.5f}")

    # Calculate Individual OOF Scores
    score_lgb = roc_auc_score(y, oof_lgb)
    score_xgb = roc_auc_score(y, oof_xgb)
    score_cat = roc_auc_score(y, oof_cat)
    
    print(f"\n--- Individual OOF Scores ---")
    print(f"LightGBM: {score_lgb:.5f}")
    print(f"XGBoost:  {score_xgb:.5f}")
    print(f"CatBoost: {score_cat:.5f}")
    
    # Blending
    print("\nBlending Predictions...")
    # Weights: 0.4 LGB + 0.3 XGB + 0.3 Cat
    oof_ensemble = (oof_lgb * 0.4) + (oof_xgb * 0.3) + (oof_cat * 0.3)
    test_ensemble = (test_lgb * 0.4) + (test_xgb * 0.3) + (test_cat * 0.3)
    
    score_ensemble = roc_auc_score(y, oof_ensemble)
    print(f"Ensemble OOF AUC Score: {score_ensemble:.5f}")
    
    return test_ensemble, score_ensemble

def create_submission(test, preds):
    print("Creating submission_ensemble.csv...")
    submission = pd.DataFrame({
        'id': test['id'],
        'loan_paid_back': preds
    })
    submission.to_csv('submission_ensemble.csv', index=False)
    print("submission_ensemble.csv saved successfully.")

def main():
    train, test = load_data()
    
    train = feature_engineering(train)
    test = feature_engineering(test)
    
    preds, score = train_ensemble(train, test)
    
    create_submission(test, preds)

if __name__ == "__main__":
    main()
