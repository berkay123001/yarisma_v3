import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

# --- CONFIG ---
TARGET_COL = 'loan_paid_back'
N_SPLITS = 5
SEED = 42

# --- MODEL PARAMETERS ---

# 1. LightGBM (Optuna Trial 0 Best Params)
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.0426,
    'num_leaves': 59,
    'max_depth': 6,
    'reg_alpha': 0.468,
    'reg_lambda': 9.56,
    'min_child_samples': 65,
    'subsample': 0.955,
    'n_estimators': 2000,
    'n_jobs': -1,
    'device': 'cpu',
    'verbosity': -1,
    'random_state': SEED
}

# 2. XGBoost (Fast CPU Params)
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.01,
    'max_depth': 6,
    'tree_method': 'hist',  # Fast histogram optimized for CPU
    'n_jobs': -1,
    'random_state': SEED,
    'n_estimators': 2000,
    'early_stopping_rounds': 50
}

# 3. CatBoost (CPU Params)
cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.01,
    'depth': 6,
    'iterations': 2000,
    'task_type': 'CPU',
    'thread_count': -1,
    'random_seed': SEED,
    'verbose': 200,
    'early_stopping_rounds': 50
}

def load_data():
    print("Loading data...")
    if os.path.exists('Processed/clean_train.csv'):
        train = pd.read_csv('Processed/clean_train.csv')
        test = pd.read_csv('Processed/clean_test.csv')
    elif os.path.exists('clean_train.csv'):
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
    else:
        raise FileNotFoundError("Could not find clean_train.csv")
    return train, test

def feature_engineering_v1(df):
    # 1. Employment / Credit Score Ratio
    df['employment_credit_ratio'] = df['employment_status'] / (df['credit_score'] + 1e-6)
    
    # 2. Employment * Grade Interaction
    df['employment_grade_interaction'] = df['employment_status'] * df['grade_subgrade']
    
    # 3. Employment * DTI Interaction
    df['employment_dti_interaction'] = df['employment_status'] * df['debt_to_income_ratio']
    
    return df

def train_and_predict(model_name, model, X, y, X_test, test_ids, output_oof_name, output_sub_name):
    print(f"\n--- Training {model_name} ---")
    print(f"Training features: {list(X.columns)}")
    
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Fit model
        if model_name == 'LightGBM':
            callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=callbacks
            )
            val_preds = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
            
        elif model_name == 'XGBoost':
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            val_preds = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
            
        elif model_name == 'CatBoost':
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True
            )
            val_preds = model.predict_proba(X_val)[:, 1]
            test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
            
        oof_preds[val_idx] = val_preds
        
    auc_score = roc_auc_score(y, oof_preds)
    print(f"{model_name} OOF AUC: {auc_score:.5f}")
    
    # Save OOF
    os.makedirs('Models', exist_ok=True)
    oof_df = pd.DataFrame({TARGET_COL: oof_preds})
    oof_df.to_csv(f'Models/{output_oof_name}', index=False)
    print(f"Saved OOF to Models/{output_oof_name}")
    
    # Save Submission
    sub_df = pd.DataFrame({'id': test_ids, TARGET_COL: test_preds})
    sub_df.to_csv(output_sub_name, index=False)
    print(f"Saved Submission to {output_sub_name}")
    
    return auc_score

def main():
    # 1. Load Data
    train, test = load_data()
    
    # 2. Feature Engineering
    print("Applying V1 Feature Engineering...")
    train = feature_engineering_v1(train)
    test = feature_engineering_v1(test)
    
    # Prepare X and y
    # Drop 'id' if it exists
    if 'id' in train.columns:
        X = train.drop([TARGET_COL, 'id'], axis=1)
    else:
        X = train.drop(TARGET_COL, axis=1)
        
    y = train[TARGET_COL]
    
    # Prepare X_test and test_ids
    if 'id' in test.columns:
        X_test = test.drop('id', axis=1)
        test_ids = test['id']
    else:
        X_test = test
        test_ids = range(len(test))
        
    print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")
    
    # 3. Train Models
    
    # LightGBM
    lgbm_model = lgb.LGBMClassifier(**lgbm_params)
    train_and_predict('LightGBM', lgbm_model, X, y, X_test, test_ids, 'oof_lgbm_optuna.csv', 'submission_optuna.csv')
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)
    train_and_predict('XGBoost', xgb_model, X, y, X_test, test_ids, 'oof_xgb_fast.csv', 'submission_xgb_fast.csv')
    
    # CatBoost
    cat_model = CatBoostClassifier(**cat_params)
    train_and_predict('CatBoost', cat_model, X, y, X_test, test_ids, 'oof_cat_fast.csv', 'submission_cat_fast.csv')
    
    print("\nAll models trained successfully!")

if __name__ == "__main__":
    main()
