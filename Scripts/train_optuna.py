import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_data():
    print("Loading data...")
    if os.path.exists('Processed/clean_train.csv'):
        train = pd.read_csv('Processed/clean_train.csv')
        test = pd.read_csv('Processed/clean_test.csv')
    elif os.path.exists('/content/clean_train.csv'):
        train = pd.read_csv('/content/clean_train.csv')
        test = pd.read_csv('/content/clean_test.csv')
    else:
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
    return train, test

def feature_engineering_v1(df):
    # 1. employment_credit_ratio
    df['employment_credit_ratio'] = df['employment_status'] / (df['credit_score'] + 1e-6)
    # 2. employment_grade_interaction
    df['employment_grade_interaction'] = df['employment_status'] * df['grade_subgrade']
    # 3. employment_dti_interaction
    df['employment_dti_interaction'] = df['employment_status'] * df['debt_to_income_ratio']
    return df

def objective(trial, X, y):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': 0.8, # Keeping this fixed as per strategy or could tune
        'n_estimators': 1000
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=callbacks
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
    return roc_auc_score(y, oof_preds)

def train_final_model(train, test, best_params):
    print("\nRetraining with Best Parameters...")
    
    X = train.drop('loan_paid_back', axis=1)
    y = train['loan_paid_back']
    X_test = test.drop('id', axis=1)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))
    oof_preds = np.zeros(len(X))
    
    # Ensure n_estimators and other fixed params are set
    final_params = best_params.copy()
    final_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': -1,
        'colsample_bytree': 0.8,
        'n_estimators': 1000
    })
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**final_params)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=callbacks
        )
        
        test_preds += model.predict_proba(X_test)[:, 1] / skf.get_n_splits()
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
    final_auc = roc_auc_score(y, oof_preds)
    print(f"Final OOF AUC with Best Params: {final_auc:.5f}")
    
    return test_preds, final_auc

def create_submission(test, preds):
    print("Creating submission_optuna.csv...")
    submission = pd.DataFrame({
        'id': test['id'],
        'loan_paid_back': preds
    })
    submission.to_csv('submission_optuna.csv', index=False)
    print("submission_optuna.csv saved successfully.")

def main():
    train, test = load_data()
    
    # Apply V1 Features
    train = feature_engineering_v1(train)
    test = feature_engineering_v1(test)
    
    X = train.drop('loan_paid_back', axis=1)
    y = train['loan_paid_back']
    
    print("Starting Optuna Optimization (20 Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=20)
    
    print("\nBest Trial:")
    print(f"  Value: {study.best_value:.5f}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    # Save best params
    os.makedirs('Models', exist_ok=True)
    with open('Models/best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    # Retrain and Submit
    preds, score = train_final_model(train, test, study.best_params)
    create_submission(test, preds)

if __name__ == "__main__":
    main()
