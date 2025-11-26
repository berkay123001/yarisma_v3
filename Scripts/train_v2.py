import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import KBinsDiscretizer
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def load_data():
    print("Loading data...")
    train = pd.read_csv('Processed/clean_train.csv')
    test = pd.read_csv('Processed/clean_test.csv')
    return train, test

def feature_engineering_v1(df):
    # 1. employment_credit_ratio
    df['employment_credit_ratio'] = df['employment_status'] / (df['credit_score'] + 1e-6)
    # 2. employment_grade_interaction
    df['employment_grade_interaction'] = df['employment_status'] * df['grade_subgrade']
    # 3. employment_dti_interaction
    df['employment_dti_interaction'] = df['employment_status'] * df['debt_to_income_ratio']
    return df

def feature_engineering_v2(train, test):
    print("Applying V2 Feature Engineering...")
    
    # Combine for consistent binning (optional but good practice if no leakage)
    # Here we will fit on train and transform both to avoid leakage
    
    # 1. Log Transform Income
    train['annual_income_log'] = np.log1p(train['annual_income'])
    test['annual_income_log'] = np.log1p(test['annual_income'])
    
    # 2. Credit Score Binning
    # Using KBinsDiscretizer
    est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    
    # Reshape for sklearn
    train_cs = train[['credit_score']]
    test_cs = test[['credit_score']]
    
    train['credit_score_bin'] = est.fit_transform(train_cs)
    test['credit_score_bin'] = est.transform(test_cs)
    
    return train, test

def train_v2(train, test):
    print("Starting V2 Training...")
    
    X = train.drop('loan_paid_back', axis=1)
    y = train['loan_paid_back']
    X_test = test.drop('id', axis=1)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros(len(X))
    test_lgb = np.zeros(len(X_test))
    
    oof_xgb = np.zeros(len(X))
    test_xgb = np.zeros(len(X_test))
    
    # LightGBM Params
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
    
    # XGBoost Params
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'max_depth': 4,
        'colsample_bytree': 0.8,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': 0
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold+1} ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # LightGBM
        model_lgb = lgb.LGBMClassifier(**lgb_params)
        callbacks_lgb = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model_lgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='auc', callbacks=callbacks_lgb)
        
        val_pred_lgb = model_lgb.predict_proba(X_val)[:, 1]
        oof_lgb[val_idx] = val_pred_lgb
        test_lgb += model_lgb.predict_proba(X_test)[:, 1] / skf.get_n_splits()
        print(f"LGB AUC: {roc_auc_score(y_val, val_pred_lgb):.5f}")
        
        # XGBoost
        model_xgb = xgb.XGBClassifier(**xgb_params, early_stopping_rounds=50)
        model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        val_pred_xgb = model_xgb.predict_proba(X_val)[:, 1]
        oof_xgb[val_idx] = val_pred_xgb
        test_xgb += model_xgb.predict_proba(X_test)[:, 1] / skf.get_n_splits()
        print(f"XGB AUC: {roc_auc_score(y_val, val_pred_xgb):.5f}")
        
    score_lgb = roc_auc_score(y, oof_lgb)
    score_xgb = roc_auc_score(y, oof_xgb)
    
    print(f"\n--- V2 Results ---")
    print(f"LightGBM OOF: {score_lgb:.5f}")
    print(f"XGBoost OOF:  {score_xgb:.5f}")
    
    return test_lgb, score_lgb

def create_submission(test, preds):
    print("Creating submission_v2.csv...")
    submission = pd.DataFrame({
        'id': test['id'],
        'loan_paid_back': preds
    })
    submission.to_csv('submission_v2.csv', index=False)
    print("submission_v2.csv saved successfully.")

def main():
    train, test = load_data()
    
    # Apply V1 first
    train = feature_engineering_v1(train)
    test = feature_engineering_v1(test)
    
    # Apply V2
    train, test = feature_engineering_v2(train, test)
    
    preds, score = train_v2(train, test)
    
    BASELINE_SCORE = 0.92217
    if score > BASELINE_SCORE:
        print(f"\nSUCCESS: New Score ({score:.5f}) > Baseline ({BASELINE_SCORE})")
        create_submission(test, preds)
    else:
        print(f"\nFAILURE: New Score ({score:.5f}) <= Baseline ({BASELINE_SCORE})")
        print("Not generating submission file.")

if __name__ == "__main__":
    main()
