import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
TARGET_COL = 'loan_paid_back'
N_SPLITS = 5
SEED = 42

# LightGBM Params (Best from Optuna)
LGBM_PARAMS = {
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
    'n_estimators': 3000, # Increased slightly for new features
    'n_jobs': -1,
    'device': 'cpu',
    'verbosity': -1,
    'random_state': SEED
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

def target_encoding(train, test, cat_cols, target_col, n_splits=5):
    print("Applying Target Encoding...")
    # Initialize new columns
    for col in cat_cols:
        train[f'{col}_target_enc'] = 0.0
        test[f'{col}_target_enc'] = 0.0

    # KFold for Train to prevent leakage
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    for train_idx, val_idx in kf.split(train):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        
        for col in cat_cols:
            # Calculate mean on training fold
            means = X_train.groupby(col)[target_col].mean()
            
            # Map to validation fold
            train.loc[val_idx, f'{col}_target_enc'] = X_val[col].map(means)
            
    # Fill NaNs in train (categories not present in training fold) with global mean
    global_mean = train[target_col].mean()
    for col in cat_cols:
        train[f'{col}_target_enc'].fillna(global_mean, inplace=True)
        
        # For Test, map using ALL train data
        means = train.groupby(col)[target_col].mean()
        test[f'{col}_target_enc'] = test[col].map(means)
        test[f'{col}_target_enc'].fillna(global_mean, inplace=True)
        
    return train, test

def aggregations(df):
    print("Applying Aggregations...")
    # GroupBy Aggregations
    
    # 1. Mean Income by Grade
    df['mean_income_by_grade'] = df.groupby('grade_subgrade')['annual_income'].transform('mean')
    df['income_div_mean_grade'] = df['annual_income'] / (df['mean_income_by_grade'] + 1)
    
    # 2. Max DTI by Employment Status
    df['max_dti_by_emp'] = df.groupby('employment_status')['debt_to_income_ratio'].transform('max')
    df['dti_div_max_emp'] = df['debt_to_income_ratio'] / (df['max_dti_by_emp'] + 1)
    
    # 3. Mean Credit Score by Loan Purpose
    df['mean_credit_by_purpose'] = df.groupby('loan_purpose')['credit_score'].transform('mean')
    
    return df

def isolation_forest_anomaly(train, test, features):
    print("Applying Isolation Forest Anomaly Detection...")
    # Combine for training IF
    combined = pd.concat([train[features], test[features]], axis=0)
    
    # Simple Imputation for IF
    combined = combined.fillna(combined.mean())
    
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=SEED, n_jobs=-1)
    iso.fit(combined)
    
    train['anomaly_score'] = iso.decision_function(train[features])
    test['anomaly_score'] = iso.decision_function(test[features])
    
    return train, test

def main():
    # 1. Load Data
    train, test = load_data()
    
    # 2. Basic V1 Features (Keep them)
    train['employment_credit_ratio'] = train['employment_status'] / (train['credit_score'] + 1e-6)
    test['employment_credit_ratio'] = test['employment_status'] / (test['credit_score'] + 1e-6)
    
    # 3. Target Encoding
    # Identify categorical columns suitable for TE
    # grade_subgrade is already ordinal encoded in previous steps? 
    # Let's check data types or assume standard names.
    # Based on previous scripts, 'grade_subgrade', 'loan_purpose' are good candidates.
    # NOTE: If they are already encoded as numbers, we can still target encode them as categories.
    
    cat_cols_to_enc = ['grade_subgrade', 'loan_purpose', 'employment_status']
    train, test = target_encoding(train, test, cat_cols_to_enc, TARGET_COL)
    
    # 4. Aggregations
    # Apply to both
    train = aggregations(train)
    test = aggregations(test)
    
    # 5. Isolation Forest
    # Use numerical features for anomaly detection
    iso_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
    train, test = isolation_forest_anomaly(train, test, iso_features)
    
    # Prepare X and y
    if 'id' in train.columns:
        X = train.drop([TARGET_COL, 'id'], axis=1)
    else:
        X = train.drop(TARGET_COL, axis=1)
    y = train[TARGET_COL]
    
    if 'id' in test.columns:
        X_test = test.drop('id', axis=1)
        test_ids = test['id']
    else:
        X_test = test
        test_ids = range(len(test))
        
    print(f"\nFinal Feature Count: {X.shape[1]}")
    print(f"Features: {list(X.columns)}")
    
    # 6. Train LightGBM
    print("\n--- Training LightGBM with Advanced Features ---")
    
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        callbacks = [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=callbacks
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        
        print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds):.5f}")
        
    auc_score = roc_auc_score(y, oof_preds)
    print(f"\nAdvanced Features OOF AUC: {auc_score:.5f}")
    
    # Compare with Baseline
    BASELINE_SCORE = 0.92277
    if auc_score > BASELINE_SCORE:
        print(f"SUCCESS: Improvement of {auc_score - BASELINE_SCORE:.5f} over baseline!")
    else:
        print(f"NOTE: Score is lower than baseline ({BASELINE_SCORE}). Check for overfitting or feature noise.")
    
    # Save Results
    os.makedirs('Models', exist_ok=True)
    pd.DataFrame({TARGET_COL: oof_preds}).to_csv('Models/oof_lgbm_advanced.csv', index=False)
    pd.DataFrame({'id': test_ids, TARGET_COL: test_preds}).to_csv('Submissions/submission_advanced.csv', index=False)
    print("Saved results to Models/oof_lgbm_advanced.csv and Submissions/submission_advanced.csv")

if __name__ == "__main__":
    main()
