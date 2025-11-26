import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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
    'n_estimators': 3000, # Increased slightly for more features
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
    else:
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
    return train, test

def feature_engineering_v1(df):
    df['employment_credit_ratio'] = df['employment_status'] / (df['credit_score'] + 1e-6)
    df['employment_grade_interaction'] = df['employment_status'] * df['grade_subgrade']
    df['employment_dti_interaction'] = df['employment_status'] * df['debt_to_income_ratio']
    return df

def apply_magic_features(df):
    # Top 5 Magic Features from Brute-Force
    df['debt_to_income_ratio_div_employment_dti_interaction'] = df['debt_to_income_ratio'] / (df['employment_dti_interaction'] + 1e-6)
    df['debt_to_income_ratio_minus_employment_dti_interaction'] = df['debt_to_income_ratio'] - df['employment_dti_interaction']
    df['debt_to_income_ratio_plus_employment_grade_interaction'] = df['debt_to_income_ratio'] + df['employment_grade_interaction']
    df['debt_to_income_ratio_minus_employment_grade_interaction'] = df['debt_to_income_ratio'] - df['employment_grade_interaction']
    df['credit_score_div_employment_credit_ratio'] = df['credit_score'] / (df['employment_credit_ratio'] + 1e-6)
    return df

def add_residual_features(train, test):
    print("Adding Residual Features (TabPFN + AutoGluon)...")
    
    # 1. TabPFN
    if os.path.exists('Models/tabpfn_oof.csv') and os.path.exists('Models/tabpfn_test.csv'):
        tabpfn_oof = pd.read_csv('Models/tabpfn_oof.csv')[TARGET_COL]
        tabpfn_test = pd.read_csv('Models/tabpfn_test.csv')[TARGET_COL]
        
        # Convert to Logits
        epsilon = 1e-6
        tabpfn_oof = np.clip(tabpfn_oof, epsilon, 1 - epsilon)
        tabpfn_test = np.clip(tabpfn_test, epsilon, 1 - epsilon)
        
        train['tabpfn_logit'] = np.log(tabpfn_oof / (1 - tabpfn_oof))
        test['tabpfn_logit'] = np.log(tabpfn_test / (1 - tabpfn_test))
        print(" - Added TabPFN Logits")
    else:
        print(" ! Warning: TabPFN files not found. Skipping.")

    # 2. AutoGluon (Original Data Model)
    # Note: Column name in csv is 'stage1_ag_pred'
    if os.path.exists('Models/stage1_autogluon_train.csv') and os.path.exists('Models/stage1_autogluon_test.csv'):
        ag_train = pd.read_csv('Models/stage1_autogluon_train.csv')['stage1_ag_pred']
        ag_test = pd.read_csv('Models/stage1_autogluon_test.csv')['stage1_ag_pred']
        
        epsilon = 1e-6
        ag_train = np.clip(ag_train, epsilon, 1 - epsilon)
        ag_test = np.clip(ag_test, epsilon, 1 - epsilon)
        
        train['autogluon_logit'] = np.log(ag_train / (1 - ag_train))
        test['autogluon_logit'] = np.log(ag_test / (1 - ag_test))
        print(" - Added AutoGluon (Original Data) Logits")
    else:
        print(" ! Warning: AutoGluon files not found. Skipping.")
        
    return train, test

def main():
    # 1. Load Data
    train, test = load_data()
    
    # 2. Feature Engineering (V1 + Magic)
    print("Applying Base & Magic Features...")
    train = feature_engineering_v1(train)
    test = feature_engineering_v1(test)
    
    train = apply_magic_features(train)
    test = apply_magic_features(test)
    
    # 3. Add Residual Features
    train, test = add_residual_features(train, test)
    
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
    
    # 4. Train Final LightGBM
    print("\n--- Training Final Boosted LightGBM (AutoGluon Enhanced) ---")
    
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
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
    print(f"\nFinal Boosted OOF AUC: {auc_score:.5f}")
    
    # Save Submission
    os.makedirs('Submissions', exist_ok=True)
    sub_df = pd.DataFrame({'id': test_ids, TARGET_COL: test_preds})
    sub_df.to_csv('Submissions/submission_boost_autogluon.csv', index=False)
    print("Saved Submission to Submissions/submission_boost_autogluon.csv")
    
    # Save OOF
    os.makedirs('Models', exist_ok=True)
    pd.DataFrame({TARGET_COL: oof_preds}).to_csv('Models/oof_boost_autogluon.csv', index=False)

if __name__ == "__main__":
    main()
