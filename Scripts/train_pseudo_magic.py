import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

# --- CONFIG ---
TARGET_COL = 'loan_paid_back'
N_SPLITS = 5
SEED = 42
PSEUDO_THRESHOLD_POS = 0.995 
PSEUDO_THRESHOLD_NEG = 0.005 

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
    'n_estimators': 2000,
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

def main():
    # 1. Load Data
    train, test = load_data()
    
    # 2. Load BEST Submission (Brute-Force) for Pseudo-Labeling
    sub_path = 'Submissions/submission_bruteforce.csv'
    if not os.path.exists(sub_path):
        print(f"Error: {sub_path} not found.")
        return
        
    print(f"Loading best submission from {sub_path}...")
    best_sub = pd.read_csv(sub_path)
    
    # 3. Create Pseudo-Labels
    print("Creating Pseudo-Labels...")
    test_copy = test.copy()
    test_copy[TARGET_COL] = best_sub[TARGET_COL]
    
    pseudo_pos = test_copy[test_copy[TARGET_COL] > PSEUDO_THRESHOLD_POS]
    pseudo_neg = test_copy[test_copy[TARGET_COL] < PSEUDO_THRESHOLD_NEG]
    
    pseudo_pos[TARGET_COL] = 1
    pseudo_neg[TARGET_COL] = 0
    
    pseudo_data = pd.concat([pseudo_pos, pseudo_neg])
    
    print(f"Pseudo-Positives: {len(pseudo_pos)}")
    print(f"Pseudo-Negatives: {len(pseudo_neg)}")
    print(f"Total Pseudo-Samples: {len(pseudo_data)}")
    
    # 4. Augment Training Data
    train_augmented = pd.concat([train, pseudo_data], axis=0).reset_index(drop=True)
    print(f"Original Train Size: {len(train)}")
    print(f"Augmented Train Size: {len(train_augmented)}")
    
    # 5. Feature Engineering (V1 + Magic)
    print("Applying Features (V1 + Magic)...")
    train_augmented = feature_engineering_v1(train_augmented)
    test = feature_engineering_v1(test)
    
    train_augmented = apply_magic_features(train_augmented)
    test = apply_magic_features(test)
    
    # Prepare X and y
    if 'id' in train_augmented.columns:
        X = train_augmented.drop([TARGET_COL, 'id'], axis=1)
    else:
        X = train_augmented.drop(TARGET_COL, axis=1)
    y = train_augmented[TARGET_COL]
    
    if 'id' in test.columns:
        X_test = test.drop('id', axis=1)
        test_ids = test['id']
    else:
        X_test = test
        test_ids = range(len(test))
        
    # 6. Train LightGBM on Augmented Data
    print("\n--- Training LightGBM on Augmented Data ---")
    
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
    print(f"\nAugmented OOF AUC: {auc_score:.5f}")
    
    # Save Submission
    os.makedirs('Submissions', exist_ok=True)
    sub_df = pd.DataFrame({'id': test_ids, TARGET_COL: test_preds})
    sub_df.to_csv('Submissions/submission_pseudo_magic.csv', index=False)
    print("Saved Submission to Submissions/submission_pseudo_magic.csv")

if __name__ == "__main__":
    main()
