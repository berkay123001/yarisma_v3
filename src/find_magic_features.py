import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import itertools
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
TARGET_COL = 'loan_paid_back'
N_SPLITS = 5
SEED = 42
TOP_N_FEATURES = 5  # Number of magic features to keep

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
    # Basic V1 features
    df['employment_credit_ratio'] = df['employment_status'] / (df['credit_score'] + 1e-6)
    df['employment_grade_interaction'] = df['employment_status'] * df['grade_subgrade']
    df['employment_dti_interaction'] = df['employment_status'] * df['debt_to_income_ratio']
    return df

def evaluate_feature(train, feature_name, target):
    # Fast evaluation using a small LightGBM
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_estimators': 100, # Fast check
        'random_state': SEED,
        'n_jobs': -1
    }
    
    # Use a single validation split for speed
    # Split last 20% as validation
    split_idx = int(len(train) * 0.8)
    X_train = train.iloc[:split_idx][[feature_name]]
    y_train = target.iloc[:split_idx]
    X_val = train.iloc[split_idx:][[feature_name]]
    y_val = target.iloc[split_idx:]
    
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    
    score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    return score

def main():
    # 1. Load Data
    train, test = load_data()
    
    # 2. Base Features
    train = feature_engineering_v1(train)
    test = feature_engineering_v1(test)
    
    # Identify numeric columns for interaction
    numeric_cols = [
        'annual_income', 'debt_to_income_ratio', 'credit_score', 
        'loan_amount', 'interest_rate'
    ]
    # Add V1 interaction features to the mix if they are numeric
    numeric_cols.extend(['employment_credit_ratio', 'employment_grade_interaction', 'employment_dti_interaction'])
    
    print(f"Base Numeric Features: {numeric_cols}")
    
    # 3. Brute-Force Generation
    print("\n--- Starting Brute-Force Feature Generation ---")
    
    new_features_scores = []
    
    # Generate pairs
    for col1, col2 in itertools.combinations(numeric_cols, 2):
        # Operations: +, -, *, /
        
        # Multiply (*)
        feat_name = f"{col1}_mul_{col2}"
        train[feat_name] = train[col1] * train[col2]
        score = evaluate_feature(train, feat_name, train[TARGET_COL])
        new_features_scores.append((feat_name, score, '*', col1, col2))
        train.drop(columns=[feat_name], inplace=True) # Clean up to save memory
        
        # Divide (/) - Both directions
        # col1 / col2
        feat_name = f"{col1}_div_{col2}"
        train[feat_name] = train[col1] / (train[col2] + 1e-6)
        score = evaluate_feature(train, feat_name, train[TARGET_COL])
        new_features_scores.append((feat_name, score, '/', col1, col2))
        train.drop(columns=[feat_name], inplace=True)
        
        # col2 / col1
        feat_name = f"{col2}_div_{col1}"
        train[feat_name] = train[col2] / (train[col1] + 1e-6)
        score = evaluate_feature(train, feat_name, train[TARGET_COL])
        new_features_scores.append((feat_name, score, '/', col2, col1))
        train.drop(columns=[feat_name], inplace=True)
        
        # Add (+)
        feat_name = f"{col1}_plus_{col2}"
        train[feat_name] = train[col1] + train[col2]
        score = evaluate_feature(train, feat_name, train[TARGET_COL])
        new_features_scores.append((feat_name, score, '+', col1, col2))
        train.drop(columns=[feat_name], inplace=True)
        
        # Subtract (-)
        feat_name = f"{col1}_minus_{col2}"
        train[feat_name] = train[col1] - train[col2]
        score = evaluate_feature(train, feat_name, train[TARGET_COL])
        new_features_scores.append((feat_name, score, '-', col1, col2))
        train.drop(columns=[feat_name], inplace=True)

    # 4. Select Top Features
    # Sort by AUC score (higher is better)
    new_features_scores.sort(key=lambda x: x[1], reverse=True)
    
    top_features = new_features_scores[:TOP_N_FEATURES]
    
    print(f"\n--- Top {TOP_N_FEATURES} Magic Features Found ---")
    print(f"{'Feature Name':<40} | {'AUC Score':<10}")
    print("-" * 55)
    
    selected_feat_names = []
    
    # Apply selected features to Train and Test
    for feat_name, score, op, c1, c2 in top_features:
        print(f"{feat_name:<40} | {score:.5f}")
        selected_feat_names.append(feat_name)
        
        if op == '*':
            train[feat_name] = train[c1] * train[c2]
            test[feat_name] = test[c1] * test[c2]
        elif op == '/':
            train[feat_name] = train[c1] / (train[c2] + 1e-6)
            test[feat_name] = test[c1] / (test[c2] + 1e-6)
        elif op == '+':
            train[feat_name] = train[c1] + train[c2]
            test[feat_name] = test[c1] + test[c2]
        elif op == '-':
            train[feat_name] = train[c1] - train[c2]
            test[feat_name] = test[c1] - test[c2]
            
    # Save list to file
    with open('magic_features_list.txt', 'w') as f:
        for item in top_features:
            f.write(f"{item[0]},{item[1]}\n")
            
    # 5. Train Full Model with Magic Features
    print("\n--- Training Full Model with Magic Features ---")
    
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
    print(f"\nBrute-Force OOF AUC: {auc_score:.5f}")
    
    # Save OOF
    os.makedirs('Models', exist_ok=True)
    pd.DataFrame({TARGET_COL: oof_preds}).to_csv('Models/oof_lgbm_bruteforce.csv', index=False)
    print("Saved OOF to Models/oof_lgbm_bruteforce.csv")
    
    # Save Submission
    os.makedirs('Submissions', exist_ok=True)
    sub_df = pd.DataFrame({'id': test_ids, TARGET_COL: test_preds})
    sub_df.to_csv('Submissions/submission_bruteforce.csv', index=False)
    print("Saved Submission to Submissions/submission_bruteforce.csv")

if __name__ == "__main__":
    main()
