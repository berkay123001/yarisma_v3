import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import os

# --- CONFIG ---
# OOF Files (Train set predictions)
OOF_FILES = {
    'LGBM_Brute': 'Models/oof_lgbm_bruteforce.csv',
    # Ah, I need to check where brute force OOF is saved. 
    # In find_magic_features.py, OOF was not saved to file! 
    # I need to re-run find_magic_features.py to save OOF or use what I have.
    # Let's check previous steps. find_magic_features.py printed OOF but didn't save it to Models/
    
    'LGBM_Optuna': 'Models/oof_lgbm_optuna.csv',
    'XGB': 'Models/oof_xgb_fast.csv',
    'CatBoost': 'Models/oof_cat_fast.csv',
    'MLP_Magic': 'Models/oof_mlp_magic.csv'
}

# Submission Files (Test set predictions)
SUB_FILES = {
    'LGBM_Brute': 'Submissions/submission_bruteforce.csv',
    'LGBM_Optuna': 'Submissions/Archive/submission_optuna.csv',
    'XGB': 'Submissions/Archive/submission_xgb_fast.csv',
    'CatBoost': 'Submissions/Archive/submission_cat_fast.csv',
    'MLP_Magic': 'Submissions/submission_mlp_magic.csv'
}

TARGET_COL = 'loan_paid_back'

def load_data():
    print("Loading data...")
    if os.path.exists('Processed/clean_train.csv'):
        train = pd.read_csv('Processed/clean_train.csv')
    elif os.path.exists('clean_train.csv'):
        train = pd.read_csv('clean_train.csv')
    else:
        raise FileNotFoundError("Could not find clean_train.csv")
        
    # Load OOF predictions
    oofs = {}
    for model_name, path in OOF_FILES.items():
        # Special handling for Brute Force if OOF is missing
        if model_name == 'LGBM_Brute' and not os.path.exists(path):
             print(f"Warning: OOF for {model_name} not found at {path}. Skipping...")
             continue

        if os.path.exists(path):
            df = pd.read_csv(path)
            if TARGET_COL in df.columns:
                oofs[model_name] = df[TARGET_COL]
            else:
                oofs[model_name] = df.iloc[:, -1]
        else:
            print(f"Warning: Missing OOF file: {path}")

    # Load Test predictions
    subs = {}
    test_ids = None
    for model_name, path in SUB_FILES.items():
        if model_name not in oofs: continue # Only load sub if OOF exists

        if os.path.exists(path):
            df = pd.read_csv(path)
            if test_ids is None and 'id' in df.columns:
                test_ids = df['id']
            
            if TARGET_COL in df.columns:
                subs[model_name] = df[TARGET_COL]
            else:
                subs[model_name] = df.iloc[:, -1]
        else:
            print(f"Warning: Missing Submission file: {path}")
            
    return train, oofs, subs, test_ids

def main():
    train, oofs, subs, test_ids = load_data()
    
    if not oofs:
        print("No models to stack!")
        return

    X_stack_train = pd.DataFrame(oofs)
    y = train[TARGET_COL]
    
    # Handle NaNs
    imputer = SimpleImputer(strategy='mean')
    X_stack_train = pd.DataFrame(imputer.fit_transform(X_stack_train), columns=X_stack_train.columns)

    # 1. Correlation Check
    print("\n--- Correlation Matrix ---")
    print(X_stack_train.corr())
    
    # 2. Meta-Model Training
    print("\n--- Training Meta-Learner (Logistic Regression) ---")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_model = LogisticRegression(random_state=42)
    
    stack_oof_preds = np.zeros(len(X_stack_train))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_stack_train, y)):
        X_train, X_val = X_stack_train.iloc[train_idx], X_stack_train.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        meta_model.fit(X_train, y_train)
        stack_oof_preds[val_idx] = meta_model.predict_proba(X_val)[:, 1]
        
    print(f"\nStacking OOF AUC: {roc_auc_score(y, stack_oof_preds):.5f}")
    
    meta_model.fit(X_stack_train, y)
    print("\n--- Weights ---")
    for name, coef in zip(X_stack_train.columns, meta_model.coef_[0]):
        print(f"{name}: {coef:.4f}")
        
    # 3. Submission
    print("\nGenerating Submission...")
    X_stack_test = pd.DataFrame(subs)
    X_stack_test = X_stack_test[X_stack_train.columns] # Ensure order
    X_stack_test = pd.DataFrame(imputer.transform(X_stack_test), columns=X_stack_test.columns)
    
    final_preds = meta_model.predict_proba(X_stack_test)[:, 1]
    
    sub = pd.DataFrame({'id': test_ids, 'loan_paid_back': final_preds})
    sub.to_csv('Submissions/submission_dream_ensemble.csv', index=False)
    print("Saved to Submissions/submission_dream_ensemble.csv")

if __name__ == "__main__":
    main()
