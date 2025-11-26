import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import os

# --- CONFIG ---
MODELS = {
    'Brute': {
        'oof': 'Models/oof_lgbm_bruteforce.csv',
        'sub': 'Submissions/submission_bruteforce.csv'
    },
    'DAE': {
        'oof': 'Models/oof_dae_boosted.csv',
        'sub': 'Submissions/submission_dae_boosted.csv'
    },
    'Genetic': {
        'oof': 'Models/oof_genetic.csv',
        'sub': 'Submissions/submission_genetic.csv'
    },
    'AutoGluon': {
        'oof': 'Models/oof_boost_autogluon.csv',
        'sub': 'Submissions/submission_boost_autogluon.csv'
    },
    'Patched': {
        'oof': 'Models/oof_patched_bruteforce.csv',
        'sub': 'Submissions/submission_patched_bruteforce.csv'
    },
    'TabNet': {
        'oof': 'Models/oof_tabnet.csv',
        'sub': 'Submissions/submission_tabnet.csv'
    },
    'Golden': {
        'oof': 'Models/oof_golden.csv',
        'sub': 'Submissions/submission_golden.csv'
    }
}

TARGET_COL = 'loan_paid_back'
TRAIN_PATH = 'Processed/clean_train.csv'

def load_data():
    print("Loading Ground Truth...")
    if os.path.exists(TRAIN_PATH):
        train = pd.read_csv(TRAIN_PATH)
    elif os.path.exists('clean_train.csv'):
        train = pd.read_csv('clean_train.csv')
    else:
        raise FileNotFoundError("Train data not found!")
    return train[TARGET_COL].values

def load_oofs_and_subs():
    print("Loading OOFs and Submissions...")
    oofs = []
    subs = []
    names = []
    
    for name, paths in MODELS.items():
        if os.path.exists(paths['oof']) and os.path.exists(paths['sub']):
            print(f" - Loaded {name}")
            oof_df = pd.read_csv(paths['oof'])
            sub_df = pd.read_csv(paths['sub'])
            
            if TARGET_COL in oof_df.columns:
                oofs.append(oof_df[TARGET_COL].values)
            else:
                oofs.append(oof_df.iloc[:, 0].values)
                
            if TARGET_COL in sub_df.columns:
                subs.append(sub_df[TARGET_COL].values)
            else:
                subs.append(sub_df.iloc[:, 1].values)
                
            names.append(name)
            
    return np.array(oofs).T, np.array(subs).T, names, sub_df['id']

def get_auc(weights, oof_preds, y_true):
    # Normalize
    weights = np.array(weights)
    if np.sum(weights) == 0: return 0
    weights /= np.sum(weights)
    
    final_pred = np.dot(oof_preds, weights)
    return roc_auc_score(y_true, final_pred)

def hill_climbing(oof_preds, y_true, names):
    print("\n--- Starting Hill Climbing Optimization ---")
    n_models = oof_preds.shape[1]
    weights = np.ones(n_models) / n_models
    best_auc = get_auc(weights, oof_preds, y_true)
    
    print(f"Initial AUC: {best_auc:.5f}")
    
    step_size = 0.01
    improved = True
    
    while improved:
        improved = False
        for i in range(n_models):
            # Try increasing weight
            original_weight = weights[i]
            weights[i] += step_size
            new_auc = get_auc(weights, oof_preds, y_true)
            
            if new_auc > best_auc:
                best_auc = new_auc
                improved = True
                # print(f"Improved! {names[i]} + : {best_auc:.6f}")
            else:
                # Revert and try decreasing
                weights[i] = original_weight - step_size
                if weights[i] < 0: weights[i] = 0 # Non-negative constraint
                
                new_auc = get_auc(weights, oof_preds, y_true)
                if new_auc > best_auc:
                    best_auc = new_auc
                    improved = True
                    # print(f"Improved! {names[i]} - : {best_auc:.6f}")
                else:
                    # Revert
                    weights[i] = original_weight
                    
    # Normalize final weights
    weights /= np.sum(weights)
    
    print("\nOptimal Weights:")
    for name, weight in zip(names, weights):
        print(f" - {name}: {weight:.4f}")
        
    print(f"\nOptimized OOF AUC: {best_auc:.5f}")
    return weights

def main():
    y_true = load_data()
    oof_preds, sub_preds, names, test_ids = load_oofs_and_subs()
    
    best_weights = hill_climbing(oof_preds, y_true, names)
    
    print("\nGenerating Optimized Submission...")
    final_test_preds = np.dot(sub_preds, best_weights)
    
    submission = pd.DataFrame({
        'id': test_ids,
        TARGET_COL: final_test_preds
    })
    
    os.makedirs('Submissions', exist_ok=True)
    out_file = 'Submissions/submission_optimized_ensemble.csv'
    submission.to_csv(out_file, index=False)
    print(f"Saved to {out_file}")

if __name__ == "__main__":
    main()
