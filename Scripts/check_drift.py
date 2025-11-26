import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

# --- CONFIG ---
TRAIN_PATH = 'Processed/clean_train.csv'
TEST_PATH = 'Processed/clean_test.csv'
DROP_COLS = ['id', 'loan_paid_back']

def main():
    print("Loading Data...")
    if os.path.exists(TRAIN_PATH):
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
    else:
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
        
    # Prepare Adversarial Data
    train['is_test'] = 0
    test['is_test'] = 1
    
    # Combine
    adv_data = pd.concat([train, test], axis=0).reset_index(drop=True)
    
    # Features to check
    features = [c for c in train.columns if c not in DROP_COLS + ['is_test']]
    print(f"Checking {len(features)} features for drift...")
    
    X = adv_data[features]
    y = adv_data['is_test']
    
    # Train Adversarial Model
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'n_jobs': -1,
        'verbose': -1,
        'random_state': 42
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    importances = np.zeros(len(features))
    
    print("Training Adversarial Model...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        importances += model.feature_importances_ / 5
        
    adv_auc = roc_auc_score(y, oof_preds)
    print(f"\nAdversarial AUC: {adv_auc:.5f}")
    
    if adv_auc > 0.60:
        print("⚠️  SIGNIFICANT DRIFT DETECTED!")
        print("The model can distinguish Train from Test. We must drop drifting features.")
    elif adv_auc > 0.55:
        print("⚠️  Minor Drift Detected.")
    else:
        print("✅ No significant drift. Train and Test are consistent.")
        
    # Show Top Drifting Features
    imp_df = pd.DataFrame({'feature': features, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False)
    
    print("\n--- Top Drifting Features ---")
    print(imp_df.head(10))
    
    # Save list of drifting features (arbitrary threshold)
    drifting_feats = imp_df[imp_df['importance'] > 50]['feature'].tolist() # Threshold needs tuning
    if drifting_feats:
        print(f"\nPotential features to drop ({len(drifting_feats)}): {drifting_feats}")

if __name__ == "__main__":
    main()
