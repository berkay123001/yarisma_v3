# 1. Kütüphaneler
!pip install pytorch-tabnet

import pandas as pd
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

# --- CONFIG ---
N_SPLITS = 5
SEED = 42
MAX_EPOCHS = 50
BATCH_SIZE = 1024
VIRTUAL_BATCH_SIZE = 128

def load_data():
    if os.path.exists('clean_train.csv'):
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
    else:
        raise FileNotFoundError("Lütfen 'clean_train.csv' ve 'clean_test.csv' yükleyin!")
    return train, test

def main():
    # 1. Load
    train, test = load_data()
    
    target_col = 'loan_paid_back'
    drop_cols = ['id', target_col]
    
    if target_col in train.columns:
        X = train.drop(columns=[c for c in drop_cols if c in train.columns]).values
        y = train[target_col].values
    else:
        raise ValueError("Target yok!")
        
    if 'id' in test.columns:
        X_test = test.drop(columns=['id']).values
        test_ids = test['id'].values
    else:
        X_test = test.values
        test_ids = range(len(test))
        
    # 2. Scale (TabNet likes scaled data)
    print("Scaling...")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    
    # 3. Train TabNet
    print("Training TabNet...")
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"\nFold {fold+1}/{N_SPLITS}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        clf = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='entmax', # "sparsemax"
            verbose=1
        )
        
        clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=['val'],
            eval_metric=['auc'],
            max_epochs=MAX_EPOCHS, 
            patience=15,
            batch_size=BATCH_SIZE, 
            virtual_batch_size=VIRTUAL_BATCH_SIZE,
            num_workers=0,
            drop_last=False
        )
        
        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
        test_preds += clf.predict_proba(X_test)[:, 1] / N_SPLITS
        
    auc = roc_auc_score(y, oof_preds)
    print(f"\nFinal TabNet OOF AUC: {auc:.5f}")
    
    # Save
    sub = pd.DataFrame({'id': test_ids, 'loan_paid_back': test_preds})
    sub.to_csv('submission_tabnet.csv', index=False)
    
    oof = pd.DataFrame({'loan_paid_back': oof_preds})
    oof.to_csv('oof_tabnet.csv', index=False)
    
    from google.colab import files
    files.download('submission_tabnet.csv')
    files.download('oof_tabnet.csv')

if __name__ == "__main__":
    main()
