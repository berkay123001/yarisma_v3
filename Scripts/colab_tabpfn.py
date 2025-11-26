# 1. TabPFN Kurulumu
!pip install tabpfn

# 2. Scriptin Çalıştırılması
import os
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
TARGET_COL = 'loan_paid_back'
N_SPLITS = 5
SEED = 42
SAMPLE_SIZE = 20000 # TabPFN is slow, so we might need to subsample for training or run in batches. 
# However, TabPFN is designed for small datasets. For large datasets like this (500k+), 
# we usually use it as an ensemble member or train on a subset.
# Chris Deotte's suggestion implies using it on "Original Data" (which is usually small).
# But here we are using it on the competition data.
# Strategy: We will train TabPFN on a subset (N=10k) and predict on the full dataset in batches.

def load_data():
    print("Loading data...")
    if os.path.exists('/content/clean_train.csv'):
        train = pd.read_csv('/content/clean_train.csv')
        test = pd.read_csv('/content/clean_test.csv')
    elif os.path.exists('clean_train.csv'):
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
    else:
        raise FileNotFoundError("Lütfen 'clean_train.csv' ve 'clean_test.csv' dosyalarını Colab'a yükleyin!")
    return train, test

def main():
    # 1. Load Data
    train, test = load_data()
    
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
        
    # TabPFN only handles numerical features well, or needs encoding.
    # Our clean data is already encoded (Ordinal/Label).
    # TabPFN handles up to ~100 features. We have ~15. Good.
    
    print("\n--- Generating TabPFN Predictions ---")
    
    # TabPFN Classifier
    # N_ensemble_configurations: Higher is better but slower. Default 3.
    classifier = TabPFNClassifier(device='cuda', N_ensemble_configurations=32)
    
    # Since TabPFN cannot train on 500k rows, we fit on a random subset.
    # It's a "Prior-Data Fitted Network", so fitting is fast (it just stores data).
    # The limit is usually around 10k samples for the context window.
    
    subset_idx = np.random.choice(len(X), size=10000, replace=False)
    X_subset = X.iloc[subset_idx]
    y_subset = y.iloc[subset_idx]
    
    print(f"Fitting TabPFN on {len(X_subset)} samples...")
    classifier.fit(X_subset, y_subset)
    
    # Predict in batches to avoid OOM
    BATCH_SIZE = 1000
    
    # 1. OOF Predictions (We need to predict on the full train set to use as feature)
    # Note: This is slightly leaky because we trained on a subset of Train.
    # But for "Stage 1 feature", it's acceptable if we are careful.
    # Ideally, we should do K-Fold, fitting on (Subset of Fold Train) and predicting on (Fold Val).
    
    print("Generating OOF Predictions (via K-Fold to avoid leakage)...")
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Processing Fold {fold+1}...")
        
        # Sample 4000 points from this fold's training data for TabPFN context
        fold_train_subset_idx = np.random.choice(train_idx, size=4000, replace=False)
        X_fold_train = X.iloc[fold_train_subset_idx]
        y_fold_train = y.iloc[fold_train_subset_idx]
        
        # Fit fresh TabPFN for this fold
        clf_fold = TabPFNClassifier(device='cuda', N_ensemble_configurations=4) # Faster config for OOF
        clf_fold.fit(X_fold_train, y_fold_train)
        
        # Predict on Validation set in batches
        X_val = X.iloc[val_idx]
        val_preds_fold = []
        
        for i in range(0, len(X_val), BATCH_SIZE):
            batch = X_val.iloc[i:i+BATCH_SIZE]
            preds = clf_fold.predict_proba(batch)[:, 1]
            val_preds_fold.extend(preds)
            
        oof_preds[val_idx] = val_preds_fold
        print(f"Fold {fold+1} AUC: {roc_auc_score(y.iloc[val_idx], val_preds_fold):.5f}")

    # 2. Test Predictions
    print("Generating Test Predictions...")
    # For Test, we can fit on a larger subset of full Train
    final_subset_idx = np.random.choice(len(X), size=8000, replace=False)
    clf_final = TabPFNClassifier(device='cuda', N_ensemble_configurations=8)
    clf_final.fit(X.iloc[final_subset_idx], y.iloc[final_subset_idx])
    
    test_preds = []
    for i in range(0, len(X_test), BATCH_SIZE):
        batch = X_test.iloc[i:i+BATCH_SIZE]
        preds = clf_final.predict_proba(batch)[:, 1]
        test_preds.extend(preds)
        if i % 10000 == 0:
            print(f"Processed {i}/{len(X_test)}")
            
    # Save Results
    os.makedirs('Models', exist_ok=True)
    pd.DataFrame({TARGET_COL: oof_preds}).to_csv('tabpfn_oof.csv', index=False)
    pd.DataFrame({'id': test_ids, TARGET_COL: test_preds}).to_csv('tabpfn_test.csv', index=False)
    print("Saved results to tabpfn_oof.csv and tabpfn_test.csv")
    
    try:
        from google.colab import files
        files.download('tabpfn_oof.csv')
        files.download('tabpfn_test.csv')
    except ImportError:
        pass

if __name__ == "__main__":
    main()
