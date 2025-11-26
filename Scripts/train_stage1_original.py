import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import roc_auc_score
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
ORIGINAL_DATA_PATH = 'Datasets/Original_Data/loan_dataset_20000.csv'
TRAIN_PATH = 'Processed/clean_train.csv'
TEST_PATH = 'Processed/clean_test.csv'
TARGET_COL = 'loan_paid_back'
SEED = 42

# LightGBM Params for Stage 1 (Simple but robust)
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'n_estimators': 1000,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': -1
}

def load_data():
    print("Loading data...")
    if not os.path.exists(ORIGINAL_DATA_PATH):
        raise FileNotFoundError(f"Original data not found at {ORIGINAL_DATA_PATH}")
    
    original = pd.read_csv(ORIGINAL_DATA_PATH)
    
    if os.path.exists(TRAIN_PATH):
        train = pd.read_csv(TRAIN_PATH)
        test = pd.read_csv(TEST_PATH)
    else:
        # Fallback if processed data not found
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
        
    return original, train, test

def preprocess_original(df):
    # Basic preprocessing to match competition data format
    # We need to encode categoricals similarly
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    # Simple Ordinal Encoding
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_cols] = enc.fit_transform(df[cat_cols])
    
    return df, enc, cat_cols

def main():
    # 1. Load Data
    original, train, test = load_data()
    
    print(f"Original Data Shape: {original.shape}")
    print(f"Competition Train Shape: {train.shape}")
    
    # 2. Preprocess Original Data
    # We need to ensure columns match. 
    # Competition data 'clean_train.csv' is already encoded.
    # We must encode 'original' data.
    
    # Check for column mismatch
    common_cols = [c for c in original.columns if c in train.columns and c != TARGET_COL]
    print(f"Common Features: {len(common_cols)}")
    
    # Encode Original Data
    # Note: Ideally we should fit encoder on combined data, but for Stage 1 it's okay to fit on Original
    # or just use the same mapping if we knew it.
    # Since 'clean_train.csv' is already numerical, we assume it used Ordinal Encoding.
    # Let's apply Ordinal Encoding to Original Data's object columns.
    
    cat_cols = original.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print(f"Encoding categorical columns in Original Data: {list(cat_cols)}")
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        original[cat_cols] = enc.fit_transform(original[cat_cols])
    
    # 3. Train Stage 1 Model on ORIGINAL Data
    print("\n--- Training Stage 1 Model on Original Data ---")
    X_orig = original[common_cols]
    y_orig = original[TARGET_COL]
    
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_orig, y_orig)
    
    score = roc_auc_score(y_orig, model.predict_proba(X_orig)[:, 1])
    print(f"Stage 1 Model Training AUC: {score:.5f}")
    
    # 4. Predict on Competition Data (Train & Test)
    print("\nGenerating Stage 1 Predictions (Residual Feature)...")
    
    # Predict on Competition Train
    stage1_train_preds = model.predict_proba(train[common_cols])[:, 1]
    
    # Predict on Competition Test
    stage1_test_preds = model.predict_proba(test[common_cols])[:, 1]
    
    # 5. Save Predictions
    os.makedirs('Models', exist_ok=True)
    
    # Save as simple CSVs to be loaded later
    pd.DataFrame({'stage1_pred': stage1_train_preds}).to_csv('Models/stage1_original_train.csv', index=False)
    pd.DataFrame({'stage1_pred': stage1_test_preds}).to_csv('Models/stage1_original_test.csv', index=False)
    
    print("Saved Stage 1 predictions to Models/stage1_original_train.csv and Models/stage1_original_test.csv")

if __name__ == "__main__":
    main()
