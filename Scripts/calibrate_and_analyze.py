import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import os

# --- CONFIG ---
SUB_FILE = 'Submissions/submission_optimized_ensemble.csv'
OOF_FILE = 'Models/oof_optimized_ensemble.csv'
TRAIN_FILE = 'Processed/clean_train.csv'
TARGET_COL = 'loan_paid_back'

def analyze_distribution(preds, name="Predictions"):
    print(f"\n--- Distribution Analysis: {name} ---")
    hist, bins = np.histogram(preds, bins=10, range=(0, 1))
    print("Bin Range | Count | %")
    print("-" * 30)
    total = len(preds)
    for i in range(len(hist)):
        print(f"{bins[i]:.1f}-{bins[i+1]:.1f}   | {hist[i]:5d} | {hist[i]/total*100:.1f}%")
        
    print(f"Min: {preds.min():.4f}, Max: {preds.max():.4f}, Mean: {preds.mean():.4f}")

def main():
    # 1. Load Data
    print("Loading Data...")
    sub_df = pd.read_csv(SUB_FILE)
    oof_df = pd.read_csv(OOF_FILE)
    
    if os.path.exists(TRAIN_FILE):
        train_df = pd.read_csv(TRAIN_FILE)
    else:
        train_df = pd.read_csv('clean_train.csv')
        
    y_true = train_df[TARGET_COL].values
    oof_preds = oof_df[TARGET_COL].values
    test_preds = sub_df[TARGET_COL].values
    
    # 2. Analyze Original Distribution
    analyze_distribution(test_preds, "Original Submission")
    
    # 3. Train Calibrator (Isotonic Regression)
    print("\n--- Training Isotonic Calibrator ---")
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_preds, y_true)
    
    # Check OOF improvement
    calibrated_oof = iso.transform(oof_preds)
    original_auc = roc_auc_score(y_true, oof_preds)
    calibrated_auc = roc_auc_score(y_true, calibrated_oof)
    
    print(f"Original OOF AUC:   {original_auc:.5f}")
    print(f"Calibrated OOF AUC: {calibrated_auc:.5f}")
    
    if calibrated_auc > original_auc:
        print("✅ Calibration improved OOF score!")
    else:
        print("⚠️ Calibration did not improve score (Isotonic is monotonic, so AUC shouldn't change much, but probabilities will be better).")
        
    # 4. Calibrate Submission
    print("\n--- Calibrating Submission ---")
    calibrated_test_preds = iso.transform(test_preds)
    
    analyze_distribution(calibrated_test_preds, "Calibrated Submission")
    
    # 5. Save
    sub_df[TARGET_COL] = calibrated_test_preds
    out_file = 'Submissions/submission_calibrated.csv'
    sub_df.to_csv(out_file, index=False)
    print(f"\nSaved calibrated submission to {out_file}")

if __name__ == "__main__":
    main()
