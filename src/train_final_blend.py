import pandas as pd
import os

# --- CONFIG ---
# Top 2 Models
FILE_1 = 'Submissions/submission_bruteforce.csv' # 0.92360
FILE_2 = 'Submissions/submission_dream_ensemble.csv' # 0.92341

# Weights (Favoring the higher score slightly)
W1 = 0.60
W2 = 0.40

OUTPUT_FILE = 'Submissions/submission_final_blend.csv'
TARGET_COL = 'loan_paid_back'

def main():
    print("--- Creating Final Blend ---")
    print(f"Model 1: {FILE_1} (Weight: {W1})")
    print(f"Model 2: {FILE_2} (Weight: {W2})")
    
    if not os.path.exists(FILE_1):
        raise FileNotFoundError(f"Missing {FILE_1}")
    if not os.path.exists(FILE_2):
        raise FileNotFoundError(f"Missing {FILE_2}")
        
    df1 = pd.read_csv(FILE_1)
    df2 = pd.read_csv(FILE_2)
    
    # Verify IDs
    if not df1['id'].equals(df2['id']):
        raise ValueError("ID mismatch!")
        
    # Weighted Average
    final_preds = (df1[TARGET_COL] * W1) + (df2[TARGET_COL] * W2)
    
    # Save
    submission = pd.DataFrame({
        'id': df1['id'],
        TARGET_COL: final_preds
    })
    
    os.makedirs('Submissions', exist_ok=True)
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved Final Blend to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
