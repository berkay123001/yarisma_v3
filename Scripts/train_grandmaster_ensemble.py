import pandas as pd
import os

# --- CONFIG ---
# Weights
W_BRUTE = 0.50
W_BOOST = 0.35
W_TABPFN = 0.15

# File Paths
FILE_BRUTE = 'Submissions/submission_bruteforce.csv'
FILE_BOOST = 'Submissions/submission_boost_residual.csv'
FILE_TABPFN = 'Models/tabpfn_test.csv'

OUTPUT_FILE = 'Submissions/submission_grandmaster_ensemble.csv'
TARGET_COL = 'loan_paid_back'

def main():
    print("--- Creating Grandmaster Ensemble ---")
    print(f"Weights: Brute={W_BRUTE}, Boost={W_BOOST}, TabPFN={W_TABPFN}")
    
    # Load Submissions
    if not os.path.exists(FILE_BRUTE):
        raise FileNotFoundError(f"Missing {FILE_BRUTE}")
    if not os.path.exists(FILE_BOOST):
        raise FileNotFoundError(f"Missing {FILE_BOOST}")
    if not os.path.exists(FILE_TABPFN):
        raise FileNotFoundError(f"Missing {FILE_TABPFN}")
        
    df_brute = pd.read_csv(FILE_BRUTE)
    df_boost = pd.read_csv(FILE_BOOST)
    df_tabpfn = pd.read_csv(FILE_TABPFN)
    
    # Verify IDs match
    if not (df_brute['id'].equals(df_boost['id']) and df_brute['id'].equals(df_tabpfn['id'])):
        raise ValueError("ID mismatch between submission files!")
        
    # Weighted Average
    print("Calculating Weighted Average...")
    final_preds = (
        (df_brute[TARGET_COL] * W_BRUTE) +
        (df_boost[TARGET_COL] * W_BOOST) +
        (df_tabpfn[TARGET_COL] * W_TABPFN)
    )
    
    # Create Submission
    submission = pd.DataFrame({
        'id': df_brute['id'],
        TARGET_COL: final_preds
    })
    
    # Save
    os.makedirs('Submissions', exist_ok=True)
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved Grandmaster Ensemble to {OUTPUT_FILE}")
    
    # Preview
    print("\nPreview:")
    print(submission.head())

if __name__ == "__main__":
    main()
