import pandas as pd
import os

# --- CONFIG ---
# Top 3 Models
FILE_BRUTE = 'Submissions/submission_bruteforce.csv' # 0.92360
FILE_DAE = 'Submissions/submission_dae_boosted.csv' # 0.92360
FILE_GENETIC = 'Submissions/submission_genetic.csv' # 0.92349

# Weights
# Brute and DAE are tied for top, so they get equal high weight.
# Genetic is slightly behind but adds unique non-linear diversity.
W_BRUTE = 0.40
W_DAE = 0.40
W_GENETIC = 0.20

OUTPUT_FILE = 'Submissions/submission_mega_ensemble.csv'
TARGET_COL = 'loan_paid_back'

def main():
    print("--- Creating Mega Ensemble ---")
    print(f"Brute-Force (0.92360): {W_BRUTE}")
    print(f"DAE-Boosted (0.92360): {W_DAE}")
    print(f"Genetic (0.92349): {W_GENETIC}")
    
    if not os.path.exists(FILE_BRUTE):
        raise FileNotFoundError(f"Missing {FILE_BRUTE}")
    if not os.path.exists(FILE_DAE):
        raise FileNotFoundError(f"Missing {FILE_DAE}")
    if not os.path.exists(FILE_GENETIC):
        raise FileNotFoundError(f"Missing {FILE_GENETIC}")
        
    df_brute = pd.read_csv(FILE_BRUTE)
    df_dae = pd.read_csv(FILE_DAE)
    df_genetic = pd.read_csv(FILE_GENETIC)
    
    # Verify IDs
    if not (df_brute['id'].equals(df_dae['id']) and df_brute['id'].equals(df_genetic['id'])):
        raise ValueError("ID mismatch between submission files!")
        
    # Weighted Average
    print("Calculating Weighted Average...")
    final_preds = (
        (df_brute[TARGET_COL] * W_BRUTE) +
        (df_dae[TARGET_COL] * W_DAE) +
        (df_genetic[TARGET_COL] * W_GENETIC)
    )
    
    # Save
    submission = pd.DataFrame({
        'id': df_brute['id'],
        TARGET_COL: final_preds
    })
    
    os.makedirs('Submissions', exist_ok=True)
    submission.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved Mega Ensemble to {OUTPUT_FILE}")
    
    # Preview
    print("\nPreview:")
    print(submission.head())

if __name__ == "__main__":
    main()
