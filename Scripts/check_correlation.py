import pandas as pd
import os

# --- CONFIG ---
FILES = {
    'Brute-Force (0.92360)': 'Submissions/submission_bruteforce.csv',
    'Boost Residual (0.92333)': 'Submissions/submission_boost_residual.csv',
    'TabPFN (Joker)': 'Models/tabpfn_test.csv'
}

TARGET_COL = 'loan_paid_back'

def main():
    print("--- Correlation Analysis ---")
    
    preds = {}
    for name, path in FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            preds[name] = df[TARGET_COL]
        else:
            print(f"Warning: {path} not found!")
            
    if not preds:
        print("No files found to analyze.")
        return
        
    df_corr = pd.DataFrame(preds)
    
    # Calculate Correlation
    corr_matrix = df_corr.corr()
    
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Calculate Diversity (1 - Correlation)
    print("\nDiversity (Lower correlation is better for ensembling):")
    print(f"Brute vs Boost: {corr_matrix.iloc[0, 1]:.5f}")
    print(f"Brute vs TabPFN: {corr_matrix.iloc[0, 2]:.5f}")
    print(f"Boost vs TabPFN: {corr_matrix.iloc[1, 2]:.5f}")

if __name__ == "__main__":
    main()
