import pandas as pd
import numpy as np
import os

# --- CONFIG ---
# Optimal Weights from Hill Climbing
WEIGHTS = {
    'Patched': 0.2959,
    'Genetic': 0.2857,
    'AutoGluon': 0.1837,
    'Brute': 0.1224,
    'DAE': 0.1122
}

FILES = {
    'Patched': 'Models/oof_patched_bruteforce.csv',
    'Genetic': 'Models/oof_genetic.csv',
    'AutoGluon': 'Models/oof_boost_autogluon.csv',
    'Brute': 'Models/oof_lgbm_bruteforce.csv',
    'DAE': 'Models/oof_dae_boosted.csv'
}

TARGET_COL = 'loan_paid_back'

def main():
    print("Generating Optimized OOF...")
    
    final_oof = None
    
    for name, weight in WEIGHTS.items():
        path = FILES[name]
        if os.path.exists(path):
            print(f" - Adding {name} (Weight: {weight})")
            df = pd.read_csv(path)
            
            # Handle column names
            if TARGET_COL in df.columns:
                preds = df[TARGET_COL].values
            else:
                preds = df.iloc[:, 0].values
                
            if final_oof is None:
                final_oof = preds * weight
            else:
                final_oof += preds * weight
        else:
            raise FileNotFoundError(f"Missing {path}")
            
    # Save
    os.makedirs('Models', exist_ok=True)
    pd.DataFrame({TARGET_COL: final_oof}).to_csv('Models/oof_optimized_ensemble.csv', index=False)
    print("Saved Models/oof_optimized_ensemble.csv")

if __name__ == "__main__":
    main()
