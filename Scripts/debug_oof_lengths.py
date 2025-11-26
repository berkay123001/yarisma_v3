import pandas as pd
import os

MODELS = {
    'Brute': 'Models/oof_lgbm_bruteforce.csv',
    'DAE': 'Models/oof_dae_boosted.csv',
    'Genetic': 'Models/oof_genetic.csv',
    'AutoGluon': 'Models/oof_boost_autogluon.csv',
    'Patched': 'Models/oof_patched_bruteforce.csv',
    'MLP': 'Models/oof_mlp_magic.csv'
}

def main():
    print("Checking OOF Lengths...")
    for name, path in MODELS.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"{name}: {len(df)} rows")
        else:
            print(f"{name}: File not found")

if __name__ == "__main__":
    main()
