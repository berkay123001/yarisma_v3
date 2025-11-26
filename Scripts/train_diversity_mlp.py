import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os

# --- CONFIG ---
TARGET_COL = 'loan_paid_back'
N_SPLITS = 5
SEED = 42

def load_data():
    print("Loading data...")
    if os.path.exists('Processed/clean_train.csv'):
        train = pd.read_csv('Processed/clean_train.csv')
        test = pd.read_csv('Processed/clean_test.csv')
    elif os.path.exists('clean_train.csv'):
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
    else:
        raise FileNotFoundError("Could not find clean_train.csv")
    return train, test

def feature_engineering_v1(df):
    # 1. Employment / Credit Score Ratio
    df['employment_credit_ratio'] = df['employment_status'] / (df['credit_score'] + 1e-6)
    
    # 2. Employment * Grade Interaction
    df['employment_grade_interaction'] = df['employment_status'] * df['grade_subgrade']
    
    # 3. Employment * DTI Interaction
    df['employment_dti_interaction'] = df['employment_status'] * df['debt_to_income_ratio']
    
    return df

def main():
    # 1. Load Data
    train, test = load_data()
    
    # 2. Feature Engineering (V1)
    print("Applying V1 Feature Engineering...")
    train = feature_engineering_v1(train)
    test = feature_engineering_v1(test)
    
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
        
    print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")
    
    # 3. Data Preprocessing for NN (Impute & Scale)
    print("\nPreprocessing for Neural Network...")
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_test_imputed = imputer.transform(X_test)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # 4. Train MLPClassifier
    print("\n--- Training MLPClassifier ---")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        early_stopping=True,
        random_state=SEED,
        max_iter=500  # Ensure enough iterations
    )
    
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        mlp.fit(X_train, y_train)
        
        val_preds = mlp.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += mlp.predict_proba(X_test_scaled)[:, 1] / N_SPLITS
        
        fold_auc = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold+1} AUC: {fold_auc:.5f}")
        
    auc_score = roc_auc_score(y, oof_preds)
    print(f"\nMLP OOF AUC: {auc_score:.5f}")
    
    # Save OOF
    os.makedirs('Models', exist_ok=True)
    oof_df = pd.DataFrame({TARGET_COL: oof_preds})
    oof_df.to_csv('Models/oof_mlp.csv', index=False)
    print("Saved OOF to Models/oof_mlp.csv")
    
    # Save Submission
    sub_df = pd.DataFrame({'id': test_ids, TARGET_COL: test_preds})
    sub_df.to_csv('submission_mlp.csv', index=False)
    print("Saved Submission to submission_mlp.csv")
    
    # 5. Correlation Check
    print("\n--- Correlation Check ---")
    if os.path.exists('Models/oof_lgbm_optuna.csv'):
        lgbm_oof = pd.read_csv('Models/oof_lgbm_optuna.csv')[TARGET_COL]
        correlation = np.corrcoef(lgbm_oof, oof_preds)[0, 1]
        print(f"Correlation with LightGBM: {correlation:.5f}")
        
        if correlation < 0.90:
            print("SUCCESS: Low correlation! This model adds significant diversity.")
        elif correlation < 0.95:
            print("GOOD: Moderate correlation. Useful for stacking.")
        else:
            print("WARNING: High correlation. Diversity gain might be limited.")
    else:
        print("Could not find 'Models/oof_lgbm_optuna.csv' for correlation check.")

if __name__ == "__main__":
    main()
