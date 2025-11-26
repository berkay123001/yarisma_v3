import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
TARGET_COL = 'loan_paid_back'
N_SPLITS = 5
SEED = 42
KNN_NEIGHBORS = 500

# LightGBM Params (Best from Optuna)
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.0426,
    'num_leaves': 59,
    'max_depth': 6,
    'reg_alpha': 0.468,
    'reg_lambda': 9.56,
    'min_child_samples': 65,
    'subsample': 0.955,
    'n_estimators': 3000,
    'n_jobs': -1,
    'device': 'cpu',
    'verbosity': -1,
    'random_state': SEED
}

def load_data():
    print("Loading data...")
    if os.path.exists('Processed/clean_train.csv'):
        train = pd.read_csv('Processed/clean_train.csv')
        test = pd.read_csv('Processed/clean_test.csv')
    else:
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
    return train, test

def apply_domain_features(df):
    print("Applying Domain Features...")
    # 1. Disposable Income Proxy
    # Assuming standard living costs, how much is left?
    # We don't have exact debt amount, but we have DTI.
    # Debt = Annual Income * DTI
    df['estimated_debt'] = df['annual_income'] * df['debt_to_income_ratio']
    df['disposable_income'] = df['annual_income'] - df['estimated_debt']
    
    # 2. Credit Utilization Proxy
    # Loan Amount / Credit Score (Higher loan with lower score is risky)
    df['credit_utilization'] = df['loan_amount'] / (df['credit_score'] + 1e-6)
    
    # 3. Payment Burden
    # We don't have monthly payment, but we can approximate.
    # Loan / Term (assuming 36 or 60 months, but we don't have term column in clean data? 
    # Wait, original data had 'loan_term'. Let's check if it's in clean data.
    if 'loan_term' in df.columns:
        df['payment_burden'] = df['loan_amount'] / df['loan_term'] / (df['annual_income'] / 12 + 1e-6)
    else:
        # Fallback: Loan / Income
        df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1e-6)
        
    # 4. Interest Burden
    df['interest_burden'] = df['loan_amount'] * df['interest_rate']
    
    return df

def apply_knn_features(train, test):
    print(f"Applying KNN Features (k={KNN_NEIGHBORS})...")
    
    # Select numeric features for distance calculation
    numeric_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
    numeric_cols = [c for c in numeric_cols if c in train.columns]
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[numeric_cols])
    X_test_scaled = scaler.transform(test[numeric_cols])
    
    # Fit KNN
    knn = NearestNeighbors(n_neighbors=KNN_NEIGHBORS, n_jobs=-1)
    knn.fit(X_train_scaled)
    
    # Find Neighbors for Train (excluding self is tricky in sklearn without custom loop, 
    # but with k=500, self-inclusion noise is minimal. Let's do it properly though.)
    
    # Actually, for Train, we should use OOF-like approach to avoid leakage.
    # But that's expensive. A simpler way is to find k+1 neighbors and exclude the first one (self).
    
    # Train Neighbors
    dists, idxs = knn.kneighbors(X_train_scaled, n_neighbors=KNN_NEIGHBORS+1)
    # Exclude self (first column)
    idxs = idxs[:, 1:]
    
    # Calculate Mean Target of Neighbors
    y_train = train[TARGET_COL].values
    neighbor_targets = y_train[idxs]
    train['knn_mean_target'] = neighbor_targets.mean(axis=1)
    
    # Test Neighbors
    dists_test, idxs_test = knn.kneighbors(X_test_scaled, n_neighbors=KNN_NEIGHBORS)
    neighbor_targets_test = y_train[idxs_test]
    test['knn_mean_target'] = neighbor_targets_test.mean(axis=1)
    
    return train, test

def apply_target_encoding(train, test):
    print("Applying Smoothed Target Encoding...")
    cat_cols = ['grade_subgrade', 'loan_purpose', 'employment_status', 'education_level', 'marital_status']
    cat_cols = [c for c in cat_cols if c in train.columns]
    
    # Target Encoder with Smoothing
    encoder = ce.TargetEncoder(cols=cat_cols, smoothing=100, min_samples_leaf=100)
    
    # Fit on Train
    encoder.fit(train[cat_cols], train[TARGET_COL])
    
    # Transform
    train_encoded = encoder.transform(train[cat_cols])
    test_encoded = encoder.transform(test[cat_cols])
    
    # Rename columns
    train_encoded.columns = [f"{c}_target_enc" for c in cat_cols]
    test_encoded.columns = [f"{c}_target_enc" for c in cat_cols]
    
    train = pd.concat([train, train_encoded], axis=1)
    test = pd.concat([test, test_encoded], axis=1)
    
    return train, test

def main():
    # 1. Load Data
    train, test = load_data()
    
    # 2. Domain Features
    train = apply_domain_features(train)
    test = apply_domain_features(test)
    
    # 3. KNN Features
    train, test = apply_knn_features(train, test)
    
    # 4. Target Encoding
    train, test = apply_target_encoding(train, test)
    
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
        
    print(f"\nFinal Feature Count: {X.shape[1]}")
    print(f"New Features: {[c for c in X.columns if c not in ['annual_income', 'debt_to_income_ratio']]}") # Just printing some
    
    # 5. Train LightGBM
    print("\n--- Training Golden LightGBM ---")
    
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=callbacks
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
        
        print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds):.5f}")
        
    auc_score = roc_auc_score(y, oof_preds)
    print(f"\nFinal Golden OOF AUC: {auc_score:.5f}")
    
    # Save Submission
    os.makedirs('Submissions', exist_ok=True)
    sub_df = pd.DataFrame({'id': test_ids, TARGET_COL: test_preds})
    sub_df.to_csv('Submissions/submission_golden.csv', index=False)
    print("Saved Submission to Submissions/submission_golden.csv")
    
    # Save OOF
    os.makedirs('Models', exist_ok=True)
    pd.DataFrame({TARGET_COL: oof_preds}).to_csv('Models/oof_golden.csv', index=False)

if __name__ == "__main__":
    main()
