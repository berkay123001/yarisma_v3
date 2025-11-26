# 1. Gerekli Kütüphanelerin Yüklenmesi
!pip install scikit-learn pandas numpy lightgbm

# 2. Scriptin Çalıştırılması
import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
TARGET_COL = 'loan_paid_back'
N_SPLITS = 5
SEED = 42

def load_data():
    print("Loading data...")
    if os.path.exists('/content/clean_train.csv'):
        train = pd.read_csv('/content/clean_train.csv')
        test = pd.read_csv('/content/clean_test.csv')
    elif os.path.exists('clean_train.csv'):
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
    else:
        raise FileNotFoundError("Lütfen 'clean_train.csv' ve 'clean_test.csv' dosyalarını Colab'a yükleyin!")
    return train, test

def target_encoding(train, test, cat_cols, target_col, n_splits=5):
    print("Applying Target Encoding...")
    for col in cat_cols:
        train[f'{col}_target_enc'] = 0.0
        test[f'{col}_target_enc'] = 0.0

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    
    for train_idx, val_idx in kf.split(train):
        X_train, X_val = train.iloc[train_idx], train.iloc[val_idx]
        for col in cat_cols:
            means = X_train.groupby(col)[target_col].mean()
            train.loc[val_idx, f'{col}_target_enc'] = X_val[col].map(means)
            
    global_mean = train[target_col].mean()
    for col in cat_cols:
        train[f'{col}_target_enc'].fillna(global_mean, inplace=True)
        means = train.groupby(col)[target_col].mean()
        test[f'{col}_target_enc'] = test[col].map(means)
        test[f'{col}_target_enc'].fillna(global_mean, inplace=True)
        
    return train, test

def aggregations(df):
    print("Applying Aggregations...")
    df['mean_income_by_grade'] = df.groupby('grade_subgrade')['annual_income'].transform('mean')
    df['income_div_mean_grade'] = df['annual_income'] / (df['mean_income_by_grade'] + 1)
    df['max_dti_by_emp'] = df.groupby('employment_status')['debt_to_income_ratio'].transform('max')
    df['dti_div_max_emp'] = df['debt_to_income_ratio'] / (df['max_dti_by_emp'] + 1)
    df['mean_credit_by_purpose'] = df.groupby('loan_purpose')['credit_score'].transform('mean')
    return df

def isolation_forest_anomaly(train, test, features):
    print("Applying Isolation Forest Anomaly Detection...")
    combined = pd.concat([train[features], test[features]], axis=0)
    combined = combined.fillna(combined.mean())
    iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=SEED, n_jobs=-1)
    iso.fit(combined)
    train['anomaly_score'] = iso.decision_function(train[features])
    test['anomaly_score'] = iso.decision_function(test[features])
    return train, test

def main():
    # 1. Load Data
    train, test = load_data()
    
    # 2. Advanced Feature Engineering (Same as Phase A)
    train['employment_credit_ratio'] = train['employment_status'] / (train['credit_score'] + 1e-6)
    test['employment_credit_ratio'] = test['employment_status'] / (test['credit_score'] + 1e-6)
    
    cat_cols_to_enc = ['grade_subgrade', 'loan_purpose', 'employment_status']
    train, test = target_encoding(train, test, cat_cols_to_enc, TARGET_COL)
    
    train = aggregations(train)
    test = aggregations(test)
    
    iso_features = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
    train, test = isolation_forest_anomaly(train, test, iso_features)
    
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
    
    # 3. Preprocessing for NN
    print("\nPreprocessing for Neural Network...")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # 4. Train MLPClassifier
    print("\n--- Training MLPClassifier (Neural Network) ---")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), # Deeper architecture
        activation='relu',
        solver='adam',
        early_stopping=True,
        validation_fraction=0.1,
        random_state=SEED,
        max_iter=500,
        learning_rate_init=0.001
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
        
        print(f"Fold {fold+1} AUC: {roc_auc_score(y_val, val_preds):.5f}")
        
    auc_score = roc_auc_score(y, oof_preds)
    print(f"\nMLP OOF AUC: {auc_score:.5f}")
    
    # Save Results
    os.makedirs('Models', exist_ok=True)
    pd.DataFrame({TARGET_COL: oof_preds}).to_csv('oof_mlp_advanced.csv', index=False)
    pd.DataFrame({'id': test_ids, TARGET_COL: test_preds}).to_csv('submission_mlp_advanced.csv', index=False)
    print("Saved results to oof_mlp_advanced.csv and submission_mlp_advanced.csv")
    
    # Download files
    try:
        from google.colab import files
        files.download('oof_mlp_advanced.csv')
        files.download('submission_mlp_advanced.csv')
    except ImportError:
        print("Not running in Colab UI, skipping download.")

if __name__ == "__main__":
    main()
