# 1. Gerekli Kütüphanelerin Yüklenmesi
!pip install scikit-learn pandas numpy lightgbm

# 2. Scriptin Çalıştırılması
import os
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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

def feature_engineering_v1(df):
    df['employment_credit_ratio'] = df['employment_status'] / (df['credit_score'] + 1e-6)
    df['employment_grade_interaction'] = df['employment_status'] * df['grade_subgrade']
    df['employment_dti_interaction'] = df['employment_status'] * df['debt_to_income_ratio']
    return df

def apply_magic_features(df):
    # Top 5 Magic Features from Brute-Force
    # 1. debt_to_income_ratio / employment_dti_interaction
    df['debt_to_income_ratio_div_employment_dti_interaction'] = df['debt_to_income_ratio'] / (df['employment_dti_interaction'] + 1e-6)
    
    # 2. debt_to_income_ratio - employment_dti_interaction
    df['debt_to_income_ratio_minus_employment_dti_interaction'] = df['debt_to_income_ratio'] - df['employment_dti_interaction']
    
    # 3. debt_to_income_ratio + employment_grade_interaction
    df['debt_to_income_ratio_plus_employment_grade_interaction'] = df['debt_to_income_ratio'] + df['employment_grade_interaction']
    
    # 4. debt_to_income_ratio - employment_grade_interaction
    df['debt_to_income_ratio_minus_employment_grade_interaction'] = df['debt_to_income_ratio'] - df['employment_grade_interaction']
    
    # 5. credit_score / employment_credit_ratio
    df['credit_score_div_employment_credit_ratio'] = df['credit_score'] / (df['employment_credit_ratio'] + 1e-6)
    
    return df

def main():
    # 1. Load Data
    train, test = load_data()
    
    # 2. Feature Engineering
    print("Applying Features (V1 + Magic)...")
    train = feature_engineering_v1(train)
    test = feature_engineering_v1(test)
    
    train = apply_magic_features(train)
    test = apply_magic_features(test)
    
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
    print("\n--- Training MLPClassifier with Magic Features ---")
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
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
    print(f"\nMLP (Magic) OOF AUC: {auc_score:.5f}")
    
    # Save Results
    os.makedirs('Models', exist_ok=True)
    pd.DataFrame({TARGET_COL: oof_preds}).to_csv('oof_mlp_magic.csv', index=False)
    pd.DataFrame({'id': test_ids, TARGET_COL: test_preds}).to_csv('submission_mlp_magic.csv', index=False)
    print("Saved results to oof_mlp_magic.csv and submission_mlp_magic.csv")
    
    try:
        from google.colab import files
        files.download('oof_mlp_magic.csv')
        files.download('submission_mlp_magic.csv')
    except ImportError:
        pass

if __name__ == "__main__":
    main()
