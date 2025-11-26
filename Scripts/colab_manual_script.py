# 1. Gerekli Kütüphanelerin Yüklenmesi
!pip install optuna lightgbm scikit-learn pandas

# 2. Scriptin Çalıştırılması
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# --- AYARLAR ---
N_TRIALS = 20  # Deneme sayısı
TARGET_COL = 'loan_paid_back'

# --- VERİ YÜKLEME ---
def load_data():
    print("Loading data...")
    # Colab için dosya yolları kontrolü
    if os.path.exists('/content/clean_train.csv'):
        train = pd.read_csv('/content/clean_train.csv')
        test = pd.read_csv('/content/clean_test.csv')
    elif os.path.exists('clean_train.csv'):
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
    else:
        raise FileNotFoundError("Lütfen 'clean_train.csv' ve 'clean_test.csv' dosyalarını Colab'a yükleyin!")
    return train, test

# --- FEATURE ENGINEERING (V1 - Magic Features) ---
def feature_engineering_v1(df):
    # 1. Employment / Credit Score Ratio
    df['employment_credit_ratio'] = df['employment_status'] / (df['credit_score'] + 1e-6)
    
    # 2. Employment * Grade Interaction
    df['employment_grade_interaction'] = df['employment_status'] * df['grade_subgrade']
    
    # 3. Employment * DTI Interaction
    df['employment_dti_interaction'] = df['employment_status'] * df['debt_to_income_ratio']
    
    return df

# --- OPTUNA OBJECTIVE ---
def objective(trial, X, y):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=callbacks
        )
        
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        
    return roc_auc_score(y, oof_preds)

# --- ANA ÇALIŞTIRMA FONKSİYONU ---
def main():
    # 1. Veri Yükleme
    train, test = load_data()
    
    # 2. Feature Engineering
    print("Applying V1 Feature Engineering...")
    train = feature_engineering_v1(train)
    test = feature_engineering_v1(test)
    
    X = train.drop(TARGET_COL, axis=1)
    y = train[TARGET_COL]
    
    # 3. Optuna Optimizasyonu
    print(f"Starting Optuna Optimization ({N_TRIALS} Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS)
    
    print("\nBest Trial:")
    print(f"  Value: {study.best_value:.5f}")
    print("  Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
        
    # 4. En İyi Parametreleri Kaydetme
    os.makedirs('Models', exist_ok=True)
    with open('Models/best_params.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
        
    # 5. Final Modeli Eğitme ve Submission Oluşturma
    print("\nRetraining with best params...")
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'n_estimators': 1000,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    })
    
    # Tüm veriyle eğit (veya yine CV yapıp ortalamasını alabilirsin, burada basitlik için tek model)
    # Ancak OOF skoru ile tutarlılık için yine CV yapıp ortalama almak daha iyidir ama
    # Submission için genelde tüm veriyle eğitilir.
    # Biz burada tüm veriyle eğitelim:
    
    model = lgb.LGBMClassifier(**best_params)
    model.fit(X, y)
    
    submission_preds = model.predict_proba(test)[:, 1]
    
    submission = pd.DataFrame({
        'id': test['id'],
        'loan_paid_back': submission_preds
    })
    
    submission.to_csv('submission_optuna.csv', index=False)
    print("submission_optuna.csv created successfully!")
    
    # Dosyaları indirme (Colab'dan local'e)
    try:
        from google.colab import files
        files.download('submission_optuna.csv')
        files.download('Models/best_params.json')
    except ImportError:
        print("Not running in Colab UI, skipping file download.")

if __name__ == "__main__":
    main()
