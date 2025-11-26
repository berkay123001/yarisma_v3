# 1. AutoGluon Kurulumu
!pip install autogluon

# 2. Scriptin Çalıştırılması
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
TARGET_COL = 'loan_paid_back'
TIME_LIMIT = 600 # 10 dakika yeterli (Veri küçük)
PRESETS = 'best_quality' # En iyi model kalitesi için

def load_data():
    print("Loading data...")
    # Dosya yollarını Colab'a göre ayarla
    # Kullanıcıdan bu dosyaları yüklemesini isteyeceğiz
    if os.path.exists('loan_dataset_20000.csv'):
        original = pd.read_csv('loan_dataset_20000.csv')
    else:
        raise FileNotFoundError("Lütfen 'loan_dataset_20000.csv' dosyasını yükleyin!")
        
    if os.path.exists('clean_train.csv'):
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
    else:
        raise FileNotFoundError("Lütfen 'clean_train.csv' ve 'clean_test.csv' dosyalarını yükleyin!")
        
    return original, train, test

def align_features(original, train, test):
    # AutoGluon kategorik verileri otomatik halleder ama sütun isimleri aynı olmalı.
    # Original verideki sütunlar ile yarışma verisindeki sütunların kesişimini alalım.
    
    common_cols = [c for c in original.columns if c in train.columns and c != TARGET_COL]
    print(f"Common Features: {len(common_cols)}")
    print(f"Features: {common_cols}")
    
    return original, train[common_cols], test[common_cols], common_cols

def main():
    # 1. Load Data
    original, train, test = load_data()
    
    # 2. Align Features
    original, X_train_comp, X_test_comp, features = align_features(original, train, test)
    
    # 3. Train AutoGluon on ORIGINAL Data
    print("\n--- Training AutoGluon on Original Data ---")
    
    # AutoGluon için veriyi hazırla
    train_data = original[features + [TARGET_COL]]
    
    predictor = TabularPredictor(label=TARGET_COL, eval_metric='roc_auc').fit(
        train_data,
        time_limit=TIME_LIMIT,
        presets=PRESETS,
        ag_args_fit={'num_gpus': 1} # GPU varsa kullan
    )
    
    # 4. Predict on Competition Data
    print("\nGenerating Stage 1 Predictions (Residual Priors)...")
    
    # Predict on Competition Train (OOF mantığı değil, direkt tahmin çünkü farklı veri seti)
    # Bu tahminler "Stage 1" özelliği olacak.
    stage1_train_preds = predictor.predict_proba(X_train_comp)[1]
    
    # Predict on Competition Test
    stage1_test_preds = predictor.predict_proba(X_test_comp)[1]
    
    # 5. Save Results
    os.makedirs('Models', exist_ok=True)
    
    pd.DataFrame({'stage1_ag_pred': stage1_train_preds}).to_csv('stage1_autogluon_train.csv', index=False)
    pd.DataFrame({'stage1_ag_pred': stage1_test_preds}).to_csv('stage1_autogluon_test.csv', index=False)
    
    print("Saved results to stage1_autogluon_train.csv and stage1_autogluon_test.csv")
    
    # Leaderboard göster
    print("\nAutoGluon Leaderboard:")
    print(predictor.leaderboard(silent=True))
    
    try:
        from google.colab import files
        files.download('stage1_autogluon_train.csv')
        files.download('stage1_autogluon_test.csv')
    except ImportError:
        pass

if __name__ == "__main__":
    main()
