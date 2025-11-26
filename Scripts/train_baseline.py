import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data():
    print("Loading data...")
    train = pd.read_csv('Processed/clean_train.csv')
    test = pd.read_csv('Processed/clean_test.csv')
    return train, test

def feature_engineering(df):
    print("Engineering features...")
    # 1. employment_credit_ratio
    # Adding epsilon to avoid division by zero if credit_score is 0 (unlikely but safe)
    df['employment_credit_ratio'] = df['employment_status'] / (df['credit_score'] + 1e-6)
    
    # 2. employment_grade_interaction
    df['employment_grade_interaction'] = df['employment_status'] * df['grade_subgrade']
    
    # 3. employment_dti_interaction
    df['employment_dti_interaction'] = df['employment_status'] * df['debt_to_income_ratio']
    
    return df

def train_model(train, test):
    print("Starting training...")
    
    # Prepare data
    X = train.drop('loan_paid_back', axis=1)
    y = train['loan_paid_back']
    X_test = test.drop('id', axis=1) # id is not a feature
    
    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    # LightGBM Parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'max_depth': 4,
        'colsample_bytree': 0.8,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'n_jobs': -1,
        'random_state': 42,
        'verbosity': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        
        # Early stopping callback
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0) # Suppress per-iteration logging
        ]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=callbacks
        )
        
        val_preds = model.predict_proba(X_val)[:, 1]
        oof_preds[val_idx] = val_preds
        
        fold_score = roc_auc_score(y_val, val_preds)
        print(f"Fold {fold+1} AUC: {fold_score:.5f}")
        
        test_preds += model.predict_proba(X_test)[:, 1] / skf.get_n_splits()
        
    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\nOverall OOF AUC Score: {overall_auc:.5f}")
    
    return test_preds, overall_auc

def create_submission(test, preds):
    print("Creating submission file...")
    submission = pd.DataFrame({
        'id': test['id'],
        'loan_paid_back': preds
    })
    submission.to_csv('submission.csv', index=False)
    print("submission.csv saved successfully.")

def main():
    train, test = load_data()
    
    train = feature_engineering(train)
    test = feature_engineering(test)
    
    # Ensure columns match (except target and id)
    # Align columns just in case, though preprocessing should have handled this
    # Using the columns from X in training
    
    preds, score = train_model(train, test)
    
    create_submission(test, preds)

if __name__ == "__main__":
    main()
