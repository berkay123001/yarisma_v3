import pandas as pd
import numpy as np
import lightgbm as lgb
from gplearn.genetic import SymbolicTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
TARGET_COL = 'loan_paid_back'
N_SPLITS = 5
SEED = 42

# Genetic Programming Params
GP_PARAMS = {
    'population_size': 2000,
    'generations': 10,
    'tournament_size': 20,
    'stopping_criteria': 1.0,
    'const_range': (-1.0, 1.0),
    'init_depth': (2, 6),
    'init_method': 'half and half',
    'function_set': ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'abs', 'neg', 'inv'],
    'metric': 'pearson',
    'p_crossover': 0.7,
    'p_subtree_mutation': 0.1,
    'p_hoist_mutation': 0.05,
    'p_point_mutation': 0.1,
    'max_samples': 0.9,
    'verbose': 1,
    'parsimony_coefficient': 0.001,
    'random_state': SEED,
    'n_jobs': -1  # Use all cores
}

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

def evolve_features(train, test):
    print("\n--- Starting Genetic Feature Evolution ---")
    
    # Prepare Data for GP (Numeric Only)
    # Drop ID and Target
    drop_cols = ['id', TARGET_COL]
    if TARGET_COL in train.columns:
        X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
        y_train = train[TARGET_COL]
    else:
        raise ValueError("Target column missing in train!")
        
    X_test = test.drop(columns=['id']) if 'id' in test.columns else test.copy()
    
    # Align columns
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    # Select only numeric columns
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    print(f"Using {len(numeric_cols)} numeric features for evolution.")
    
    X_train_num = X_train[numeric_cols]
    X_test_num = X_test[numeric_cols]
    
    # Impute NaNs (GP doesn't like NaNs)
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_num)
    X_test_imp = imputer.transform(X_test_num)
    
    # Initialize SymbolicTransformer
    gp = SymbolicTransformer(**GP_PARAMS)
    
    # Fit
    print("Evolving features (this may take a while)...")
    gp.fit(X_train_imp, y_train)
    
    # Transform
    print("Generating new features...")
    new_features_train = gp.transform(X_train_imp)
    new_features_test = gp.transform(X_test_imp)
    
    print(f"Generated {new_features_train.shape[1]} new genetic features.")
    
    # Print Top Formulas
    print("\n--- Top Generated Formulas ---")
    for i, program in enumerate(gp._best_programs):
        if program is not None:
            print(f"Feature {i}: {program}")
            
    # Add to DataFrames
    for i in range(new_features_train.shape[1]):
        col_name = f'genetic_{i}'
        train[col_name] = new_features_train[:, i]
        test[col_name] = new_features_test[:, i]
        
    return train, test

def main():
    # 1. Load Data
    train, test = load_data()
    
    # 2. Evolve Features
    train, test = evolve_features(train, test)
    
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
    
    # 4. Train LightGBM
    print("\n--- Training LightGBM with Genetic Features ---")
    
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
    print(f"\nFinal Genetic OOF AUC: {auc_score:.5f}")
    
    # Save Submission
    os.makedirs('Submissions', exist_ok=True)
    sub_df = pd.DataFrame({'id': test_ids, TARGET_COL: test_preds})
    sub_df.to_csv('Submissions/submission_genetic.csv', index=False)
    print("Saved Submission to Submissions/submission_genetic.csv")
    
    # Save OOF
    os.makedirs('Models', exist_ok=True)
    pd.DataFrame({TARGET_COL: oof_preds}).to_csv('Models/oof_genetic.csv', index=False)

if __name__ == "__main__":
    main()
