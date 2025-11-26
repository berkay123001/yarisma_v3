import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OrdinalEncoder
import os
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
TRAIN_PATH = 'Processed/clean_train.csv'
OOF_PATH = 'Models/oof_lgbm_bruteforce.csv' # Using our best model's OOF
TARGET_COL = 'loan_paid_back'
OUTPUT_REPORT = 'Analysis/error_patterns.md'

def load_data():
    print("Loading data...")
    if not os.path.exists(TRAIN_PATH):
        # Fallback
        if os.path.exists('clean_train.csv'):
            train = pd.read_csv('clean_train.csv')
        else:
            raise FileNotFoundError(f"Train data not found at {TRAIN_PATH}")
    else:
        train = pd.read_csv(TRAIN_PATH)
        
    if not os.path.exists(OOF_PATH):
        raise FileNotFoundError(f"OOF predictions not found at {OOF_PATH}")
        
    oof = pd.read_csv(OOF_PATH)
    
    # Ensure lengths match
    if len(train) != len(oof):
        # Check if OOF is just predictions column or has IDs
        print(f"Warning: Length mismatch! Train: {len(train)}, OOF: {len(oof)}")
        # Assuming OOF is aligned with Train (StratifiedKFold preserves index usually, but let's check)
        # If OOF was saved as just a column, we assume it matches the train index.
    
    return train, oof

def analyze_errors(train, oof):
    print("Identifying Hard Samples...")
    
    # Get predictions
    if TARGET_COL in oof.columns:
        preds = oof[TARGET_COL]
    else:
        # Assume single column is prediction
        preds = oof.iloc[:, 0]
        
    # Calculate Absolute Error
    y_true = train[TARGET_COL]
    errors = np.abs(y_true - preds)
    
    # Define Hard Samples (Top 5% worst errors)
    threshold = np.percentile(errors, 95)
    print(f"Error Threshold (Top 5%): {threshold:.4f}")
    
    train['error'] = errors
    train['is_hard_sample'] = (errors > threshold).astype(int)
    
    n_hard = train['is_hard_sample'].sum()
    print(f"Number of Hard Samples: {n_hard} / {len(train)}")
    
    return train

def find_patterns(df):
    print("Training Detective Tree...")
    
    # Features to analyze (exclude target and error columns)
    exclude_cols = [TARGET_COL, 'id', 'error', 'is_hard_sample', 'fold']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['is_hard_sample']
    
    # Handle Categoricals for Decision Tree
    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols])
        # Fill NaNs
        X = X.fillna(-1)
    
    # Train simple Decision Tree
    # max_depth=3 for interpretability
    dt = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
    dt.fit(X, y)
    
    # Extract Rules
    rules = export_text(dt, feature_names=list(X.columns))
    
    # Feature Importance
    importances = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
    
    return rules, importances

def generate_report(rules, importances, df):
    print(f"Generating report to {OUTPUT_REPORT}...")
    
    os.makedirs('Analysis', exist_ok=True)
    
    with open(OUTPUT_REPORT, 'w') as f:
        f.write("# 🕵️‍♂️ Error Analysis Report\n\n")
        f.write("## 🎯 Objective\n")
        f.write("Identify patterns in the 'Hard Samples' (Top 5% worst errors) of the Brute-Force model.\n\n")
        
        f.write("## 📊 Key Suspects (Feature Importance)\n")
        f.write("The Decision Tree found these features most useful for distinguishing Hard Samples:\n\n")
        for feature, imp in importances.items():
            f.write(f"- **{feature}**: {imp:.4f}\n")
        f.write("\n")
        
        f.write("## 📜 The Rules (Decision Tree Logic)\n")
        f.write("Here is the logic the tree used to find errors:\n\n")
        f.write("```text\n")
        f.write(rules)
        f.write("```\n\n")
        
        f.write("## 🔍 Hard Sample Stats\n")
        hard_samples = df[df['is_hard_sample'] == 1]
        normal_samples = df[df['is_hard_sample'] == 0]
        
        f.write("| Feature | Hard Samples Mean | Normal Samples Mean |\n")
        f.write("| :--- | :--- | :--- |\n")
        
        # Compare means of top features
        for feature in importances.index:
            if feature in df.columns: # Ensure it exists (might be encoded)
                 # Check if numeric
                if pd.api.types.is_numeric_dtype(df[feature]):
                    mean_hard = hard_samples[feature].mean()
                    mean_normal = normal_samples[feature].mean()
                    f.write(f"| {feature} | {mean_hard:.2f} | {mean_normal:.2f} |\n")
                
    print("Report generated successfully.")

def main():
    # 1. Load
    train, oof = load_data()
    
    # 2. Analyze
    df_analyzed = analyze_errors(train, oof)
    
    # 3. Find Patterns
    rules, importances = find_patterns(df_analyzed)
    
    # 4. Report
    generate_report(rules, importances, df_analyzed)
    
    # Print to console for immediate feedback
    print("\n--- Top Suspects ---")
    print(importances)
    print("\n--- Rules ---")
    print(rules)

if __name__ == "__main__":
    main()
