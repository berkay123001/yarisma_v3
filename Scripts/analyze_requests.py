import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
import warnings

warnings.filterwarnings('ignore')

def load_data():
    print("Loading data...")
    return pd.read_csv('Processed/clean_train.csv')

def test_polynomial_features(df, target_col):
    print("\n--- 1. Polynomial Features Investigation ---")
    focus_vars = ['annual_income', 'loan_amount', 'interest_rate']
    
    results = []
    
    for col in focus_vars:
        if col not in df.columns:
            continue
            
        # Base correlation
        base_corr = df[col].corr(df[target_col])
        results.append({'Feature': col, 'Transformation': 'None', 'Correlation': base_corr})
        
        # Square
        sq_col = df[col] ** 2
        sq_corr = sq_col.corr(df[target_col])
        results.append({'Feature': col, 'Transformation': 'Square (^2)', 'Correlation': sq_corr})
        
        # Sqrt
        sqrt_col = np.sqrt(df[col].clip(lower=0))
        sqrt_corr = sqrt_col.corr(df[target_col])
        results.append({'Feature': col, 'Transformation': 'Sqrt', 'Correlation': sqrt_corr})
        
        # Log1p
        log_col = np.log1p(df[col].clip(lower=0))
        log_corr = log_col.corr(df[target_col])
        results.append({'Feature': col, 'Transformation': 'Log1p', 'Correlation': log_corr})
        
    results_df = pd.DataFrame(results)
    # Calculate absolute improvement over None
    results_df['Abs_Correlation'] = results_df['Correlation'].abs()
    
    print(results_df.sort_values(['Feature', 'Abs_Correlation'], ascending=[True, False]))
    return results_df

def test_binning_strategy(df, target_col):
    print("\n--- 2. Binning Strategy (Discretization) ---")
    focus_vars = ['loan_amount', 'credit_score']
    
    for col in focus_vars:
        if col not in df.columns:
            continue
            
        print(f"\nAnalyzing Bins for {col}:")
        
        # Create 5 quantile bins
        try:
            est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
            bins = est.fit_transform(df[[col]]).flatten()
            
            temp_df = pd.DataFrame({
                'bin': bins,
                'target': df[target_col]
            })
            
            # Calculate default rate per bin
            bin_stats = temp_df.groupby('bin')['target'].agg(['mean', 'count'])
            bin_stats['mean'] = bin_stats['mean'].round(4)
            print(bin_stats)
            
            # Check if there's a monotonic trend or distinct groups
            diffs = bin_stats['mean'].diff().abs().mean()
            print(f"Average difference between adjacent bins: {diffs:.4f}")
            
        except Exception as e:
            print(f"Binning failed for {col}: {e}")

def test_dti_thresholds(df, target_col):
    print("\n--- 3. Debt-to-Income (DTI) Thresholds ---")
    col = 'debt_to_income_ratio'
    
    if col not in df.columns:
        return

    # Check correlation of raw DTI
    raw_corr = df[col].corr(df[target_col])
    print(f"Raw DTI Correlation: {raw_corr:.4f}")
    
    # Test thresholds from 0.1 to 0.6
    thresholds = np.arange(0.1, 0.7, 0.05)
    best_thresh = None
    best_corr = 0
    
    print("Testing thresholds (Binary Flag: DTI > Threshold):")
    for t in thresholds:
        flag = (df[col] > t).astype(int)
        corr = flag.corr(df[target_col])
        
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_thresh = t
            
        # print(f"Threshold > {t:.2f}: Correlation = {corr:.4f}")
        
    print(f"Best Threshold: > {best_thresh:.2f} with Correlation: {best_corr:.4f}")
    
    # Check if it beats raw correlation significantly
    if abs(best_corr) > abs(raw_corr) * 1.1: # 10% improvement
        print("VERDICT: Significant threshold found!")
    else:
        print("VERDICT: No threshold significantly outperforms raw feature.")

def main():
    train = load_data()
    target = 'loan_paid_back'
    
    test_polynomial_features(train, target)
    test_binning_strategy(train, target)
    test_dti_thresholds(train, target)

if __name__ == "__main__":
    main()
