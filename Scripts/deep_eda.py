import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import lightgbm as lgb
import os
import warnings

# Configuration
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
os.makedirs('Analysis/images', exist_ok=True)

def load_data():
    print("Loading data...")
    train = pd.read_csv('Processed/clean_train.csv')
    test = pd.read_csv('Processed/clean_test.csv')
    return train, test

def calculate_mutual_information(df, target_col, top_n=20):
    print("\n--- Calculating Mutual Information ---")
    # Identify numerical and categorical columns
    # Assuming preprocessing already handled encoding, but let's check
    # For MI, we need numeric inputs. If there are object cols, we need to encode them temporarily
    
    df_mi = df.copy()
    X = df_mi.drop(columns=[target_col, 'id'], errors='ignore')
    y = df_mi[target_col]
    
    # Simple label encoding for any remaining object columns (though clean data should be numeric)
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    # Fill NaNs for MI calculation
    X = X.fillna(0) 
    
    # Calculate MI
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    
    print("Top MI Scores:")
    print(mi_scores.head(10))
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x=mi_scores.head(top_n).values, y=mi_scores.head(top_n).index, palette='viridis')
    plt.title(f'Top {top_n} Features by Mutual Information')
    plt.xlabel('MI Score')
    plt.tight_layout()
    plt.savefig('Analysis/images/mutual_information.png')
    plt.close()
    
    return mi_scores

def adversarial_validation(train, test):
    print("\n--- Running Adversarial Validation ---")
    
    # Prepare data
    train_adv = train.drop(columns=['loan_paid_back'], errors='ignore').copy()
    test_adv = test.copy()
    
    # Add target
    train_adv['is_test'] = 0
    test_adv['is_test'] = 1
    
    # Combine
    combined = pd.concat([train_adv, test_adv], axis=0).reset_index(drop=True)
    
    # Drop ID
    X = combined.drop(columns=['is_test', 'id'], errors='ignore')
    y = combined['is_test']
    
    # Encode categoricals if any
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    # Train LightGBM
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'seed': 42,
        'n_jobs': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    feature_importances = pd.DataFrame(index=X.columns)
    feature_importances['importance'] = 0
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=100,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(0)]
        )
        
        y_pred = model.predict(X_val)
        auc = roc_auc_score(y_val, y_pred)
        aucs.append(auc)
        feature_importances['importance'] += model.feature_importance(importance_type='gain') / 5
        
    mean_auc = np.mean(aucs)
    print(f"Adversarial Validation AUC: {mean_auc:.4f}")
    
    if mean_auc > 0.70:
        print("WARNING: Significant covariate shift detected!")
    else:
        print("Train and Test distributions are relatively similar.")
        
    # Plot Feature Importance for Drift
    feature_importances = feature_importances.sort_values('importance', ascending=False)
    print("Top Drifting Features:")
    print(feature_importances.head(5))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.head(10)['importance'], y=feature_importances.head(10).index, palette='magma')
    plt.title('Top Features Contributing to Covariate Shift')
    plt.tight_layout()
    plt.savefig('Analysis/images/adversarial_drift.png')
    plt.close()
    
    return feature_importances

def interaction_discovery(df, target_col, top_features):
    print("\n--- Discovering Interactions ---")
    # Focus on top numeric features
    top_feats = top_features.index[:5].tolist() # Top 5 from MI
    
    X = df[top_feats].copy()
    y = df[target_col]
    
    best_corr = 0
    best_interaction = None
    
    interactions = []
    
    for i in range(len(top_feats)):
        for j in range(i + 1, len(top_feats)):
            f1 = top_feats[i]
            f2 = top_feats[j]
            
            # Division
            # Avoid division by zero
            div_feat = X[f1] / (X[f2] + 1e-6)
            corr_div = div_feat.corr(y)
            
            # Multiplication
            mul_feat = X[f1] * X[f2]
            corr_mul = mul_feat.corr(y)
            
            interactions.append({
                'Interaction': f'{f1} / {f2}',
                'Correlation': corr_div
            })
            interactions.append({
                'Interaction': f'{f1} * {f2}',
                'Correlation': corr_mul
            })
            
    interactions_df = pd.DataFrame(interactions)
    interactions_df['Abs_Correlation'] = interactions_df['Correlation'].abs()
    interactions_df = interactions_df.sort_values('Abs_Correlation', ascending=False)
    
    print("Top Interactions by Correlation:")
    print(interactions_df.head(5))
    
    return interactions_df

def run_pca(df, target_col):
    print("\n--- Running PCA ---")
    X = df.drop(columns=[target_col, 'id'], errors='ignore')
    y = df[target_col]
    
    # Encode and Scale
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        
    X = X.fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['Target'] = y
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Target', data=pca_df, alpha=0.5, palette='coolwarm')
    plt.title('PCA Projection (2D)')
    plt.tight_layout()
    plt.savefig('Analysis/images/pca_projection.png')
    plt.close()
    
    print("PCA plot saved.")

def main():
    train, test = load_data()
    
    # 1. Mutual Information
    mi_scores = calculate_mutual_information(train, 'loan_paid_back')
    
    # 2. Adversarial Validation
    drift_feats = adversarial_validation(train, test)
    
    # 3. Interaction Discovery
    interaction_discovery(train, 'loan_paid_back', mi_scores)
    
    # 4. PCA
    run_pca(train, 'loan_paid_back')
    
    print("\nDeep EDA Completed Successfully.")

if __name__ == "__main__":
    main()
