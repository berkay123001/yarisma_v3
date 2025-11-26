import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import os

def preprocess_data():
    print("Loading data...")
    # Load datasets
    train_path = 'Datasets/playground-series-s5e11/train.csv'
    test_path = 'Datasets/playground-series-s5e11/test.csv'
    
    if not os.path.exists(train_path):
        # Fallback if the user meant the root Datasets folder directly (based on initial ls)
        # But the ls showed they are in a subdir. I will stick to the path found in ls.
        # Wait, the ls showed Datasets/playground-series-s5e11/train.csv
        pass

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Original Train Shape: {df_train.shape}")
    print(f"Original Test Shape: {df_test.shape}")

    # 1. ID Handling
    # Drop id from train features
    if 'id' in df_train.columns:
        df_train = df_train.drop(columns=['id'])
    
    # Keep id for test submission, but drop from features
    test_ids = df_test['id']
    df_test_features = df_test.drop(columns=['id'])

    # 2. Target Separation
    target_col = 'loan_paid_back'
    y_train = df_train[target_col]
    X_train = df_train.drop(columns=[target_col])
    X_test = df_test_features

    # Identify columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

    print(f"Numeric Columns: {list(numeric_cols)}")
    print(f"Categorical Columns: {list(categorical_cols)}")

    # 3. Imputation
    print("Imputing missing values...")
    # Numeric: Mean
    num_imputer = SimpleImputer(strategy='mean')
    X_train[numeric_cols] = num_imputer.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])

    # Categorical: Most Frequent
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

    # 4. Encoding
    print("Encoding categorical variables...")
    # Ordinal Encoding for Tree-based models
    # handle_unknown='use_encoded_value' is crucial for robustness
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    X_train[categorical_cols] = encoder.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

    # 5. Reconstruction
    # Combine X_train and y_train
    clean_train = pd.concat([X_train, y_train], axis=1)
    
    # Combine test_ids and X_test
    clean_test = pd.concat([test_ids, X_test], axis=1)

    # 6. Save
    output_dir = 'Processed'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving processed files to '{output_dir}/'...")
    clean_train.to_csv(os.path.join(output_dir, 'clean_train.csv'), index=False)
    clean_test.to_csv(os.path.join(output_dir, 'clean_test.csv'), index=False)

    print("-" * 30)
    print("Preprocessing Complete!")
    print(f"Clean Train Shape: {clean_train.shape}")
    print(f"Clean Test Shape: {clean_test.shape}")
    print(f"Encoded Columns: {list(categorical_cols)}")
    print("-" * 30)

if __name__ == "__main__":
    preprocess_data()
