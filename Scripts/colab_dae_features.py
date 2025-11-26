# 1. Gerekli Kütüphaneler
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import os

# --- CONFIG ---
NOISE_RATE = 0.15 # %15 Swap Noise
EPOCHS = 50
BATCH_SIZE = 256
BOTTLENECK_DIM = 32 # Yeni özellik sayısı

def load_data():
    print("Loading data...")
    if os.path.exists('clean_train.csv'):
        train = pd.read_csv('clean_train.csv')
        test = pd.read_csv('clean_test.csv')
    else:
        raise FileNotFoundError("Lütfen 'clean_train.csv' ve 'clean_test.csv' dosyalarını yükleyin!")
    return train, test

def swap_noise(data, noise_rate):
    # Swap Noise: Rastgele satırlardan değer alıp değiştirme
    n_samples = data.shape[0]
    n_features = data.shape[1]
    
    data_noisy = data.copy()
    
    for col in range(n_features):
        mask = np.random.rand(n_samples) < noise_rate
        # Rastgele başka satırlardan değer seç
        random_indices = np.random.randint(0, n_samples, size=np.sum(mask))
        data_noisy[mask, col] = data[random_indices, col]
        
    return data_noisy

def build_dae(input_dim):
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(1500, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1500, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1500, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # Bottleneck (Latent Features)
    bottleneck = layers.Dense(BOTTLENECK_DIM, activation='linear', name='bottleneck')(x)
    
    # Decoder
    x = layers.Dense(1500, activation='relu')(bottleneck)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1500, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(1500, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    outputs = layers.Dense(input_dim, activation='linear')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    encoder = models.Model(inputs=inputs, outputs=bottleneck)
    
    model.compile(optimizer='adam', loss='mse')
    return model, encoder

def main():
    # 1. Load Data
    train, test = load_data()
    
    # Drop non-feature columns
    target_col = 'loan_paid_back'
    drop_cols = ['id', target_col]
    
    if target_col in train.columns:
        X_train = train.drop(columns=[c for c in drop_cols if c in train.columns])
    else:
        X_train = train.copy()
        
    X_test = test.drop(columns=['id']) if 'id' in test.columns else test.copy()
    
    # Align columns
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    print(f"Features: {X_train.shape[1]}")
    
    # 2. Preprocessing (RankGauss is best for NN)
    print("Preprocessing (RankGauss)...")
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    
    # Combine for scaling
    X_all = pd.concat([X_train, X_test], axis=0)
    X_all_scaled = scaler.fit_transform(X_all)
    
    X_train_scaled = X_all_scaled[:len(X_train)]
    X_test_scaled = X_all_scaled[len(X_train):]
    
    # 3. Add Noise
    print(f"Adding Swap Noise ({NOISE_RATE})...")
    X_train_noisy = swap_noise(X_train_scaled, NOISE_RATE)
    
    # 4. Train DAE
    print("Training Denoising Autoencoder...")
    dae, encoder = build_dae(X_train.shape[1])
    
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    
    dae.fit(
        X_train_noisy, X_train_scaled, # Input: Noisy, Target: Clean
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[reduce_lr, early_stop],
        verbose=1
    )
    
    # 5. Extract Features
    print("Extracting Deep Features...")
    train_features = encoder.predict(X_train_scaled)
    test_features = encoder.predict(X_test_scaled)
    
    # 6. Save
    cols = [f'dae_{i}' for i in range(BOTTLENECK_DIM)]
    
    df_train_feat = pd.DataFrame(train_features, columns=cols)
    df_test_feat = pd.DataFrame(test_features, columns=cols)
    
    df_train_feat.to_csv('dae_features_train.csv', index=False)
    df_test_feat.to_csv('dae_features_test.csv', index=False)
    
    print(f"Saved {BOTTLENECK_DIM} new features to dae_features_train.csv and dae_features_test.csv")
    
    # Download for Colab
    try:
        from google.colab import files
        files.download('dae_features_train.csv')
        files.download('dae_features_test.csv')
    except ImportError:
        pass

if __name__ == "__main__":
    main()
