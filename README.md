# Kaggle Playground S5E11 - Loan Prediction (Score: 0.92366)

This repository contains the complete solution for the Kaggle Playground Series Season 5, Episode 11 competition.
**Final Public Leaderboard Score:** `0.92366`

## 🏆 Project Summary

We built a robust machine learning pipeline that combines explicit mathematical feature engineering, deep learning, evolutionary algorithms, and automated machine learning. The final submission is a weighted ensemble of 7 different models, optimized using a Hill Climbing algorithm.

### 🥇 Best Models (OOF Scores)
1.  **Optimized Ensemble:** `0.92336` (CV) -> `0.92366` (LB)
2.  **Brute-Force LightGBM:** `0.92311` (CV) -> `0.92360` (LB)
3.  **DAE-Boosted LightGBM:** `0.92311` (CV) -> `0.92360` (LB)

---

## 🛠️ What Worked (The Winning Strategy)

### 1. Feature Engineering
*   **Brute-Force Interactions:** We generated thousands of arithmetic interactions (`+`, `-`, `*`, `/`) and selected the top performing ones using a fast LightGBM.
    *   *Key Feature:* `debt_to_income_ratio / employment_dti_interaction`
*   **Genetic Programming (`gplearn`):** We evolved complex non-linear formulas that maximized correlation with the target.
    *   *Example:* `add(mul(log(X8), log(sqrt(mul(X8, log(X8))))), X1)`
*   **Denoising Autoencoder (DAE):** We trained a neural network to reconstruct noisy data, extracting 32 deep latent features from the bottleneck layer.
*   **Golden Features:** Domain-specific financial ratios (e.g., `Disposable Income`, `Credit Utilization`) and KNN-based features.

### 2. Model Diversity
*   **LightGBM:** Tuned with Optuna (50 trials).
*   **AutoGluon:** Used as a "Stage 1" model trained on the original dataset (20k rows) to provide a robust prior.
*   **TabPFN:** Transformer-based tabular model used for residual boosting.

### 3. Advanced Techniques
*   **Pseudo-Labeling:** Augmented training data with high-confidence predictions from the best model.
*   **Residual Boosting:** Used predictions from auxiliary models (TabPFN, Original Data Model) as features.
*   **Hill Climbing Optimization:** Instead of manual weighting, we used a custom algorithm to mathematically find the optimal ensemble weights.

---

## ❌ What Didn't Work (Lessons Learned)

*   **TabNet:** Despite being a state-of-the-art architecture, it achieved a lower score (`0.91274`) compared to tree-based models and was assigned 0 weight in the final ensemble.
*   **Adversarial Validation:** We checked for covariate drift between Train and Test sets. The AUC was `0.50`, indicating no drift. While good news, it meant we couldn't get a "free boost" by dropping drifting features.
*   **Simple Averaging:** A simple average of models performed worse than the Hill Climbing optimized ensemble.

---

## 📂 Repository Structure

```
├── Analysis/               # Error analysis reports and EDA
├── Datasets/               # Original and processed data
├── Models/                 # OOF predictions and saved models
├── Processed/              # Cleaned CSV files
├── Scripts/                # Python scripts for training and inference
│   ├── train_bruteforce.py         # Main LightGBM model
│   ├── train_dae_boosted.py        # DAE feature extraction + Training
│   ├── train_genetic_features.py   # Genetic Programming
│   ├── train_golden_features.py    # Domain & KNN features
│   ├── optimize_ensemble_weights.py # Hill Climbing Algorithm
│   └── ...
├── Submissions/            # Final CSV files for Kaggle
└── README.md               # This file
```

## 🚀 How to Reproduce

1.  **Install Dependencies:**
    ```bash
    pip install lightgbm xgboost catboost gplearn tensorflow scikit-learn pandas numpy
    ```

2.  **Run the Pipeline:**
    ```bash
    # 1. Preprocessing
    python Scripts/data_preprocessing.py

    # 2. Train Base Models
    python Scripts/find_magic_features.py  # Brute-Force
    python Scripts/train_genetic_features.py # Genetic
    python Scripts/train_golden_features.py # Golden

    # 3. Train DAE (Requires GPU/Colab)
    # Run Scripts/colab_dae_features.py on Colab and download features

    # 4. Train DAE-Boosted Model
    python Scripts/train_dae_boosted.py

    # 5. Optimize Ensemble
    python Scripts/optimize_ensemble_weights.py
    ```

3.  **Final Submission:**
    The final file will be generated at `Submissions/submission_optimized_ensemble.csv`.

---

**Author:** Berkay HSRT
**Date:** November 2025
