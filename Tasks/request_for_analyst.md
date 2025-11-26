# Request for Data Analyst: Advanced Feature Engineering

**Priority:** High
**Context:** We have established a strong baseline (AUC ~0.922) using basic interactions. To close the gap to the top leaderboard (0.928), we need to extract deeper signals from the data.

## Objectives

### 1. Polynomial Features Investigation
Please investigate if non-linear transformations of continuous variables yield better separation.
*   **Focus Variables:** `annual_income`, `loan_amount`, `interest_rate`.
*   **Transformations:** Square (`^2`), Square Root (`sqrt`), Log (`log1p`).
*   **Hypothesis:** Extreme values in income or loan amount might have a disproportionate effect on repayment probability.

### 2. Binning Strategy (Discretization)
Continuous variables might have "risk bands".
*   **Task:** Analyze `loan_amount` and `credit_score`.
*   **Action:** Create bins (e.g., `low`, `medium`, `high`, `very_high`) and check the default rate within each bin.
*   **Goal:** If specific bins show distinct risk profiles, we can encode them as Ordinal features.

### 3. Debt-to-Income (DTI) Thresholds
*   **Task:** Analyze the `debt_to_income_ratio` distribution.
*   **Hypothesis:** There might be a critical threshold (e.g., > 0.40) where default risk spikes non-linearly.
*   **Deliverable:** Identify if such a "cliff" exists and propose a binary flag feature (e.g., `is_high_dti`).

## Output Required
*   A brief report or updated `Analysis/master_strategy.md` with your findings.
*   Python code snippets for the most promising new features.
