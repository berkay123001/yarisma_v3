# Feature Engineering Analysis Report (v2)

## Executive Summary
We tested the three hypotheses requested by the Lead ML Engineer.
*   **APPROVED:** Binning `credit_score` (Strong signal), Log-transform `annual_income`.
*   **REJECTED:** Binning `loan_amount`, Polynomials for `interest_rate`, DTI Thresholds.

## Detailed Findings

### 1. Polynomial Features
*   **Hypothesis:** Non-linear transformations improve correlation.
*   **Verdict:** **PARTIALLY APPROVED**
*   **Findings:**
    *   `annual_income`: **Log1p** improves correlation slightly (0.006 -> 0.008). Given income's skew, this is theoretically sound.
    *   `interest_rate`: Squaring provides negligible gain (< 0.001). **REJECTED**.
    *   `loan_amount`: No transformation helped. **REJECTED**.

### 2. Binning Strategy
*   **Hypothesis:** Risk bands exist for continuous variables.
*   **Verdict:** **MIXED**
*   **Findings:**
    *   `credit_score`: **APPROVED**. Strong monotonic trend found.
        *   Bin 0: 65.7% Repayment
        *   Bin 4: 92.6% Repayment
        *   *Action:* Create 5 ordinal bins.
    *   `loan_amount`: **REJECTED**. No clear trend (flat default rate ~80% across all bins).

### 3. DTI Thresholds
*   **Hypothesis:** A "cliff" exists where risk spikes.
*   **Verdict:** **REJECTED**
*   **Findings:**
    *   Raw `debt_to_income_ratio` correlation: **-0.3357**
    *   Best Binary Threshold (> 0.15): **-0.3008**
    *   *Conclusion:* The continuous variable contains more information than any binary flag. Do not discretize.

## Implementation Code
Copy these snippets into your pipeline:

```python
# 1. Log Transform Income
df['annual_income_log'] = np.log1p(df['annual_income'])

# 2. Credit Score Binning (Quantile-based)
from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df['credit_score_bin'] = est.fit_transform(df[['credit_score']])
```
