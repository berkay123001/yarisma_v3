# 🕵️‍♂️ Error Analysis Report

## 🎯 Objective
Identify patterns in the 'Hard Samples' (Top 5% worst errors) of the Brute-Force model.

## 📊 Key Suspects (Feature Importance)
The Decision Tree found these features most useful for distinguishing Hard Samples:

- **debt_to_income_ratio**: 0.4231
- **credit_score**: 0.4123
- **employment_status**: 0.1647
- **annual_income**: 0.0000
- **loan_amount**: 0.0000

## 📜 The Rules (Decision Tree Logic)
Here is the logic the tree used to find errors:

```text
|--- credit_score <= 699.50
|   |--- debt_to_income_ratio <= 0.18
|   |   |--- credit_score <= 656.50
|   |   |   |--- class: 1
|   |   |--- credit_score >  656.50
|   |   |   |--- class: 1
|   |--- debt_to_income_ratio >  0.18
|   |   |--- employment_status <= 3.50
|   |   |   |--- class: 0
|   |   |--- employment_status >  3.50
|   |   |   |--- class: 0
|--- credit_score >  699.50
|   |--- debt_to_income_ratio <= 0.15
|   |   |--- employment_status <= 2.50
|   |   |   |--- class: 0
|   |   |--- employment_status >  2.50
|   |   |   |--- class: 1
|   |--- debt_to_income_ratio >  0.15
|   |   |--- employment_status <= 3.50
|   |   |   |--- class: 1
|   |   |--- employment_status >  3.50
|   |   |   |--- class: 0
```

## 🔍 Hard Sample Stats
| Feature | Hard Samples Mean | Normal Samples Mean |
| :--- | :--- | :--- |
| debt_to_income_ratio | 0.12 | 0.12 |
| credit_score | 660.11 | 682.01 |
| employment_status | 0.50 | 0.70 |
| annual_income | 47777.54 | 48235.08 |
| loan_amount | 15109.89 | 15015.58 |
