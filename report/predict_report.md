# Code Report: `predict.py`

**Topic:** Bank Direct Marketing — Term Deposit Subscription Prediction
**Perspective:** ML Methodology Analysis

---

## 1. Overview

This script implements a binary classification pipeline that predicts whether a customer will subscribe to a term deposit (`y`) based on bank direct marketing data. It applies two classifiers to two preprocessed dataset versions (differing only in their missing-value handling strategy) and compares their performance using **5-Fold Stratified Cross-Validation** with accuracy as the metric.

---

## 2. Pipeline Structure

```
Preprocessed CSVs (2)  ──►  Model Training/Evaluation (2)  ──►  5-Fold CV Accuracy  ──►  Summary Comparison
```

| Stage | Component | Notes |
|-------|-----------|-------|
| Data Input | `preprocessed_mean.csv`, `preprocessed_ffill.csv` | Assumed to be model-ready (encoding/scaling already applied) |
| Models | Logistic Regression, Random Forest | Linear vs. non-linear representatives |
| Evaluation | StratifiedKFold (k=5), accuracy | Preserves class proportions |
| Output | Per-fold scores, mean/std, best configuration | Aggregated in a DataFrame |

---

## 3. ML Methodology Analysis

### 3.1 Missing Value Imputation Strategy

The script treats two imputation methods as **independent dataset versions** and compares model performance across them.

- **Mean Imputation**: Replaces missing values with the column mean. Stable, but flattens the distribution (reduces variance).
- **Forward-Fill**: Replaces missing values with the previous row's value. Justified only when row order carries temporal meaning — which is rarely the case for this dataset.

> **Methodological note:** The UCI Bank Marketing dataset (and its variants) typically has no meaningful row ordering, so forward-fill is acceptable as an experimental comparison but is **not theoretically well-grounded** as a production choice.

### 3.2 Model Selection

| Model | Characteristics | Role in This Study |
|-------|-----------------|---------------------|
| Logistic Regression (`max_iter=2000`) | Linear decision boundary, interpretable coefficients | Baseline |
| Random Forest (`n_estimators=200`) | Captures non-linearity and feature interactions, provides feature importance | Strong comparator |

`max_iter=2000` is a reasonable choice for ensuring convergence, and 200 trees for the Random Forest strike a sensible balance between variance reduction and computational cost. However, **no hyperparameter tuning** (e.g., GridSearchCV) is performed for either model.

### 3.3 Evaluation: Stratified K-Fold

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
```

- **Stratified**: Appropriate given that term deposit subscription rates are typically ~10–11%, producing significant **class imbalance**.
- **shuffle=True + fixed random_state**: Ensures reproducibility.
- **n_jobs=-1**: Parallelizes folds for faster execution.

### 3.4 ⚠ Limitation: Accuracy as the Sole Metric

This is the most important methodological concern in the script.

- The Bank Marketing dataset has roughly **11% positive class** (`y=1`).
- A trivial classifier predicting "all negative" would already achieve **~89% accuracy** — meaning accuracy provides almost no discriminative signal about model quality.
- **Recommended alternatives**: ROC-AUC, F1-score, Precision-Recall AUC, or — from a business standpoint — **recall** (minimize missed prospects) or **precision** (maximize call-cost efficiency).

### 3.5 Data Leakage Concern

The fact that preprocessing happens **outside** the training script (results saved as CSVs) introduces a potential risk:

- If the mean used for imputation was computed over the **entire dataset**, the training folds were implicitly informed by the test fold statistics — a classic case of **data leakage**.
- The recommended fix is to wrap imputation, scaling, and the model in a single `Pipeline`, so that each fold's preprocessing is fit only on its own training portion within CV.

---

## 4. Code Quality Assessment

**Strengths**
- Constants (paths, random_state) are cleanly separated at the top.
- The nested `versions × models` loop produces a clean comparative experiment structure.
- Results are aggregated into a DataFrame for easy sorting and comparison.

**Areas for Improvement**
- The hardcoded absolute path (`/Users/jimoon/...`) reduces portability — consider `argparse` or relative paths.
- Only one metric (accuracy) is reported — multi-metric evaluation is needed.
- Preprocessing is separated from modeling, raising leakage risk — recommend `sklearn.pipeline.Pipeline`.
- Hyperparameters are fixed — `GridSearchCV` or `RandomizedSearchCV` would likely improve results.

---

## 5. Conclusion and Recommendations

This script is a well-structured baseline experiment that systematically compares **2 preprocessing strategies × 2 models** using K-Fold cross-validation. From an ML methodology perspective, however, three improvements are essential:

1. **Diversify evaluation metrics** — Accuracy alone is misleading on imbalanced data. At minimum, **report ROC-AUC and F1** alongside accuracy.
2. **Integrate preprocessing into a Pipeline** — Embedding imputation inside CV eliminates the leakage risk.
3. **Add hyperparameter tuning** — In particular, Random Forest's `max_depth` and `min_samples_leaf`, and Logistic Regression's `C` (regularization strength), have strong effects on performance.

Incorporating these three changes would elevate the current baseline into a substantially more trustworthy model comparison.
