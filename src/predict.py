"""
Bank Direct Marketing - Term Deposit Subscription Prediction
Inputs : preprocessed_mean.csv, preprocessed_ffill.csv (model-ready)
Models : Logistic Regression & Random Forest
Eval   : K-Fold Cross-Validation (Accuracy)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

RANDOM_STATE = 42
DATA_DIR  = "data/processed"
PATH_MEAN  = f"{DATA_DIR}/preprocessed_mean.csv"
PATH_FFILL = f"{DATA_DIR}/preprocessed_ffill.csv"


# ---------------------------------------------------------------
# 1. Load preprocessed (model-ready) data
# ---------------------------------------------------------------
versions = {
    "A. Mean Imputation": pd.read_csv(PATH_MEAN),
    "B. Forward-Fill"   : pd.read_csv(PATH_FFILL),
}

# ---------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

# ---------------------------------------------------------------
# 3. K-Fold Cross-Validation
# ---------------------------------------------------------------
print("=" * 70)
print("K-FOLD CROSS-VALIDATION (k=5, Accuracy)")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
summary_rows = []

for v_name, df in versions.items():
    X = df.drop(columns=["y"])
    y = df["y"].astype(int)
    print(f"\n--- {v_name}  (X shape: {X.shape}) ---")
    for m_name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv,
                                 scoring="accuracy", n_jobs=-1)
        print(f"\n  [{m_name}]")
        print(f"    Fold accuracies : {np.round(scores, 4)}")
        print(f"    Mean accuracy   : {scores.mean():.4f}")
        print(f"    Std             : {scores.std():.4f}")
        summary_rows.append({
            "Version" : v_name,
            "Model"   : m_name,
            "Mean Acc": scores.mean(),
            "Std"     : scores.std(),
        })

# ---------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
summary = pd.DataFrame(summary_rows).sort_values("Mean Acc", ascending=False)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

best = summary.iloc[0]
print(f"\nBest configuration: {best['Version']} + {best['Model']} "
      f"(Mean Accuracy = {best['Mean Acc']:.4f})")
