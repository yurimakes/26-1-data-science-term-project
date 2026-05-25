"""
Bank Direct Marketing - Model Evaluation
=========================================

Purpose:
    Evaluate the performance of classification models for term deposit
    subscription prediction. Compare preprocessing versions and machine
    learning models using multiple evaluation metrics. Provide additional
    evaluation beyond accuracy because the target class is imbalanced.

Inputs:
    data/processed/preprocessed_mean.csv
        - Model-ready dataset created using mean/mode imputation.
    data/processed/preprocessed_ffill.csv
        - Model-ready dataset created using forward-fill and backfill imputation.

Models evaluated:
    - Logistic Regression (baseline)
    - Random Forest Classifier (ensemble)

Evaluation Methods:
    - Stratified 5-Fold Cross-Validation (multiple metrics)
    - Train-Test Split for final best model evaluation

Evaluation Metrics:
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - ROC-AUC
    - Confusion Matrix

Outputs:
    results/tables/model_comparison_metrics.csv
        - Cross-validation results for all model and preprocessing combinations.
    results/tables/best_model_test_metrics.csv
        - Final test-set evaluation results of the best model.
    results/figures/confusion_matrix_best_model.png
        - Confusion matrix visualization for the best model.

Note:
    The original predict.py evaluates models mainly using accuracy.
    This file extends the evaluation by adding precision, recall, F1-score,
    ROC-AUC, and confusion matrix analysis.
    Since the dataset has a class imbalance, recall and F1-score are important
    for evaluating how well the model identifies actual subscribers.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# Global random seed for reproducibility.
# Used as the default value in evaluate_classification_models().
RANDOM_STATE = 42


def evaluate_classification_models(
    data_paths=None,
    models=None,
    target_col="y",
    cv_splits=5,
    test_size=0.2,
    scoring=None,
    best_metric="F1-score",
    result_dir="results",
    random_state=RANDOM_STATE,
    n_jobs=-1
):
    """
    Evaluate multiple classification models on multiple preprocessed datasets.

    This function compares different preprocessing versions and classification
    models — including multiple hyperparameter configurations — using stratified
    k-fold cross-validation. It also selects the best model based on the chosen
    metric and evaluates it on a separate test set.

    Parameters
    ----------
    data_paths : dict, optional
        Dictionary containing dataset names and CSV file paths.
        If None, the function uses the following default datasets:

        - "Mean Imputation": data/processed/preprocessed_mean.csv
        - "Forward-Fill":    data/processed/preprocessed_ffill.csv

    models : dict, optional
        Dictionary containing model names and scikit-learn classifier objects.
        If None, the function uses multiple configurations of Logistic Regression
        and Random Forest to compare various hyperparameter combinations:

        - "LR (C=0.1)":         Logistic Regression with strong regularization
        - "LR (C=1.0)":         Logistic Regression with default regularization
        - "LR (C=10.0)":        Logistic Regression with weak regularization
        - "RF (n=100, depth=None)": Random Forest with 100 trees, unlimited depth
        - "RF (n=200, depth=None)": Random Forest with 200 trees, unlimited depth
        - "RF (n=200, depth=10)":   Random Forest with 200 trees, max depth 10

    target_col : str, default="y"
        Name of the target column. In this project, "y" indicates whether
        the customer subscribed to a term deposit.

    cv_splits : int, default=5
        Number of folds for Stratified K-Fold cross-validation.

    test_size : float, default=0.2
        Proportion of the dataset used for final test-set evaluation.

    scoring : dict, optional
        Dictionary of evaluation metrics used in cross-validation.
        If None, accuracy, precision, recall, F1-score, and ROC-AUC are used.

    best_metric : str, default="F1-score"
        Metric used to select the best model after cross-validation.
        Must be one of the column names produced in the cross-validation summary.

    result_dir : str, default="results"
        Directory where result tables and figures are saved.
        Sub-directories "tables/" and "figures/" are created automatically.

    random_state : int, default=42
        Random seed for reproducibility. Applied to all models, cross-validation
        splits, and the train-test split.

    n_jobs : int, default=-1
        Number of CPU cores used for model evaluation.
        -1 means using all available cores.

    Returns
    -------
    summary : pandas.DataFrame
        Cross-validation results for all preprocessing and model combinations,
        sorted by the chosen best_metric in descending order.

    final_metrics_df : pandas.DataFrame
        Final test-set evaluation results of the selected best model.

    cm : numpy.ndarray
        Confusion matrix of the selected best model on the test set.

    Examples
    --------
    # Run with default settings
    summary, final_metrics, cm = evaluate_classification_models()

    # Use custom models with different hyperparameter combinations
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    custom_models = {
        "LR (C=0.01)": LogisticRegression(C=0.01, max_iter=2000, random_state=42),
        "RF (n=300)":  RandomForestClassifier(n_estimators=300, random_state=42),
    }
    summary, final_metrics, cm = evaluate_classification_models(
        models=custom_models,
        best_metric="Recall"
    )
    """

    # ---------------------------------------------------------------
    # 1. Set default input datasets
    # ---------------------------------------------------------------
    if data_paths is None:
        data_paths = {
            "Mean Imputation": "data/processed/preprocessed_mean.csv",
            "Forward-Fill":    "data/processed/preprocessed_ffill.csv",
        }

    # ---------------------------------------------------------------
    # 2. Set default classification models
    #    Multiple hyperparameter configurations are included so that
    #    the function fulfills the open-source SW requirement of
    #    training/testing models with various parameter combinations.
    # ---------------------------------------------------------------
    if models is None:
        models = {
            # --- Logistic Regression: three regularization strengths ---
            # C controls inverse regularization strength (smaller C = stronger regularization)
            "LR (C=0.1)": LogisticRegression(
                C=0.1,
                max_iter=2000,
                random_state=random_state
            ),
            "LR (C=1.0)": LogisticRegression(
                C=1.0,           # scikit-learn default
                max_iter=2000,
                random_state=random_state
            ),
            "LR (C=10.0)": LogisticRegression(
                C=10.0,
                max_iter=2000,
                random_state=random_state
            ),

            # --- Random Forest: combinations of n_estimators and max_depth ---
            # n_estimators: number of trees; max_depth: maximum tree depth (None = unlimited)
            "RF (n=100, depth=None)": RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=random_state,
                n_jobs=n_jobs
            ),
            "RF (n=200, depth=None)": RandomForestClassifier(
                n_estimators=200,
                max_depth=None,   # best performer from original predict.py
                random_state=random_state,
                n_jobs=n_jobs
            ),
            "RF (n=200, depth=10)": RandomForestClassifier(
                n_estimators=200,
                max_depth=10,     # constrained depth to reduce overfitting risk
                random_state=random_state,
                n_jobs=n_jobs
            ),
        }

    # ---------------------------------------------------------------
    # 3. Set default evaluation metrics for cross-validation
    #    cross_validate() accepts a scoring dict and returns per-metric scores.
    # ---------------------------------------------------------------
    if scoring is None:
        scoring = {
            "accuracy":  "accuracy",
            "precision": "precision",
            "recall":    "recall",
            "f1":        "f1",
            "roc_auc":   "roc_auc",
        }

    # ---------------------------------------------------------------
    # 4. Create output directories
    #    os.makedirs with exist_ok=True avoids errors if they already exist.
    # ---------------------------------------------------------------
    table_dir  = os.path.join(result_dir, "tables")
    figure_dir = os.path.join(result_dir, "figures")

    os.makedirs(table_dir,  exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 5. Load preprocessed datasets
    # ---------------------------------------------------------------
    versions = {}
    for version_name, path in data_paths.items():
        versions[version_name] = pd.read_csv(path)

    # ---------------------------------------------------------------
    # 6. Stratified K-Fold Cross-Validation
    #    StratifiedKFold preserves the class ratio in every fold,
    #    which is important when the target is imbalanced.
    # ---------------------------------------------------------------
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    summary_rows = []

    print("=" * 80)
    print("MODEL EVALUATION WITH MULTIPLE METRICS")
    print("=" * 80)

    for version_name, df in versions.items():
        # Separate features (X) and target (y)
        X = df.drop(columns=[target_col])
        y = df[target_col].astype(int)

        for model_name, model in models.items():
            # cross_validate evaluates multiple metrics in one call,
            # unlike cross_val_score which handles only one metric.
            scores = cross_validate(
                model,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs
            )

            # Average each metric across the k folds
            row = {
                "Version":   version_name,
                "Model":     model_name,
                "Accuracy":  scores["test_accuracy"].mean(),
                "Precision": scores["test_precision"].mean(),
                "Recall":    scores["test_recall"].mean(),
                "F1-score":  scores["test_f1"].mean(),
                "ROC-AUC":   scores["test_roc_auc"].mean(),
            }

            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    # Validate the chosen best_metric against available columns
    if best_metric not in summary.columns:
        raise ValueError(
            f"best_metric must be one of {list(summary.columns)}. "
            f"Received: '{best_metric}'"
        )

    # Sort all combinations by the chosen metric (highest first)
    summary = summary.sort_values(best_metric, ascending=False)

    print("\n[Cross-Validation Summary]")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save the cross-validation summary to CSV
    summary_path = os.path.join(table_dir, "model_comparison_metrics.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved evaluation summary to: {summary_path}")

    # ---------------------------------------------------------------
    # 7. Select the best model based on the chosen metric
    # ---------------------------------------------------------------
    best         = summary.iloc[0]        # row with the highest metric value
    best_version = best["Version"]
    best_model_name = best["Model"]

    print("\n" + "=" * 80)
    print(f"BEST MODEL BASED ON {best_metric.upper()}")
    print("=" * 80)
    print(best)

    # ---------------------------------------------------------------
    # 8. Train-test evaluation for the selected best model
    #    stratify=y ensures class balance is preserved in the split.
    # ---------------------------------------------------------------
    df_best = versions[best_version]
    X = df_best.drop(columns=[target_col])
    y = df_best[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # clone() creates a new untrained copy of the best model
    best_model = clone(models[best_model_name])
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)

    # Compute ROC-AUC using predicted probabilities if available
    if hasattr(best_model, "predict_proba"):
        y_prob = best_model.predict_proba(X_test)[:, 1]  # probability of class 1
        test_roc_auc = roc_auc_score(y_test, y_prob)
    else:
        test_roc_auc = None  # fallback for models without predict_proba

    print("\n[Classification Report]")
    print(classification_report(y_test, y_pred, digits=4))

    # ---------------------------------------------------------------
    # 9. Confusion Matrix
    #    Rows = true labels, Columns = predicted labels
    #    TN | FP
    #    FN | TP
    # ---------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No", "Yes"]   # "No" = did not subscribe, "Yes" = subscribed
    )

    disp.plot()
    plt.title(f"Confusion Matrix - {best_model_name} ({best_version})")
    plt.tight_layout()

    cm_path = os.path.join(figure_dir, "confusion_matrix_best_model.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix to: {cm_path}")

    # ---------------------------------------------------------------
    # 10. Save final test-set metrics to CSV
    # ---------------------------------------------------------------
    final_metrics = {
        "Best Version":   best_version,
        "Best Model":     best_model_name,
        "Test Accuracy":  accuracy_score(y_test, y_pred),
        "Test Precision": precision_score(y_test, y_pred, zero_division=0),
        "Test Recall":    recall_score(y_test, y_pred, zero_division=0),
        "Test F1-score":  f1_score(y_test, y_pred, zero_division=0),
        "Test ROC-AUC":   test_roc_auc,
    }

    final_metrics_df = pd.DataFrame([final_metrics])

    final_metrics_path = os.path.join(table_dir, "best_model_test_metrics.csv")
    final_metrics_df.to_csv(final_metrics_path, index=False)

    print("\n[Best Model Test Metrics]")
    print(final_metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved best model test metrics to: {final_metrics_path}")

    return summary, final_metrics_df, cm


if __name__ == "__main__":
    evaluate_classification_models()
