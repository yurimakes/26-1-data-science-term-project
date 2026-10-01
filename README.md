# Bank Marketing Classification

> Data Science Term Project · 5-person team project  
> Binary classification of term-deposit subscription using the UCI Bank Marketing dataset.

## Overview

This project builds and evaluates a machine-learning workflow for predicting whether a customer will subscribe to a term deposit after a direct-marketing campaign.

The repository focuses on:

- data cleaning and preprocessing
- comparison of missing-value handling strategies
- Logistic Regression and Random Forest classification
- stratified cross-validation
- evaluation with multiple metrics for an imbalanced target
- reproducible result tables and confusion-matrix outputs

## My Contribution

**Choi Yu-ri**

- Model evaluation and validation
- Multi-metric classification analysis
- Final interpretation and conclusion
- Open-source SW / GitHub repository management
- Report writing

## Problem

The target variable is imbalanced, so accuracy alone can overstate model quality.

To address this, the final evaluation considers:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

The evaluation workflow uses **F1-score as the primary selection criterion among the tested configurations**, while also reviewing recall and ROC-AUC to avoid relying on accuracy alone.

## Pipeline

```text
Raw Bank Marketing Data
        |
        v
Duplicate Removal / Missing-Value Handling
        |
        +--> Mean / Mode Imputation
        |
        +--> Forward-Fill Experiment
        |
        v
Categorical Encoding + Standard Scaling
        |
        v
Logistic Regression / Random Forest
        |
        v
Stratified 5-Fold Cross-Validation
        |
        v
Multi-Metric Evaluation
        |
        v
Result Tables + Confusion Matrix
```

## Model Comparison

The evaluation compares multiple configurations of Logistic Regression and Random Forest.

| Model family | Main configurations |
|---|---|
| Logistic Regression | `C=0.1`, `C=1.0`, `C=10.0` |
| Random Forest | `n_estimators=100/200`, `max_depth=None/10` |

### Best Tested Configuration by F1-score

Among the configurations evaluated in this coursework, the final test-set evaluation selected:

- **Preprocessing:** Forward-Fill experiment
- **Model:** Random Forest (`n_estimators=200`, `max_depth=None`)

| Metric | Test result |
|---|---:|
| Accuracy | 0.8897 |
| Precision | 0.5511 |
| Recall | 0.2989 |
| F1-score | 0.3876 |
| ROC-AUC | 0.7742 |

### Result Interpretation

The relatively high accuracy should **not** be interpreted as strong overall classification performance by itself.

Because the positive class is a minority, a model can achieve high accuracy by predicting the majority class well while still missing many actual subscribers. In this experiment:

- **Precision = 0.5511** means that about 55% of customers predicted as subscribers were actually subscribers.
- **Recall = 0.2989** means that the model identified only about 30% of the actual subscribers.
- **F1-score = 0.3876** reflects the limited balance between precision and recall.
- **ROC-AUC = 0.7742** indicates that the model has meaningful discriminatory ability, but there is substantial room for improvement.

This result is therefore treated as a **baseline analytical outcome rather than a production-ready model**.

The project intentionally reports these limitations instead of presenting accuracy alone as the main success metric.

### Why This Matters

One of the main findings of this project is that **accuracy can be misleading for imbalanced classification problems**.

The initial workflow emphasized accuracy, but the evaluation was extended to include precision, recall, F1-score, ROC-AUC, and a confusion matrix. This made it possible to identify the low positive-class recall that would otherwise be hidden by the overall accuracy.

Detailed comparison results are available in:

- `results/tables/model_comparison_metrics.csv`
- `results/tables/best_model_test_metrics.csv`
- `results/figures/confusion_matrix_best_model.png`

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── bank_direct_marketing_campaigns.csv
│   └── processed/
│       ├── data_cleaning_finished.csv
│       ├── data_encoded.csv
│       ├── data_final_preprocessed.csv
│       ├── duplicate_removed.csv
│       ├── preprocessed_ffill.csv
│       └── preprocessed_mean.csv
├── report/
│   └── predict_report.md
├── results/
│   ├── figures/
│   │   └── confusion_matrix_best_model.png
│   └── tables/
│       ├── best_model_test_metrics.csv
│       └── model_comparison_metrics.csv
├── src/
│   ├── TermProject.py
│   ├── evaluation.py
│   ├── predict.py
│   └── preprocess.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run preprocessing

```bash
python src/preprocess.py
```

### 3. Run the baseline model comparison

```bash
python src/predict.py
```

### 4. Run the multi-metric evaluation

```bash
python src/evaluation.py
```

Optional step-by-step preprocessing and scaling verification:

```bash
python src/TermProject.py
```

## Technical Stack

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- seaborn
- Git / GitHub

## Methodological Notes

This repository preserves both the original coursework workflow and a later multi-metric evaluation extension.

Important limitations and improvement opportunities include:

1. **Class imbalance**  
   Accuracy is not sufficient by itself. The current model shows relatively low recall for the positive class, so F1-score, precision, recall, ROC-AUC, and the confusion matrix are reported together.

2. **Forward-fill is experimental**  
   Forward-fill is included as a comparison strategy, but row order in this dataset does not provide a strong theoretical basis for using it as a production imputation method.

3. **Potential preprocessing leakage**  
   Some preprocessing is performed before cross-validation. A stronger production workflow would place imputation, encoding, scaling, and modeling inside a scikit-learn `Pipeline` so transformations are fitted only on each training fold.

4. **Limited imbalance handling**  
   The current Random Forest configuration does not explicitly use imbalance-aware techniques such as `class_weight="balanced"`, resampling, or threshold tuning.

5. **Limited model search**  
   The current comparison covers Logistic Regression and Random Forest with a small number of parameter combinations. Broader model selection and tuning could improve performance.

See `report/predict_report.md` for the detailed methodology review.

## Future Improvements

If this project is extended beyond the coursework version, the following improvements would be the next priorities:

- rebuild preprocessing and modeling with a scikit-learn `Pipeline`
- evaluate `class_weight="balanced"` for Logistic Regression and Random Forest
- compare resampling approaches such as SMOTE where appropriate
- tune the classification threshold based on business objectives
- use `GridSearchCV` or `RandomizedSearchCV` for broader hyperparameter optimization
- compare additional models such as gradient boosting methods
- evaluate Precision-Recall AUC in addition to ROC-AUC
- define a business-oriented objective, such as maximizing subscriber recall under a fixed contact budget

The current repository is therefore best understood as a **coursework baseline with explicit methodological review and a clear path for further development**.

## Team

- Shin In-seub — Topic Management and Project Direction, Presentation, PPT, and Script Preparation
- Jo Yoon-sung — Objective Setting, Data Preprocessing, and Report Writing
- Moon Ji-seok — Algorithms, Modeling, and Report Writing
- Choi Yu-ri — Evaluation, Conclusion, Open Source SW (GitHub), and Report Writing
- Han Sang-hoo

## Dataset

This project uses the **Bank Marketing** dataset from the UCI Machine Learning Repository.

- Dataset: Bank Marketing
- DOI: `10.24432/C5K306`
- License: **CC BY 4.0**

UCI dataset page:  
https://archive.ics.uci.edu/dataset/222/bank+marketing

The dataset may be shared and adapted under CC BY 4.0 with appropriate attribution.

## License

Project source code is distributed under the repository's `LICENSE` file.
