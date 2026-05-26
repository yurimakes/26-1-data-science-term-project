# 26-1-data-science-term-project

Bank Marketing classification project for Data Science term project.

## Project Topic

Bank Marketing Classification Project

This project predicts whether a customer will subscribe to a fixed-term deposit using the Bank Marketing dataset. The project includes data preprocessing, model training, model evaluation, and Open-Source SW documentation.

## Team Members

- Shin In-seub: Topic Management and Project Direction, Presentation, PPT, and script preparation
- Jo Yoon-sung: Objective Setting, Data Preprocessing, and Report Writing
- Moon Ji-seok: Algorithms, Modeling, and Report Writing
- Choi Yu-ri: Evaluation, Conclusion, Open Source SW (GitHub), and Report Writing
- Han Sang-hoo

## Main Features

- Data cleaning and preprocessing
- Categorical encoding and numerical feature scaling
- Logistic Regression and Random Forest modeling
- Stratified K-Fold Cross-Validation
- Multi-metric evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Confusion matrix visualization

## How to Run

Install the required packages:

pip install -r requirements.txt

Run preprocessing:

python src/preprocess.py

Run modeling:

python src/predict.py

Run evaluation:

python src/evaluation.py

Optional > 
run step-by-step preprocessing and scaling verification:

python src/TermProject.py

## Output Files

The evaluation script generates the following output files:

results/tables/model_comparison_metrics.csv
results/tables/best_model_test_metrics.csv
results/figures/confusion_matrix_best_model.png

## Dataset

The dataset used in this project is the Bank Marketing Dataset from Kaggle / UCI Machine Learning Repository.

Dataset link:
https://www.kaggle.com/datasets/ruthgn/bank-marketing-data-set
