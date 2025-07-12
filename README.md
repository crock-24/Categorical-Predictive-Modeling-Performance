# Classification Modeling

This repository contains R code and analysis from a classification modeling assignment for Week 6 of an applied predictive modeling course. It covers two datasets: one for predicting hepatic injury and another for predicting telecom customer churn.

The objective is to build, tune, and evaluate classification models using standard machine learning workflows in R.

---

## Problem 12.1: Hepatic Injury Prediction

### Objective
Predict hepatic injury status using biological predictors from the `hepatic` dataset.

### Methods Used
- Preprocessing: zero-variance filter, correlation filtering
- Stratified train/test split (80/20)
- Models:
  - Linear Discriminant Analysis (LDA)
  - Partial Least Squares Discriminant Analysis (PLS-DA)
  - Penalized Logistic Regression (GLMNET)

### Performance Summary
| Model                          | Tuning Parameters                 | Kappa   |
|-------------------------------|-----------------------------------|---------|
| Linear Discriminant Analysis  | N/A                               | 0.1413  |
| Partial Least Squares (PLS-DA)| Components = 1                    | 0.0714  |
| Penalized Model (GLMNET)      | Alpha = 0.2, Lambda = 0.01        | **0.1642** |

> **Best model:** Penalized GLM with α = 0.2 and λ = 0.01

### Top 5 Important Predictors
From the penalized model:
- Z167
- Z38
- Z42
- Z100
- Z71

---

## Problem 12.3: Churn Prediction

### Objective
Predict telecom customer churn using service and usage data (`mlc_churn` dataset).

### Methods Used
- Removed near-zero variance and highly correlated predictors
- Random train/test split (80/20)
- Models:
  - Logistic Regression
  - Penalized Logistic Regression (GLMNET)
  - Neural Network
  - Mixed Discriminant Analysis (MDA)

### Performance Summary
| Model              | Tuning Parameters               | ROC AUC  |
|-------------------|----------------------------------|----------|
| Logistic Regression| N/A                             | 0.7360   |
| Penalized Model    | Alpha = 0, Lambda = 0.0575      | 0.7647   |
| Neural Network     | Size = 3, Decay = 1             | **0.7984** |
| Mixed Discriminant | Subclasses = 2                  | 0.7913   |

> **Best model:** Neural Network (size = 3, decay = 1)

---

## Tools & Packages

- [`caret`](https://cran.r-project.org/web/packages/caret/index.html) for modeling workflow
- `glmnet` for penalized regression
- `pROC` for ROC analysis
- `nnet` for neural nets
- `corrplot` for correlation heatmaps
- `mda` for mixed discriminant analysis

---

## Notes

- All random sampling is done with `set.seed(100)` for reproducibility.
- Models are evaluated using **Kappa** (multi-class) or **ROC AUC** (binary) as appropriate.
- Only built-in datasets from `AppliedPredictiveModeling` and `modeldata` are used; no external data files are required.
