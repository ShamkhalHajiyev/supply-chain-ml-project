# Comprehensive Modeling Report
## Supply Chain Late Delivery Prediction

**Generated:** 2025-12-05 12:20:38

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Dataset Information](#dataset-information)
3. [Feature Engineering](#feature-engineering)
4. [Model Configurations](#model-configurations)
5. [Training Process](#training-process)
6. [Hyperparameter Tuning](#hyperparameter-tuning)
7. [Model Performance Comparison](#model-performance-comparison)
8. [Detailed Model Metrics](#detailed-model-metrics)
9. [Confusion Matrix Analysis](#confusion-matrix-analysis)
10. [Feature Importance](#feature-importance)
11. [Threshold Optimization](#threshold-optimization)
12. [Model Evaluation](#model-evaluation)
13. [Recommendations](#recommendations)
14. [Appendix](#appendix)

---

## Executive Summary

**Best Model:** XGBoost
- **Test F1 Score:** 0.7775
- **Test Accuracy:** 0.7772
- **Test ROC-AUC:** 0.8677
- **Optimal Threshold:** 0.4600
- **Fit Status:** ⚠️ OVERFITTING

**Modeling Summary:**
- Total models trained: 9
- Models with hyperparameter tuning: 6
- Models with threshold optimization: 9

## Dataset Information

### Training Set
- **Samples:** 126,363
- **Features:** 26
- **Class Distribution:**
  - On-time: 57,080 (45.17%)
  - Late Delivery: 69,283 (54.83%)

### Validation Set
- **Samples:** 18,052
- **Class Distribution:**
  - On-time: 8,154 (45.17%)
  - Late Delivery: 9,898 (54.83%)

### Test Set
- **Samples:** 36,104
- **Class Distribution:**
  - On-time: 16,308 (45.17%)
  - Late Delivery: 19,796 (54.83%)

## Feature Engineering

### Feature Overview
- **Total Features:** 26
- **Feature Types:** Engineered features from raw supply chain data

### Feature List
| # | Feature Name |
|---|--------------|
| 1 | department_name_encoded |
| 2 | sales_per_item |
| 3 | profit_margin_pct |
| 4 | order_item_quantity |
| 5 | customer_segment_encoded |
| 6 | is_high_value |
| 7 | order_day_of_week |
| 8 | shipping_urgency |
| 9 | category_popularity |
| 10 | order_region_encoded |
| 11 | order_value |
| 12 | order_month |
| 13 | is_weekend |
| 14 | scheduled_shipping_days |
| 15 | category_name_encoded |
| 16 | customer_lifetime_value |
| 17 | order_state_encoded |
| 18 | order_quarter |
| 19 | customer_order_count |
| 20 | order_country_encoded |
| 21 | order_city_encoded |
| 22 | discount_rate |
| 23 | product_popularity |
| 24 | type_encoded |
| 25 | shipping_mode_encoded |
| 26 | market_encoded |

### Data Leakage Prevention

The following columns were **excluded** to prevent data leakage:
- `late_delivery_risk` - This IS the target variable
- `delivery_status` - Categorical form of target
- `days_for_shipping_(real)` - Only known after delivery
- `shipping_date_(dateorders)` - Actual shipping date (post-hoc)

**Note:** Models achieve realistic ~70-80% accuracy because leaky features are properly excluded.

## Model Configurations

### Models Trained

| Model | Type | Description |
|-------|------|-------------|
| Logistic Regression | Baseline Linear | Linear baseline model |
| Random Forest | Ensemble Tree | Tree-based ensemble |
| Extra Trees | Ensemble Tree | Tree-based ensemble |
| Gradient Boosting | Boosting | Gradient boosting model |
| XGBoost | Gradient Boosting | Advanced gradient boosting |
| CatBoost | Gradient Boosting | Advanced gradient boosting |
| LightGBM | Gradient Boosting | Advanced gradient boosting |
| Voting Ensemble | Meta Ensemble | Combines multiple base models |
| Stacking Ensemble | Meta Ensemble | Combines multiple base models |

### Model Hyperparameters

#### Logistic Regression

**Key Parameters:**
- `C`: 1.0
- `solver`: lbfgs

#### Random Forest

**Key Parameters:**
- `max_depth`: 28
- `min_samples_leaf`: 7
- `min_samples_split`: 6
- `n_estimators`: 58

#### Extra Trees

**Key Parameters:**
- `max_depth`: 18
- `min_samples_leaf`: 3
- `min_samples_split`: 10
- `n_estimators`: 126

#### Gradient Boosting

**Key Parameters:**
- `learning_rate`: 0.06690992453172911
- `max_depth`: 10
- `min_samples_leaf`: 7
- `min_samples_split`: 8
- `n_estimators`: 287
- `subsample`: 0.6154900799844732

#### XGBoost

**Key Parameters:**
- `colsample_bytree`: 0.6624074561769746
- `learning_rate`: 0.1205712628744377
- `max_depth`: 10
- `n_estimators`: 175
- `subsample`: 0.8394633936788146

#### CatBoost

**Key Parameters:**
- `depth`: 10
- `iterations`: 104
- `learning_rate`: 0.27047297227177763
- `subsample`: 0.8838559385524739

#### LightGBM

**Key Parameters:**
- `colsample_bytree`: 0.8953622178900084
- `learning_rate`: 0.28009047436880896
- `max_depth`: 10
- `n_estimators`: 177
- `num_leaves`: 31
- `subsample`: 0.6159269215693287

#### Voting Ensemble

*Using default parameters*

#### Stacking Ensemble

*Using default parameters*

## Training Process

### Training Methodology

- **Random State:** 42 (for reproducibility)
- **Parallel Training:** Enabled
- **CPU Cores Used:** All available cores
- **Data Split:** Train (70%) / Validation (10%) / Test (20%)
- **Stratification:** Enabled (maintains class distribution)

## Model Performance Comparison

| Model | Train Acc | Test Acc | Train F1 | Test F1 | Test ROC-AUC | Acc Gap | Status |
|-------|-----------|----------|----------|---------|--------------|---------|--------|
| XGBoost | 0.9139 | 0.7772 | 0.9141 | 0.7775 | 0.8677 | 0.1367 | ⚠️ OVERFITTING |
| Gradient Boosting | 0.9123 | 0.7764 | 0.9125 | 0.7766 | 0.8677 | 0.1359 | ⚠️ OVERFITTING |
| Random Forest | 0.9302 | 0.7716 | 0.9304 | 0.7708 | 0.8856 | 0.1586 | ⚠️ OVERFITTING |
| CatBoost | 0.8360 | 0.7661 | 0.8363 | 0.7664 | 0.8542 | 0.0698 | ⚠️ OVERFITTING |
| LightGBM | 0.7762 | 0.7352 | 0.7757 | 0.7345 | 0.8156 | 0.0410 | ✅ GOOD FIT |
| Stacking Ensemble | 0.7208 | 0.7099 | 0.7186 | 0.7074 | 0.7799 | 0.0110 | ✅ GOOD FIT |
| Extra Trees | 0.7570 | 0.7101 | 0.7545 | 0.7055 | 0.8226 | 0.0469 | ✅ GOOD FIT |
| Voting Ensemble | 0.7086 | 0.7047 | 0.7041 | 0.7000 | 0.7791 | 0.0040 | ✅ GOOD FIT |
| Logistic Regression | 0.6875 | 0.6893 | 0.6851 | 0.6870 | 0.7071 | -0.0017 | ✅ GOOD FIT |

## Detailed Model Metrics

### XGBoost

**Training Metrics:**
- Accuracy: 0.9139
- Precision: 0.9169
- Recall: 0.9139
- F1 Score: 0.9141
- ROC-AUC: 0.9797

**Test Metrics:**
- Accuracy: 0.7772
- Precision: 0.7870
- Recall: 0.7772
- F1 Score: 0.7775
- ROC-AUC: 0.8677

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.1367
- F1 Gap (Train - Test): 0.1366
- Status: ⚠️ OVERFITTING

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13667  2641
       Late       5404  14392
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,667 (37.85%) - Correctly predicted on-time
- False Positives (FP): 2,641 (7.31%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 5,404 (14.97%) - Missed late deliveries (Type II error)
- True Positives (TP): 14,392 (39.86%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8449 - Of predicted late, 84.49% are actually late
- Recall (Late): 0.7270 - Of actual late, 72.70% are correctly identified
- Specificity (On-time): 0.8381 - Of actual on-time, 83.81% are correctly identified

**Optimal Threshold:** 0.4600
- F1 Score with Optimal Threshold: 0.7815

---

### Gradient Boosting

**Training Metrics:**
- Accuracy: 0.9123
- Precision: 0.9153
- Recall: 0.9123
- F1 Score: 0.9125
- ROC-AUC: 0.9801

**Test Metrics:**
- Accuracy: 0.7764
- Precision: 0.7877
- Recall: 0.7764
- F1 Score: 0.7766
- ROC-AUC: 0.8677

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.1359
- F1 Gap (Train - Test): 0.1359
- Status: ⚠️ OVERFITTING

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13777  2531
       Late       5542  14254
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,777 (38.16%) - Correctly predicted on-time
- False Positives (FP): 2,531 (7.01%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 5,542 (15.35%) - Missed late deliveries (Type II error)
- True Positives (TP): 14,254 (39.48%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8492 - Of predicted late, 84.92% are actually late
- Recall (Late): 0.7200 - Of actual late, 72.00% are correctly identified
- Specificity (On-time): 0.8448 - Of actual on-time, 84.48% are correctly identified

**Optimal Threshold:** 0.4500
- F1 Score with Optimal Threshold: 0.7844

---

### Random Forest

**Training Metrics:**
- Accuracy: 0.9302
- Precision: 0.9327
- Recall: 0.9302
- F1 Score: 0.9304
- ROC-AUC: 0.9892

**Test Metrics:**
- Accuracy: 0.7716
- Precision: 0.7948
- Recall: 0.7716
- F1 Score: 0.7708
- ROC-AUC: 0.8856

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.1586
- F1 Gap (Train - Test): 0.1596
- Status: ⚠️ OVERFITTING

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   14448  1860
       Late       6386  13410
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 14,448 (40.02%) - Correctly predicted on-time
- False Positives (FP): 1,860 (5.15%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 6,386 (17.69%) - Missed late deliveries (Type II error)
- True Positives (TP): 13,410 (37.14%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8782 - Of predicted late, 87.82% are actually late
- Recall (Late): 0.6774 - Of actual late, 67.74% are correctly identified
- Specificity (On-time): 0.8859 - Of actual on-time, 88.59% are correctly identified

**Optimal Threshold:** 0.4200
- F1 Score with Optimal Threshold: 0.8046

---

### CatBoost

**Training Metrics:**
- Accuracy: 0.8360
- Precision: 0.8449
- Recall: 0.8360
- F1 Score: 0.8363
- ROC-AUC: 0.9295

**Test Metrics:**
- Accuracy: 0.7661
- Precision: 0.7772
- Recall: 0.7661
- F1 Score: 0.7664
- ROC-AUC: 0.8542

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.0698
- F1 Gap (Train - Test): 0.0699
- Status: ⚠️ OVERFITTING

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13582  2726
       Late       5718  14078
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,582 (37.62%) - Correctly predicted on-time
- False Positives (FP): 2,726 (7.55%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 5,718 (15.84%) - Missed late deliveries (Type II error)
- True Positives (TP): 14,078 (38.99%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8378 - Of predicted late, 83.78% are actually late
- Recall (Late): 0.7112 - Of actual late, 71.12% are correctly identified
- Specificity (On-time): 0.8328 - Of actual on-time, 83.28% are correctly identified

**Optimal Threshold:** 0.4700
- F1 Score with Optimal Threshold: 0.7706

---

### LightGBM

**Training Metrics:**
- Accuracy: 0.7762
- Precision: 0.7961
- Recall: 0.7762
- F1 Score: 0.7757
- ROC-AUC: 0.8788

**Test Metrics:**
- Accuracy: 0.7352
- Precision: 0.7553
- Recall: 0.7352
- F1 Score: 0.7345
- ROC-AUC: 0.8156

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.0410
- F1 Gap (Train - Test): 0.0412
- Status: ✅ GOOD FIT

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13691  2617
       Late       6942  12854
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,691 (37.92%) - Correctly predicted on-time
- False Positives (FP): 2,617 (7.25%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 6,942 (19.23%) - Missed late deliveries (Type II error)
- True Positives (TP): 12,854 (35.60%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8308 - Of predicted late, 83.08% are actually late
- Recall (Late): 0.6493 - Of actual late, 64.93% are correctly identified
- Specificity (On-time): 0.8395 - Of actual on-time, 83.95% are correctly identified

**Optimal Threshold:** 0.4500
- F1 Score with Optimal Threshold: 0.7373

---

### Stacking Ensemble

**Training Metrics:**
- Accuracy: 0.7208
- Precision: 0.7502
- Recall: 0.7208
- F1 Score: 0.7186
- ROC-AUC: 0.8116

**Test Metrics:**
- Accuracy: 0.7099
- Precision: 0.7389
- Recall: 0.7099
- F1 Score: 0.7074
- ROC-AUC: 0.7799

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.0110
- F1 Gap (Train - Test): 0.0111
- Status: ✅ GOOD FIT

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13806  2502
       Late       7973  11823
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,806 (38.24%) - Correctly predicted on-time
- False Positives (FP): 2,502 (6.93%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 7,973 (22.08%) - Missed late deliveries (Type II error)
- True Positives (TP): 11,823 (32.75%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8253 - Of predicted late, 82.53% are actually late
- Recall (Late): 0.5972 - Of actual late, 59.72% are correctly identified
- Specificity (On-time): 0.8466 - Of actual on-time, 84.66% are correctly identified

**Optimal Threshold:** 0.4400
- F1 Score with Optimal Threshold: 0.7087

---

### Extra Trees

**Training Metrics:**
- Accuracy: 0.7570
- Precision: 0.7933
- Recall: 0.7570
- F1 Score: 0.7545
- ROC-AUC: 0.9502

**Test Metrics:**
- Accuracy: 0.7101
- Precision: 0.7519
- Recall: 0.7101
- F1 Score: 0.7055
- ROC-AUC: 0.8226

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.0469
- F1 Gap (Train - Test): 0.0490
- Status: ✅ GOOD FIT

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   14393  1915
       Late       8550  11246
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 14,393 (39.87%) - Correctly predicted on-time
- False Positives (FP): 1,915 (5.30%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 8,550 (23.68%) - Missed late deliveries (Type II error)
- True Positives (TP): 11,246 (31.15%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8545 - Of predicted late, 85.45% are actually late
- Recall (Late): 0.5681 - Of actual late, 56.81% are correctly identified
- Specificity (On-time): 0.8826 - Of actual on-time, 88.26% are correctly identified

**Optimal Threshold:** 0.3700
- F1 Score with Optimal Threshold: 0.7397

---

### Voting Ensemble

**Training Metrics:**
- Accuracy: 0.7086
- Precision: 0.7491
- Recall: 0.7086
- F1 Score: 0.7041
- ROC-AUC: 0.8159

**Test Metrics:**
- Accuracy: 0.7047
- Precision: 0.7452
- Recall: 0.7047
- F1 Score: 0.7000
- ROC-AUC: 0.7791

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.0040
- F1 Gap (Train - Test): 0.0041
- Status: ✅ GOOD FIT

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   14267  2041
       Late       8621  11175
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 14,267 (39.52%) - Correctly predicted on-time
- False Positives (FP): 2,041 (5.65%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 8,621 (23.88%) - Missed late deliveries (Type II error)
- True Positives (TP): 11,175 (30.95%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8456 - Of predicted late, 84.56% are actually late
- Recall (Late): 0.5645 - Of actual late, 56.45% are correctly identified
- Specificity (On-time): 0.8748 - Of actual on-time, 87.48% are correctly identified

**Optimal Threshold:** 0.4100
- F1 Score with Optimal Threshold: 0.7076

---

### Logistic Regression

**Training Metrics:**
- Accuracy: 0.6875
- Precision: 0.7141
- Recall: 0.6875
- F1 Score: 0.6851
- ROC-AUC: 0.7035

**Test Metrics:**
- Accuracy: 0.6893
- Precision: 0.7152
- Recall: 0.6893
- F1 Score: 0.6870
- ROC-AUC: 0.7071

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): -0.0017
- F1 Gap (Train - Test): -0.0019
- Status: ✅ GOOD FIT

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13339  2969
       Late       8250  11546
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,339 (36.95%) - Correctly predicted on-time
- False Positives (FP): 2,969 (8.22%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 8,250 (22.85%) - Missed late deliveries (Type II error)
- True Positives (TP): 11,546 (31.98%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.7955 - Of predicted late, 79.55% are actually late
- Recall (Late): 0.5832 - Of actual late, 58.32% are correctly identified
- Specificity (On-time): 0.8179 - Of actual on-time, 81.79% are correctly identified

**Optimal Threshold:** 0.4700
- F1 Score with Optimal Threshold: 0.6885

---

## Threshold Optimization

### Optimization Methodology

Classification thresholds were optimized to maximize F1 score on the validation set.
The optimization process:
1. Evaluated thresholds from 0.1 to 0.9 in steps of 0.01
2. Calculated F1 score for each threshold
3. Selected threshold with highest F1 score
4. Applied optimal threshold to test set for final evaluation

### Threshold Optimization Results

| Model | Optimal Threshold | Default (0.5) F1 | Optimized F1 | Improvement |
|-------|-------------------|-----------------|--------------|-------------|
| CatBoost | 0.4700 | 0.7664 | 0.7706 | +0.0042 (+0.55%) |
| Extra Trees | 0.3700 | 0.7055 | 0.7397 | +0.0343 (+4.86%) |
| Gradient Boosting | 0.4500 | 0.7766 | 0.7844 | +0.0077 (+0.99%) |
| LightGBM | 0.4500 | 0.7345 | 0.7373 | +0.0028 (+0.38%) |
| Logistic Regression | 0.4700 | 0.6870 | 0.6885 | +0.0015 (+0.22%) |
| Random Forest | 0.4200 | 0.7708 | 0.8046 | +0.0338 (+4.39%) |
| Stacking Ensemble | 0.4400 | 0.7074 | 0.7087 | +0.0012 (+0.18%) |
| Voting Ensemble | 0.4100 | 0.7000 | 0.7076 | +0.0076 (+1.08%) |
| XGBoost | 0.4600 | 0.7775 | 0.7815 | +0.0039 (+0.51%) |

**Best Threshold Improvements:**
- Extra Trees: +0.0343 F1 improvement with threshold 0.3700
- Random Forest: +0.0338 F1 improvement with threshold 0.4200
- Gradient Boosting: +0.0077 F1 improvement with threshold 0.4500
- Voting Ensemble: +0.0076 F1 improvement with threshold 0.4100
- CatBoost: +0.0042 F1 improvement with threshold 0.4700

## Model Evaluation

### Evaluation Metrics Explained

- **Accuracy:** Overall correctness of predictions
- **Precision:** Of predicted late deliveries, how many are actually late (reduces false alarms)
- **Recall:** Of actual late deliveries, how many are correctly identified (reduces missed deliveries)
- **F1 Score:** Harmonic mean of precision and recall (balanced metric)
- **ROC-AUC:** Area under ROC curve (ability to distinguish between classes)
- **Specificity:** Of actual on-time deliveries, how many are correctly identified

### Overfitting Analysis

**⚠️ Overfitting Detected (4 models):**
- Random Forest: Train-Test accuracy gap of 0.1586
- Gradient Boosting: Train-Test accuracy gap of 0.1359
- XGBoost: Train-Test accuracy gap of 0.1367
- CatBoost: Train-Test accuracy gap of 0.0698

**✅ Good Fit (5 models):**
Models show good generalization with acceptable train-test gap

## Hyperparameter Tuning

### Tuning Summary

The following models were optimized using Optuna (Bayesian optimization):

- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- CatBoost
- LightGBM

### Tuning Methodology

- **Optimization Algorithm:** Tree-structured Parzen Estimator (TPE)
- **Pruning Strategy:** MedianPruner (stops unpromising trials early)
- **Objective Metric:** F1 Score (weighted)
- **Validation:** Fast validation sampling (50% of validation set during tuning)

### Tuned Model Parameters

#### Random Forest

- `max_depth`: 28
- `min_samples_leaf`: 7
- `min_samples_split`: 6
- `n_estimators`: 58

#### Extra Trees

- `max_depth`: 18
- `min_samples_leaf`: 3
- `min_samples_split`: 10
- `n_estimators`: 126

#### Gradient Boosting

- `learning_rate`: 0.06690992453172911
- `max_depth`: 10
- `min_samples_leaf`: 7
- `min_samples_split`: 8
- `n_estimators`: 287
- `subsample`: 0.6154900799844732

#### XGBoost

- `colsample_bytree`: 0.6624074561769746
- `learning_rate`: 0.1205712628744377
- `max_depth`: 10
- `n_estimators`: 175
- `reg_alpha`: None
- `reg_lambda`: None
- `subsample`: 0.8394633936788146

#### CatBoost

- `depth`: 10
- `iterations`: 104
- `learning_rate`: 0.27047297227177763
- `subsample`: 0.8838559385524739

#### LightGBM

- `colsample_bytree`: 0.8953622178900084
- `learning_rate`: 0.28009047436880896
- `max_depth`: 10
- `n_estimators`: 177
- `num_leaves`: 31
- `reg_alpha`: 0.0
- `reg_lambda`: 0.0
- `subsample`: 0.6159269215693287

## Confusion Matrix Analysis

### Summary Statistics

| Model | TN | FP | FN | TP | Precision | Recall | Specificity |
|-------|----|----|----|----|-----------|--------|-------------|
| XGBoost | 13,667 | 2,641 | 5,404 | 14,392 | 0.8449 | 0.7270 | 0.8381 |
| Gradient Boosting | 13,777 | 2,531 | 5,542 | 14,254 | 0.8492 | 0.7200 | 0.8448 |
| Random Forest | 14,448 | 1,860 | 6,386 | 13,410 | 0.8782 | 0.6774 | 0.8859 |
| CatBoost | 13,582 | 2,726 | 5,718 | 14,078 | 0.8378 | 0.7112 | 0.8328 |
| LightGBM | 13,691 | 2,617 | 6,942 | 12,854 | 0.8308 | 0.6493 | 0.8395 |
| Stacking Ensemble | 13,806 | 2,502 | 7,973 | 11,823 | 0.8253 | 0.5972 | 0.8466 |
| Extra Trees | 14,393 | 1,915 | 8,550 | 11,246 | 0.8545 | 0.5681 | 0.8826 |
| Voting Ensemble | 14,267 | 2,041 | 8,621 | 11,175 | 0.8456 | 0.5645 | 0.8748 |
| Logistic Regression | 13,339 | 2,969 | 8,250 | 11,546 | 0.7955 | 0.5832 | 0.8179 |

## Feature Importance

### Feature Importance for Best Model (XGBoost)

| Rank | Feature | Importance | Cumulative % |
|------|---------|------------|---------------|
| 1 | scheduled_shipping_days | 0.3710 | 37.10% |
| 2 | shipping_urgency | 0.2661 | 63.72% |
| 3 | shipping_mode_encoded | 0.2359 | 87.31% |
| 4 | type_encoded | 0.0264 | 89.95% |
| 5 | customer_lifetime_value | 0.0057 | 90.52% |
| 6 | order_city_encoded | 0.0056 | 91.09% |
| 7 | order_country_encoded | 0.0056 | 91.65% |
| 8 | is_high_value | 0.0055 | 92.19% |
| 9 | order_state_encoded | 0.0055 | 92.74% |
| 10 | customer_order_count | 0.0053 | 93.27% |
| 11 | order_region_encoded | 0.0053 | 93.79% |
| 12 | order_day_of_week | 0.0052 | 94.32% |
| 13 | order_month | 0.0052 | 94.84% |
| 14 | customer_segment_encoded | 0.0052 | 95.36% |
| 15 | order_quarter | 0.0052 | 95.88% |
| 16 | market_encoded | 0.0049 | 96.36% |
| 17 | is_weekend | 0.0048 | 96.84% |
| 18 | sales_per_item | 0.0037 | 97.21% |
| 19 | order_value | 0.0036 | 97.57% |
| 20 | product_popularity | 0.0036 | 97.94% |

**Insights:**
- Top 5 features account for 90.52% of total importance
- Top 10 features account for 93.27% of total importance

### Feature Importance Comparison (Top 10)

Comparing top features across different models:

**Most Important Features (appearing in top 10 of multiple models):**
- customer_lifetime_value (appears in 6 models' top 10)
- order_city_encoded (appears in 6 models' top 10)
- order_state_encoded (appears in 6 models' top 10)
- customer_order_count (appears in 6 models' top 10)
- shipping_urgency (appears in 5 models' top 10)
- order_country_encoded (appears in 5 models' top 10)
- shipping_mode_encoded (appears in 4 models' top 10)
- scheduled_shipping_days (appears in 4 models' top 10)
- type_encoded (appears in 4 models' top 10)
- profit_margin_pct (appears in 3 models' top 10)

## Recommendations

### Best Model for Production
- **Recommended:** XGBoost
- **Reason:** Highest test F1 score (0.7775)
- **Use threshold:** 0.4600 (instead of default 0.5)

### Overfitting Concerns
The following models show signs of overfitting:
- **Random Forest**: Accuracy gap of 0.1586
- **Gradient Boosting**: Accuracy gap of 0.1359
- **XGBoost**: Accuracy gap of 0.1367
- **CatBoost**: Accuracy gap of 0.0698

### Model Selection Guidance
- **For highest accuracy:** Use the best model (XGBoost/CatBoost/LightGBM)
- **For interpretability:** Consider Logistic Regression or Random Forest
- **For production stability:** Consider Voting or Stacking Ensemble
- **For fast inference:** LightGBM or XGBoost are recommended

## Appendix

### Model Files

All trained models are saved in the `models/` directory with timestamps.

### Reproducibility

- Random State: 42
- All models use the same random seed for reproducibility
- Data splits are stratified to maintain class distribution

### Technical Details

- **Python Version:** Check with `python --version`
- **Library Versions:** See `pyproject.toml`
- **Training Method:** Parallel execution using joblib
- **Optimization:** Optuna with TPE sampler

---

*Report generated by Supply Chain ML Pipeline on 2025-12-05 12:20:38*

*For questions or issues, refer to the project documentation.*
