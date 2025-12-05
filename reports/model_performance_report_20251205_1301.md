# Comprehensive Modeling Report
## Supply Chain Late Delivery Prediction

**Generated:** 2025-12-05 13:01:15

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
- **Test F1 Score:** 0.7823
- **Test Accuracy:** 0.7820
- **Test ROC-AUC:** 0.8722
- **Optimal Threshold:** 0.4400
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
| 1 | product_popularity |
| 2 | department_name_encoded |
| 3 | discount_rate |
| 4 | order_value |
| 5 | order_month |
| 6 | scheduled_shipping_days |
| 7 | is_weekend |
| 8 | sales_per_item |
| 9 | market_encoded |
| 10 | order_state_encoded |
| 11 | shipping_mode_encoded |
| 12 | is_high_value |
| 13 | order_quarter |
| 14 | order_country_encoded |
| 15 | shipping_urgency |
| 16 | category_popularity |
| 17 | customer_order_count |
| 18 | profit_margin_pct |
| 19 | order_day_of_week |
| 20 | type_encoded |
| 21 | order_item_quantity |
| 22 | category_name_encoded |
| 23 | order_region_encoded |
| 24 | order_city_encoded |
| 25 | customer_segment_encoded |
| 26 | customer_lifetime_value |

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
- `learning_rate`: 0.2536999076681771
- `max_depth`: 8
- `min_samples_leaf`: 2
- `min_samples_split`: 13
- `n_estimators`: 144
- `subsample`: 0.662397808134481

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
- `iterations`: 102
- `learning_rate`: 0.2871240081991256
- `subsample`: 0.9044273299427488

#### LightGBM

**Key Parameters:**
- `colsample_bytree`: 0.8583425762812965
- `learning_rate`: 0.29666576064052386
- `max_depth`: 9
- `n_estimators`: 183
- `num_leaves`: 31
- `subsample`: 0.9148886768311731

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
| XGBoost | 0.9194 | 0.7820 | 0.9196 | 0.7823 | 0.8722 | 0.1374 | ⚠️ OVERFITTING |
| Random Forest | 0.9303 | 0.7720 | 0.9304 | 0.7712 | 0.8856 | 0.1582 | ⚠️ OVERFITTING |
| CatBoost | 0.8358 | 0.7706 | 0.8361 | 0.7709 | 0.8595 | 0.0652 | ⚠️ OVERFITTING |
| Gradient Boosting | 0.8691 | 0.7543 | 0.8694 | 0.7548 | 0.8361 | 0.1148 | ⚠️ OVERFITTING |
| LightGBM | 0.7800 | 0.7340 | 0.7796 | 0.7333 | 0.8177 | 0.0460 | ✅ GOOD FIT |
| Stacking Ensemble | 0.7220 | 0.7098 | 0.7197 | 0.7075 | 0.7804 | 0.0121 | ✅ GOOD FIT |
| Extra Trees | 0.7584 | 0.7103 | 0.7560 | 0.7057 | 0.8233 | 0.0481 | ✅ GOOD FIT |
| Voting Ensemble | 0.7090 | 0.7050 | 0.7045 | 0.7004 | 0.7791 | 0.0040 | ✅ GOOD FIT |
| Logistic Regression | 0.6875 | 0.6891 | 0.6850 | 0.6869 | 0.7070 | -0.0017 | ✅ GOOD FIT |

## Detailed Model Metrics

### XGBoost

**Training Metrics:**
- Accuracy: 0.9194
- Precision: 0.9219
- Recall: 0.9194
- F1 Score: 0.9196
- ROC-AUC: 0.9818

**Test Metrics:**
- Accuracy: 0.7820
- Precision: 0.7912
- Recall: 0.7820
- F1 Score: 0.7823
- ROC-AUC: 0.8722

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.1374
- F1 Gap (Train - Test): 0.1372
- Status: ⚠️ OVERFITTING

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13698  2610
       Late       5262  14534
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,698 (37.94%) - Correctly predicted on-time
- False Positives (FP): 2,610 (7.23%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 5,262 (14.57%) - Missed late deliveries (Type II error)
- True Positives (TP): 14,534 (40.26%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8478 - Of predicted late, 84.78% are actually late
- Recall (Late): 0.7342 - Of actual late, 73.42% are correctly identified
- Specificity (On-time): 0.8400 - Of actual on-time, 84.00% are correctly identified

**Optimal Threshold:** 0.4400
- F1 Score with Optimal Threshold: 0.7877

---

### Random Forest

**Training Metrics:**
- Accuracy: 0.9303
- Precision: 0.9328
- Recall: 0.9303
- F1 Score: 0.9304
- ROC-AUC: 0.9892

**Test Metrics:**
- Accuracy: 0.7720
- Precision: 0.7950
- Recall: 0.7720
- F1 Score: 0.7712
- ROC-AUC: 0.8856

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.1582
- F1 Gap (Train - Test): 0.1592
- Status: ⚠️ OVERFITTING

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   14444  1864
       Late       6367  13429
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 14,444 (40.01%) - Correctly predicted on-time
- False Positives (FP): 1,864 (5.16%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 6,367 (17.64%) - Missed late deliveries (Type II error)
- True Positives (TP): 13,429 (37.20%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8781 - Of predicted late, 87.81% are actually late
- Recall (Late): 0.6784 - Of actual late, 67.84% are correctly identified
- Specificity (On-time): 0.8857 - Of actual on-time, 88.57% are correctly identified

**Optimal Threshold:** 0.4200
- F1 Score with Optimal Threshold: 0.8041

---

### CatBoost

**Training Metrics:**
- Accuracy: 0.8358
- Precision: 0.8437
- Recall: 0.8358
- F1 Score: 0.8361
- ROC-AUC: 0.9288

**Test Metrics:**
- Accuracy: 0.7706
- Precision: 0.7805
- Recall: 0.7706
- F1 Score: 0.7709
- ROC-AUC: 0.8595

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.0652
- F1 Gap (Train - Test): 0.0652
- Status: ⚠️ OVERFITTING

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13562  2746
       Late       5538  14258
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,562 (37.56%) - Correctly predicted on-time
- False Positives (FP): 2,746 (7.61%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 5,538 (15.34%) - Missed late deliveries (Type II error)
- True Positives (TP): 14,258 (39.49%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8385 - Of predicted late, 83.85% are actually late
- Recall (Late): 0.7202 - Of actual late, 72.02% are correctly identified
- Specificity (On-time): 0.8316 - Of actual on-time, 83.16% are correctly identified

**Optimal Threshold:** 0.4600
- F1 Score with Optimal Threshold: 0.7751

---

### Gradient Boosting

**Training Metrics:**
- Accuracy: 0.8691
- Precision: 0.8733
- Recall: 0.8691
- F1 Score: 0.8694
- ROC-AUC: 0.9489

**Test Metrics:**
- Accuracy: 0.7543
- Precision: 0.7613
- Recall: 0.7543
- F1 Score: 0.7548
- ROC-AUC: 0.8361

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.1148
- F1 Gap (Train - Test): 0.1146
- Status: ⚠️ OVERFITTING

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13005  3303
       Late       5568  14228
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,005 (36.02%) - Correctly predicted on-time
- False Positives (FP): 3,303 (9.15%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 5,568 (15.42%) - Missed late deliveries (Type II error)
- True Positives (TP): 14,228 (39.41%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8116 - Of predicted late, 81.16% are actually late
- Recall (Late): 0.7187 - Of actual late, 71.87% are correctly identified
- Specificity (On-time): 0.7975 - Of actual on-time, 79.75% are correctly identified

**Optimal Threshold:** 0.4900
- F1 Score with Optimal Threshold: 0.7557

---

### LightGBM

**Training Metrics:**
- Accuracy: 0.7800
- Precision: 0.7995
- Recall: 0.7800
- F1 Score: 0.7796
- ROC-AUC: 0.8831

**Test Metrics:**
- Accuracy: 0.7340
- Precision: 0.7538
- Recall: 0.7340
- F1 Score: 0.7333
- ROC-AUC: 0.8177

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.0460
- F1 Gap (Train - Test): 0.0463
- Status: ✅ GOOD FIT

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13653  2655
       Late       6948  12848
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,653 (37.82%) - Correctly predicted on-time
- False Positives (FP): 2,655 (7.35%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 6,948 (19.24%) - Missed late deliveries (Type II error)
- True Positives (TP): 12,848 (35.59%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8287 - Of predicted late, 82.87% are actually late
- Recall (Late): 0.6490 - Of actual late, 64.90% are correctly identified
- Specificity (On-time): 0.8372 - Of actual on-time, 83.72% are correctly identified

**Optimal Threshold:** 0.4700
- F1 Score with Optimal Threshold: 0.7372

---

### Stacking Ensemble

**Training Metrics:**
- Accuracy: 0.7220
- Precision: 0.7512
- Recall: 0.7220
- F1 Score: 0.7197
- ROC-AUC: 0.8129

**Test Metrics:**
- Accuracy: 0.7098
- Precision: 0.7384
- Recall: 0.7098
- F1 Score: 0.7075
- ROC-AUC: 0.7804

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.0121
- F1 Gap (Train - Test): 0.0123
- Status: ✅ GOOD FIT

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13784  2524
       Late       7952  11844
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,784 (38.18%) - Correctly predicted on-time
- False Positives (FP): 2,524 (6.99%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 7,952 (22.03%) - Missed late deliveries (Type II error)
- True Positives (TP): 11,844 (32.81%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8243 - Of predicted late, 82.43% are actually late
- Recall (Late): 0.5983 - Of actual late, 59.83% are correctly identified
- Specificity (On-time): 0.8452 - Of actual on-time, 84.52% are correctly identified

**Optimal Threshold:** 0.4600
- F1 Score with Optimal Threshold: 0.7094

---

### Extra Trees

**Training Metrics:**
- Accuracy: 0.7584
- Precision: 0.7941
- Recall: 0.7584
- F1 Score: 0.7560
- ROC-AUC: 0.9507

**Test Metrics:**
- Accuracy: 0.7103
- Precision: 0.7516
- Recall: 0.7103
- F1 Score: 0.7057
- ROC-AUC: 0.8233

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.0481
- F1 Gap (Train - Test): 0.0503
- Status: ✅ GOOD FIT

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   14378  1930
       Late       8530  11266
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 14,378 (39.82%) - Correctly predicted on-time
- False Positives (FP): 1,930 (5.35%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 8,530 (23.63%) - Missed late deliveries (Type II error)
- True Positives (TP): 11,266 (31.20%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8537 - Of predicted late, 85.37% are actually late
- Recall (Late): 0.5691 - Of actual late, 56.91% are correctly identified
- Specificity (On-time): 0.8817 - Of actual on-time, 88.17% are correctly identified

**Optimal Threshold:** 0.3800
- F1 Score with Optimal Threshold: 0.7408

---

### Voting Ensemble

**Training Metrics:**
- Accuracy: 0.7090
- Precision: 0.7493
- Recall: 0.7090
- F1 Score: 0.7045
- ROC-AUC: 0.8152

**Test Metrics:**
- Accuracy: 0.7050
- Precision: 0.7454
- Recall: 0.7050
- F1 Score: 0.7004
- ROC-AUC: 0.7791

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): 0.0040
- F1 Gap (Train - Test): 0.0042
- Status: ✅ GOOD FIT

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   14267  2041
       Late       8610  11186
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 14,267 (39.52%) - Correctly predicted on-time
- False Positives (FP): 2,041 (5.65%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 8,610 (23.85%) - Missed late deliveries (Type II error)
- True Positives (TP): 11,186 (30.98%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.8457 - Of predicted late, 84.57% are actually late
- Recall (Late): 0.5651 - Of actual late, 56.51% are correctly identified
- Specificity (On-time): 0.8748 - Of actual on-time, 87.48% are correctly identified

**Optimal Threshold:** 0.4100
- F1 Score with Optimal Threshold: 0.7075

---

### Logistic Regression

**Training Metrics:**
- Accuracy: 0.6875
- Precision: 0.7141
- Recall: 0.6875
- F1 Score: 0.6850
- ROC-AUC: 0.7035

**Test Metrics:**
- Accuracy: 0.6891
- Precision: 0.7152
- Recall: 0.6891
- F1 Score: 0.6869
- ROC-AUC: 0.7070

**Overfitting Analysis:**
- Accuracy Gap (Train - Test): -0.0017
- F1 Gap (Train - Test): -0.0018
- Status: ✅ GOOD FIT

**Confusion Matrix:**
```
                Predicted
              On-time  Late
Actual On-time   13341  2967
       Late       8256  11540
```

**Confusion Matrix Breakdown:**
- True Negatives (TN): 13,341 (36.95%) - Correctly predicted on-time
- False Positives (FP): 2,967 (8.22%) - Incorrectly predicted late (Type I error)
- False Negatives (FN): 8,256 (22.87%) - Missed late deliveries (Type II error)
- True Positives (TP): 11,540 (31.96%) - Correctly predicted late

**Per-Class Metrics:**
- Precision (Late): 0.7955 - Of predicted late, 79.55% are actually late
- Recall (Late): 0.5829 - Of actual late, 58.29% are correctly identified
- Specificity (On-time): 0.8181 - Of actual on-time, 81.81% are correctly identified

**Optimal Threshold:** 0.4600
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
| CatBoost | 0.4600 | 0.7709 | 0.7751 | +0.0042 (+0.54%) |
| Extra Trees | 0.3800 | 0.7057 | 0.7408 | +0.0351 (+4.98%) |
| Gradient Boosting | 0.4900 | 0.7548 | 0.7557 | +0.0009 (+0.12%) |
| LightGBM | 0.4700 | 0.7333 | 0.7372 | +0.0038 (+0.52%) |
| Logistic Regression | 0.4600 | 0.6869 | 0.6885 | +0.0016 (+0.23%) |
| Random Forest | 0.4200 | 0.7712 | 0.8041 | +0.0329 (+4.26%) |
| Stacking Ensemble | 0.4600 | 0.7075 | 0.7094 | +0.0019 (+0.27%) |
| Voting Ensemble | 0.4100 | 0.7004 | 0.7075 | +0.0071 (+1.02%) |
| XGBoost | 0.4400 | 0.7823 | 0.7877 | +0.0054 (+0.69%) |

**Best Threshold Improvements:**
- Extra Trees: +0.0351 F1 improvement with threshold 0.3800
- Random Forest: +0.0329 F1 improvement with threshold 0.4200
- Voting Ensemble: +0.0071 F1 improvement with threshold 0.4100
- XGBoost: +0.0054 F1 improvement with threshold 0.4400
- CatBoost: +0.0042 F1 improvement with threshold 0.4600

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
- Random Forest: Train-Test accuracy gap of 0.1582
- Gradient Boosting: Train-Test accuracy gap of 0.1148
- XGBoost: Train-Test accuracy gap of 0.1374
- CatBoost: Train-Test accuracy gap of 0.0652

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

- `learning_rate`: 0.2536999076681771
- `max_depth`: 8
- `min_samples_leaf`: 2
- `min_samples_split`: 13
- `n_estimators`: 144
- `subsample`: 0.662397808134481

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
- `iterations`: 102
- `learning_rate`: 0.2871240081991256
- `subsample`: 0.9044273299427488

#### LightGBM

- `colsample_bytree`: 0.8583425762812965
- `learning_rate`: 0.29666576064052386
- `max_depth`: 9
- `n_estimators`: 183
- `num_leaves`: 31
- `reg_alpha`: 0.0
- `reg_lambda`: 0.0
- `subsample`: 0.9148886768311731

## Confusion Matrix Analysis

### Summary Statistics

| Model | TN | FP | FN | TP | Precision | Recall | Specificity |
|-------|----|----|----|----|-----------|--------|-------------|
| XGBoost | 13,698 | 2,610 | 5,262 | 14,534 | 0.8478 | 0.7342 | 0.8400 |
| Random Forest | 14,444 | 1,864 | 6,367 | 13,429 | 0.8781 | 0.6784 | 0.8857 |
| CatBoost | 13,562 | 2,746 | 5,538 | 14,258 | 0.8385 | 0.7202 | 0.8316 |
| Gradient Boosting | 13,005 | 3,303 | 5,568 | 14,228 | 0.8116 | 0.7187 | 0.7975 |
| LightGBM | 13,653 | 2,655 | 6,948 | 12,848 | 0.8287 | 0.6490 | 0.8372 |
| Stacking Ensemble | 13,784 | 2,524 | 7,952 | 11,844 | 0.8243 | 0.5983 | 0.8452 |
| Extra Trees | 14,378 | 1,930 | 8,530 | 11,266 | 0.8537 | 0.5691 | 0.8817 |
| Voting Ensemble | 14,267 | 2,041 | 8,610 | 11,186 | 0.8457 | 0.5651 | 0.8748 |
| Logistic Regression | 13,341 | 2,967 | 8,256 | 11,540 | 0.7955 | 0.5829 | 0.8181 |

## Feature Importance

### Feature Importance for Best Model (XGBoost)

| Rank | Feature | Importance | Cumulative % |
|------|---------|------------|---------------|
| 1 | shipping_urgency | 0.5326 | 53.26% |
| 2 | scheduled_shipping_days | 0.2336 | 76.62% |
| 3 | shipping_mode_encoded | 0.1324 | 89.86% |
| 4 | type_encoded | 0.0192 | 91.78% |
| 5 | customer_lifetime_value | 0.0048 | 92.26% |
| 6 | order_city_encoded | 0.0047 | 92.73% |
| 7 | order_state_encoded | 0.0046 | 93.19% |
| 8 | order_month | 0.0046 | 93.65% |
| 9 | customer_order_count | 0.0045 | 94.09% |
| 10 | order_country_encoded | 0.0045 | 94.54% |
| 11 | order_day_of_week | 0.0044 | 94.98% |
| 12 | is_weekend | 0.0042 | 95.40% |
| 13 | order_region_encoded | 0.0042 | 95.83% |
| 14 | customer_segment_encoded | 0.0042 | 96.25% |
| 15 | market_encoded | 0.0041 | 96.65% |
| 16 | order_quarter | 0.0041 | 97.06% |
| 17 | is_high_value | 0.0035 | 97.40% |
| 18 | category_popularity | 0.0030 | 97.71% |
| 19 | sales_per_item | 0.0030 | 98.01% |
| 20 | profit_margin_pct | 0.0030 | 98.31% |

**Insights:**
- Top 5 features account for 92.26% of total importance
- Top 10 features account for 94.54% of total importance

### Feature Importance Comparison (Top 10)

Comparing top features across different models:

**Most Important Features (appearing in top 10 of multiple models):**
- customer_lifetime_value (appears in 6 models' top 10)
- order_city_encoded (appears in 6 models' top 10)
- order_state_encoded (appears in 6 models' top 10)
- order_country_encoded (appears in 6 models' top 10)
- shipping_urgency (appears in 5 models' top 10)
- customer_order_count (appears in 5 models' top 10)
- order_day_of_week (appears in 4 models' top 10)
- shipping_mode_encoded (appears in 4 models' top 10)
- scheduled_shipping_days (appears in 4 models' top 10)
- profit_margin_pct (appears in 3 models' top 10)

## Recommendations

### Best Model for Production
- **Recommended:** XGBoost
- **⚠️ WARNING:** Selected model shows overfitting
  - Test F1 Score: 0.7823
  - Accuracy Gap: 0.1374 (high gap indicates overfitting)
  - **Recommendation:** Consider using a model with GOOD FIT status if available
- **Use threshold:** 0.4400 (instead of default 0.5)

### Overfitting Concerns
The following models show signs of overfitting:
- **Random Forest**: Accuracy gap of 0.1582
- **Gradient Boosting**: Accuracy gap of 0.1148
- **XGBoost**: Accuracy gap of 0.1374
- **CatBoost**: Accuracy gap of 0.0652

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

*Report generated by Supply Chain ML Pipeline on 2025-12-05 13:01:15*

*For questions or issues, refer to the project documentation.*
