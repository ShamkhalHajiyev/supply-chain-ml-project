# Model Overview

## Supply Chain Late Delivery Prediction System

---

## 1. Problem Definition

### Prediction Target
- **Target Variable**: `late_delivery_risk` (binary: 1 = late, 0 = on-time)
- **Problem Type**: Binary Classification
- **Business Context**: Predict whether an order will be delivered late **before** the delivery occurs

### Target Distribution
| Class | Count | Percentage |
|-------|-------|------------|
| Late (1) | ~99,000 | 54.8% |
| On-Time (0) | ~81,500 | 45.2% |

The dataset exhibits slight class imbalance favoring the positive (late) class.

---

## 2. Business Objectives

### Primary Goal
Enable proactive intervention for orders predicted to be late, allowing:
- Shipping upgrades to expedite delivery
- Proactive customer communication
- Resource reallocation to high-risk shipments

### Key Business Metrics
| Metric | Business Meaning | Priority |
|--------|------------------|----------|
| **Recall** | Percentage of actual late deliveries caught | High |
| **Precision** | Percentage of predicted late that are actually late | Medium |
| **F1 Score** | Balance between recall and precision | High |
| **ROC-AUC** | Overall discriminative ability | Medium |

### Business Value Targets
- **Cost per late delivery**: ~$75 (customer service, refunds, reputation damage)
- **Cost per intervention**: ~$15 (shipping upgrade, proactive communication)
- **Target**: Reduce late delivery costs by 50%+ through prediction

---

## 3. Data Leakage Prevention (Critical)

### Excluded Features (Post-Outcome Information)
The following features are **excluded** from modeling because they would only be known **after** delivery:

| Feature | Reason for Exclusion |
|---------|---------------------|
| `late_delivery_risk` | This IS the target variable |
| `delivery_status` | Categorical form of target (Advance/Late/On-time/Canceled) |
| `days_for_shipping_(real)` | Actual shipping days - only known after delivery |
| `shipping_date_(dateorders)` | Actual shipping date - post-hoc information |
| `delivery_days` | Calculated from actual delivery date |
| `delivery_status_encoded` | Encoded version of target |

### Why This Matters
Without proper leakage prevention, models can achieve ~100% accuracy by using information that won't be available at prediction time. Our realistic accuracy of ~70-85% reflects genuine predictive power using only pre-delivery information.

---

## 4. Current Modeling Approach

### Model Pipeline
1. **Data Preprocessing**: Clean, encode, and normalize features
2. **Feature Engineering**: Create 26 derived features from raw data
3. **Model Training**: Multiple algorithms including boosting methods
4. **Hyperparameter Tuning**: RandomizedSearchCV / Optuna
5. **Evaluation**: Multi-metric assessment with test set

### Models Currently Implemented
| Model Type | Models | Purpose |
|------------|--------|---------|
| Baseline | Logistic Regression | Establish minimum performance |
| Tree-Based | Decision Tree, Random Forest, Extra Trees | Capture non-linear patterns |
| Boosting | XGBoost, LightGBM, CatBoost, Gradient Boosting | State-of-the-art performance |
| Ensemble | Voting, Stacking | Combine model strengths |

### Current Performance (Approximate)
| Model | Test F1 | Test ROC-AUC |
|-------|---------|--------------|
| Logistic Regression | ~0.69 | ~0.71 |
| Random Forest | ~0.70 | ~0.79 |
| XGBoost | ~0.70 | ~0.78 |
| LightGBM | ~0.70 | ~0.78 |
| CatBoost | ~0.69 | ~0.76 |

---

## 5. Major Limitations of Current Approach

### 1. Feature Selection
- **Issue**: No systematic, automated feature selection with explicit rationale
- **Impact**: May include redundant or low-importance features
- **Solution**: Implement leakage-aware, importance-based feature selection pipeline

### 2. Modeling Sequence
- **Issue**: No clear progression from baseline to tuned to ensemble
- **Impact**: Difficult to attribute improvements to specific changes
- **Solution**: Implement structured sequence: Baseline → Tuning → Threshold → Ensemble → Selection

### 3. LightGBM Focus
- **Issue**: LightGBM present but not leveraged as core model with reusable wrapper
- **Impact**: Missing parallelization benefits and consistent interface
- **Solution**: Create sklearn-compatible LightGBM wrapper with sensible defaults

### 4. Threshold Optimization
- **Issue**: Fixed 0.5 threshold for classification
- **Impact**: Sub-optimal precision/recall tradeoff for business needs
- **Solution**: Implement systematic threshold optimization for business metrics

### 5. Ensembling Strategy
- **Issue**: Basic ensembles without proper CV-based stacking to avoid leakage
- **Impact**: Potential overfitting in meta-learner
- **Solution**: Implement proper CV-based stacking with documented rationale

### 6. Feature Rationale
- **Issue**: No explicit documentation of why features are kept/dropped
- **Impact**: Lack of transparency for stakeholders
- **Solution**: Generate feature selection report with per-feature decisions

### 7. Interpretability
- **Issue**: Basic SHAP without interactive exploration
- **Impact**: Limited ability for stakeholders to explore insights
- **Solution**: Enhanced SHAP notebook with interactivity and business narratives

---

## 6. Recommended Modeling Strategy

### Phase 1: Baseline
- Simple Logistic Regression with balanced class weights
- Establishes minimum performance benchmark
- Documents baseline metrics for comparison

### Phase 2: Single Model Optimization
- **Primary**: LightGBM (fast, handles categoricals, parallel)
- **Secondary**: XGBoost, CatBoost for comparison
- Optuna-based hyperparameter tuning with parallelization

### Phase 3: Threshold Optimization
- Optimize for F1, Youden's J, or business-specific metric
- Generate threshold vs metric curves
- Document optimal threshold with justification

### Phase 4: Ensembling
- Voting ensemble of top 3 models
- CV-based stacking to avoid leakage
- Document ensemble rationale

### Phase 5: Final Selection
- Compare all candidates on validation/test set
- Select based on performance, stability, and business constraints
- Document final choice with reasoning

---

## 7. Feature Categories (from feature_description.md)

### Available at Order Time (Safe for Prediction)
- **Customer Features**: Segment, location, historical behavior
- **Order Features**: Date, quantity, discounts, profit metrics
- **Product Features**: Category, price, popularity
- **Shipping Features**: Mode, scheduled days (NOT actual)
- **Location Features**: Market, region, coordinates

### NOT Available at Order Time (Must Exclude)
- **Delivery Outcomes**: Status, actual shipping days, risk labels
- **Post-Event Timestamps**: Actual shipping date

---

## 8. Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Test F1 Score | > 0.72 | Held-out test set |
| Test ROC-AUC | > 0.80 | Held-out test set |
| Recall (Late Class) | > 60% | Catch majority of late deliveries |
| False Positive Rate | < 20% | Minimize unnecessary interventions |
| Model Stability | Low variance | Cross-validation variance < 0.03 |
| Inference Latency | < 100ms | Production readiness |

---

*Last Updated: 2025-12-05*
*Version: 2.0 - Refactored Pipeline*
