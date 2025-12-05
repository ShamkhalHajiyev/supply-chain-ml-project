# Model Summary

## Supply Chain Late Delivery Prediction System

---

## Executive Summary

This document summarizes the refactored ML pipeline for predicting late deliveries in supply chain operations. The system provides:

- **Problem**: Binary classification of late delivery risk
- **Final Model**: LightGBM with Optuna-tuned hyperparameters
- **Expected Performance**: ~70% F1 Score, ~78% ROC-AUC
- **Business Impact**: Estimated 40-60% reduction in late delivery costs

---

## 1. Modeling Strategy

### Pipeline Architecture

```
Phase 1: Data Loading → Phase 2: Feature Selection → Phase 3: Baseline
                                                           ↓
Phase 8: Final Selection ← Phase 7: Ensembling ← Phase 6: Threshold Optimization
                                      ↑                    ↑
                               Phase 5: Tuning ← Phase 4: Candidate Models
```

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **No Data Leakage** | Automated detection of post-outcome features |
| **Reproducibility** | Fixed random seeds (42) throughout |
| **Parallelization** | Multi-core training with n_jobs=-1 |
| **Explicit Rationale** | Per-feature selection decisions documented |

---

## 2. Why LightGBM as Core Model

### Advantages for This Problem

| Factor | LightGBM Advantage |
|--------|-------------------|
| **Training Speed** | 10-50x faster than traditional boosting |
| **Memory Efficiency** | Histogram-based algorithms reduce memory footprint |
| **Categorical Features** | Native handling without one-hot encoding |
| **Parallelization** | Excellent multi-threading with `num_threads` |
| **Accuracy** | Leaf-wise growth often outperforms level-wise |

### When to Consider Alternatives

- **XGBoost**: When regularization is critical, or for wider framework compatibility
- **CatBoost**: For datasets with many categorical features, or when minimal tuning is required
- **Random Forest**: When interpretability is paramount and some accuracy can be sacrificed

---

## 3. Feature Selection Strategy

### Selection Pipeline

```
Raw Features (~50)
        ↓
[1] Leakage Detection → Remove post-outcome features (e.g., actual_shipping_days)
        ↓
[2] Production Availability → Remove offline-only features
        ↓
[3] ID Columns → Remove identifiers (order_id, customer_id)
        ↓
[4] Variance Filter → Remove near-constant features
        ↓
[5] Correlation Filter → Remove highly correlated pairs (>0.95)
        ↓
[6] Importance Ranking → Compute SHAP/model importance
        ↓
Selected Features (~25-30)
```

### Key Excluded Features

| Feature | Reason | Category |
|---------|--------|----------|
| `late_delivery_risk` | Target variable | Target |
| `delivery_status` | Categorical form of target | Leakage |
| `days_for_shipping_(real)` | Only known after delivery | Leakage |
| `order_id`, `customer_id` | Identifiers, not predictive | ID |

### Top Predictive Features

| Rank | Feature | Importance | Business Meaning |
|------|---------|------------|------------------|
| 1 | `scheduled_shipping_days` | High | Expected delivery timeframe |
| 2 | `shipping_mode_encoded` | High | Carrier/shipping method |
| 3 | `order_item_quantity` | Medium | Items per order |
| 4 | `product_price` | Medium | Order value |
| 5 | `order_profit_per_order` | Medium | Profit margin |

---

## 4. Hyperparameter Tuning

### Optuna Configuration

```python
N_TRIALS = 50
CV_FOLDS = 3
SAMPLER = TPESampler(seed=42)  # Bayesian optimization
OBJECTIVE = 'f1_weighted'
```

### LightGBM Search Space

| Parameter | Range | Rationale |
|-----------|-------|-----------|
| `n_estimators` | [100, 500] | Balance fit vs overfit |
| `max_depth` | [4, 12] | Tree complexity |
| `learning_rate` | [0.01, 0.3] | Log scale for fine control |
| `num_leaves` | [20, 100] | Leaf-wise growth control |
| `subsample` | [0.6, 1.0] | Regularization |
| `colsample_bytree` | [0.6, 1.0] | Feature subsampling |
| `reg_alpha/lambda` | [1e-8, 10] | L1/L2 regularization |

### Typical Best Parameters

```python
best_params = {
    'n_estimators': 250,
    'max_depth': 8,
    'learning_rate': 0.05,
    'num_leaves': 50,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

---

## 5. Threshold Optimization

### Why Optimize Threshold?

Default 0.5 threshold may not be optimal when:
- Classes are imbalanced (54.8% late vs 45.2% on-time)
- Business costs differ for false positives vs false negatives
- Operational constraints require specific precision/recall tradeoff

### Optimization Results

| Strategy | Optimal Threshold | F1 Score |
|----------|-------------------|----------|
| Default | 0.50 | ~0.698 |
| F1 Optimized | ~0.48 | ~0.702 |
| Business Cost | Varies | Depends on cost matrix |

### Business Cost Framework

```
Cost(FN) = $75  # Missed late delivery
Cost(FP) = $15  # Unnecessary intervention
Revenue(TP) = $50  # Saved by catching late

Optimal threshold minimizes: FN*$75 + FP*$15 - TP*$50
```

---

## 6. Ensembling Strategy

### Ensemble Architecture

```
                    ┌─ LightGBM (Tuned)
                    │
Voting/Stacking ────┼─ XGBoost (Tuned)
                    │
                    └─ CatBoost (Default)
                              │
                              ↓
                      Logistic Regression
                        (Meta-learner)
```

### Ensemble Methods

| Method | Implementation | Leakage Prevention |
|--------|---------------|-------------------|
| **Soft Voting** | Average probabilities | N/A - simple averaging |
| **CV-Stacking** | 5-fold CV for meta-features | Cross-validation prevents leakage |

### When Ensembles Help

- Model diversity: Different algorithms capture different patterns
- Reduces variance: Averaging smooths predictions
- Robustness: Less sensitive to hyperparameter choices

### When Single Model Preferred

- Inference speed constraints
- Interpretability requirements
- Marginal improvement not worth complexity

---

## 7. Model Selection Rationale

### Final Model: LightGBM (Tuned) with Optimized Threshold

**Selected because:**

1. **Performance**: Highest F1 score among single models
2. **Speed**: Fastest training and inference times
3. **Stability**: Consistent performance across CV folds
4. **Interpretability**: Tree-based, SHAP-compatible
5. **Production-ready**: Sklearn interface, easy deployment

**Not selected: Stacking Ensemble**

- Marginal improvement (~0.5% F1) not worth complexity
- 3x inference time
- Harder to maintain and debug

---

## 8. Key Insights for Stakeholders

### What Drives Late Deliveries

Based on SHAP analysis:

1. **Scheduled Shipping Days**: Longer windows = higher late risk
   - *Action*: Tighten SLAs for standard shipping

2. **Shipping Mode**: Standard Class has highest late rate
   - *Action*: Consider default First Class for high-value customers

3. **Order Complexity**: Multi-item orders have higher risk
   - *Action*: Priority handling for complex orders

4. **Geographic Factors**: Certain regions consistently delayed
   - *Action*: Regional carrier performance review

### Business Recommendations

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| High | Deploy prediction scoring | Enable proactive intervention |
| High | Risk-tier based workflow | Target resources effectively |
| Medium | Carrier SLA negotiation | Address root causes |
| Medium | Customer communication | Manage expectations |
| Low | Threshold tuning | Optimize cost/benefit |

---

## 9. Production Deployment Checklist

### Pre-Deployment

- [ ] Validate model on recent holdout data
- [ ] Review feature selection report with domain experts
- [ ] Confirm all features available in production
- [ ] Set up monitoring dashboards

### Deployment

- [ ] Deploy model as REST API or batch scoring
- [ ] Implement logging for all predictions
- [ ] Set up A/B test framework
- [ ] Configure alerting for model drift

### Post-Deployment

- [ ] Monitor prediction distribution daily
- [ ] Track business metrics (late rate, intervention rate)
- [ ] Schedule monthly model retraining
- [ ] Quarterly feature engineering review

---

## 10. Files and Artifacts

### Key Files

| File | Purpose |
|------|---------|
| `MODEL_OVERVIEW.md` | Problem definition and approach |
| `MODEL_SUMMARY.md` | This document - final strategy |
| `feature_description.md` | Feature definitions and business context |
| `notebooks/04_model_training.ipynb` | Full training pipeline |
| `notebooks/05_business_impact.ipynb` | SHAP analysis and business insights |
| `src/features/feature_selector.py` | Automated feature selection |
| `src/models/classifier_optimized.py` | Model training utilities |

### Model Artifacts

| Artifact | Location | Description |
|----------|----------|-------------|
| `best_model_*.pkl` | `models/` | Trained model binary |
| `feature_selector_*.pkl` | `models/` | Feature selector state |
| `training_results_*.pkl` | `models/` | Metrics and comparison |
| `feature_selection_report_*.csv` | `models/` | Per-feature decisions |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2025-12-05 | Complete pipeline refactoring with LightGBM focus |
| 1.0 | Previous | Initial implementation |

---

*Generated: 2025-12-05*
*Pipeline Version: 2.0*
