#  flight-delay-baselines

**Classical ML baseline models for flight arrival delay prediction.**

This repository establishes the performance ceiling for non-graph-aware models on the flight delay prediction task, providing the reference point against which the Graph Neural Network approach (in `flight-delay-gnn`) is evaluated.

> *"A model that doesn't beat a naive historical average isn't a model — it's noise."*

---

## Why Baselines?

Before deploying a complex GNN architecture, we need to answer a fundamental question: **does the graph structure actually help?**

The workflow is:

```
Historical Mean  →  Random Forest  →  XGBoost  →  [ GNN benchmark ]
    (naïve)           (ensemble)      (SOTA tabular)    (this repo's goal)
```

Each step raises the performance bar. If our GNN from `flight-delay-gnn` cannot outperform XGBoost on RMSE and F1, it means the relational (graph) information — delay propagation paths, network topology — adds no predictive value beyond what's already in the tabular features. That would be a surprising finding worth investigating.

---

## Models

### 0. Historical Mean Baseline (`HistoricalMeanBaseline`)

The simplest possible model: for each route (Origin → Dest), predict the historical average arrival delay seen during training.

- **Prediction:** `ŷ(ATL→JFK) = mean(ArrDelay | Origin=ATL, Dest=JFK)`
- **Unknown routes:** falls back to the global training mean
- **Purpose:** sanity check floor — any real model must beat this

### 1. Random Forest (`RandomForestRegressor`)

Ensemble of 200 decorrelated decision trees trained on bootstrap samples with random feature subsampling (`max_features="sqrt"`).

Key properties relevant to flight data:
- **Robust to outliers** — extreme delays don't dominate splits
- **No feature scaling required** — tree splits are invariant to monotone transformations
- **Free OOB score** — out-of-bag samples provide a bias-corrected generalization estimate
- **Interpretable** — Gini-based feature importance reveals what drives predictions

### 2. XGBoost (`XGBRegressor`)

Gradient boosting where each tree minimizes the residuals of the previous ensemble. The de facto standard for tabular ML competitions.

Key configuration choices:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 0.05 | Small LR + more trees → better generalization |
| `max_depth` | 6 | Standard; deeper → overfitting on delay tails |
| `subsample` | 0.8 | Stochastic boosting reduces variance |
| `colsample_bytree` | 0.8 | Decorrelates trees like RF |
| `reg_alpha` | 0.1 | L1: pushes unimportant feature weights to zero |
| `early_stopping_rounds` | 30 | Stops when val RMSE stagnates → prevents overfitting |

---

## Evaluation Metrics

We evaluate both **regression** (exact minutes) and **classification** (delayed > 15 min) simultaneously:

### Regression

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | $\sqrt{\frac{1}{n}\sum(\hat{y}_i - y_i)^2}$ | Penalizes large errors more — sensitive to extreme delays |
| **MAE** | $\frac{1}{n}\sum\|\hat{y}_i - y_i\|$ | Average absolute error in minutes — intuitive |
| **R²** | $1 - \frac{\text{SS}_\text{res}}{\text{SS}_\text{tot}}$ | Fraction of variance explained; 1.0 = perfect, 0.0 = mean-only |
| **MBE** | $\frac{1}{n}\sum(\hat{y}_i - y_i)$ | Mean Bias Error — signed, detects systematic over/under-prediction |

### Classification (threshold: > 15 minutes, per FAA definition)

| Metric | Why it matters |
|--------|---------------|
| **F1-Score** | Harmonic mean of Precision & Recall — robust to class imbalance (~35% delayed) |
| **Precision** | Of all predicted delays, how many were real? (operational cost: false alarms) |
| **Recall** | Of all real delays, how many did we catch? (passenger impact: missed delays) |
| **AUC-ROC** | Threshold-independent discrimination — 0.5 = random, 1.0 = perfect |

> **️Warning on Accuracy:** With ~35% positive class rate, a model that always predicts "on time" achieves 65% accuracy. Never use accuracy as the primary metric here — use F1.

---

## Temporal Train/Val/Test Split

**Critical design decision:** data is split **chronologically**, not randomly.

```
Jan 2023 ──────────────── Feb 2023 ──────── Mar 2023
│◄────────── Train (70%) ──────────►│◄─Val─►│◄─Test─►│
```

A random shuffle would cause **data leakage**: the model sees future flight patterns during training (e.g., learning post-storm recovery patterns before the storm). This inflates validation metrics and makes the model useless in deployment.

---

## Project Structure

```
flight-delay-baselines/
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Simulate/load data, temporal splitting
│   ├── train_baselines.py   # HistoricalMean, RandomForest, XGBoost training
│   └── evaluate.py          # RMSE, MAE, R², F1, AUC-ROC metrics
│
├── models/                  # Saved .pkl model files
├── results/
│   ├── baseline_results.csv    # All metrics in one table
│   └── training_metadata.json  # Hyperparameters, feature importances
│
├── main.py                  # CLI entry point
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/<your-username>/flight-delay-baselines.git
cd flight-delay-baselines
pip install -r requirements.txt

# Run with synthetic data (no external files needed)
python main.py

# Run with real data from the ETL pipeline
python main.py --source parquet \
    --filepath ../flight-network-etl/data/processed/flights_featured.parquet

# Larger dataset
python main.py --n-samples 200000
```

---

## Sample Results

Typical results on synthetic data (80k samples, Q1 2023 range):

| Model | Split | RMSE ↓ | MAE ↓ | R² ↑ | F1 ↑ | AUC-ROC ↑ |
|-------|-------|--------|-------|------|------|-----------|
| HistoricalMean | Test | 28.4 | 18.7 | 0.31 | 0.61 | — |
| RandomForest | Test | 7.2 | 4.1 | 0.94 | 0.91 | 0.97 |
| **XGBoost** | **Test** | **6.8** | **3.8** | **0.95** | **0.92** | **0.98** |
| GNN (target) | Test | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

The GNN target column will be filled in once `flight-delay-gnn` training is complete.

> **Note:** These results are on simulated data with explicitly engineered correlations. Results on real BTS data will differ — typically RMSE increases and R² decreases due to unpredictable delay events (ground stops, crew issues, etc.) that don't appear in historical features.

---

## Feature Importance (XGBoost, top features)

```
DepDelay              ████████████████████████████████  0.621
rolling_avg_arr_delay ████████                          0.148
visibility_mean       █████                             0.089
Distance              ████                              0.071
hour_sin              ██                                0.031
flight_category_worst ██                                0.028
is_peak_hour          █                                 0.012
...
```

`DepDelay` dominates — departure delay is the strongest single predictor of arrival delay. This is expected: most delay recovery happens in the air (pilots flying slightly faster), but large departure delays almost always result in large arrival delays.

---

## Next Step

The baseline results from this repo are imported directly into `flight-delay-gnn/` to compute the **percentage improvement** of the GNN over the strongest tabular baseline (XGBoost).

---

## Requirements

- Python 3.10+
- scikit-learn ≥ 1.5
- xgboost ≥ 2.1
- numpy, pandas (see `requirements.txt`)
