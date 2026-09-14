# Comparative Benchmark of Four Machine Learning Classifiers

## Activity Context

Academic project (Actividad en Inglés) comparing four classification models on a
single robust dataset, following a 3-phase methodology:

1. **Phase 1 - Strategic Model Selection** (theoretical comparison table)
2. **Phase 2 - The Classification Challenge** (experimentation)
3. **Phase 3 - The Verdict** (analysis and benchmarking)

## Selected Models (one per family)

| Family | Model |
|--------|-------|
| Family 1 - Geometric/Distance | SVM (linear & RBF kernels) |
| Family 1 - Geometric/Distance | KNN (Euclidean & Manhattan) |
| Family 2 - Probabilistic/Generative | Gaussian Naive Bayes |
| Family 3 - Ensemble/Stochastic | Gradient Boosting |

## Dataset

- **Source:** UCI Machine Learning Repository — Dry Bean Dataset (id 602)
- **Instances:** 13,611 (> 2,500 required)
- **Features:** 16 numeric (all scaled)
- **Target:** 7 bean classes (SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON)
- **Task type:** Multiclass classification
- **Primary metric:** F1-Score (macro)
- **Local file:** `DryBean_Dataset.csv` (auto-loads if present, otherwise downloads from `https://archive.ics.uci.edu/static/public/602/data.csv`)

## Pipeline (equal conditions for all models)

1. Stratified train/test split (80/20)
2. `StandardScaler` + `ColumnTransformer` (same preprocessing for all 4 models)
3. `RandomizedSearchCV` per model (scoring = `f1_macro`, 3-fold inner CV)
4. Final evaluation with `StratifiedKFold` (5-fold) cross-validation
5. Metrics collected: F1-macro, accuracy, precision, recall, confusion matrix, training and prediction time

## Measured Results (from a full run)

| Model | Test F1 | CV F1 (mean ± std) | Test Acc | Train (s) | Predict (s) |
|-------------------------|---------|--------------------|----------|-----------|-------------|
| SVM (linear/RBF)        | 0.9361  | 0.9443 ± 0.0057    | 0.9243   | 59.3      | 2.04        |
| Gradient Boosting       | 0.9354  | 0.9366 ± 0.0058    | 0.9236   | 500.9     | 0.15        |
| KNN                     | 0.9315  | 0.9376 ± 0.0060    | 0.9185   | 12.5      | 4.20        |
| GaussianNB              | 0.9091  | 0.9065 ± 0.0033    | 0.8979   | 0.6       | 0.01        |

Note: the Gradient Boosting train time shown is its full randomized-search time;
its final single-model fit is far faster. Times vary by machine.

## Verdict (Phase 3 conclusions)

- **Best accuracy/generalization trade-off:** SVM achieved the highest test F1-macro
  (0.9361) with a stable CV (0.9443 ± 0.0057). Gradient Boosting was essentially tied
  (0.9354) but took far longer to train.
- **Does the complex model justify its cost?** Gradient Boosting does NOT justify its
  cost here: it nearly matches SVM in F1 but takes ~5-8x longer to train. KNN and
  GaussianNB train in seconds and remain excellent for prototyping.
- **Decision boundaries vs dimensionality:** KNN suffers most from the curse of
  dimensionality; SVM (RBF kernel) builds smooth non-linear boundaries in PCA space;
  GaussianNB draws simpler quadratic boundaries (independence assumption); Gradient
  Boosting builds piecewise-constant (rectangular) regions and can overfit with too
  many trees.

## Repository Contents

- `classification_comparison.py` — full runnable script (Python, English)
- `DryBean_Dataset.csv` — the dataset (13,611 rows)
- `plots/` — generated visualizations:
  - `confusion_matrix_*.png` (one per model)
  - `f1_vs_training_time.png` (cost-benefit chart)
  - `decision_boundaries_pca.png` (2D PCA decision boundaries)

## How to Run

Requirements:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

Run:

```bash
python classification_comparison.py
```

The script prints all phase tables/metrics to the console and saves every figure
to the `plots/` folder. Total runtime is roughly 10-20 minutes on a normal laptop.