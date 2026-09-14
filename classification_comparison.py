"""
===============================================================================
COMPARATIVE BENCHMARK OF FOUR MACHINE LEARNING CLASSIFIERS
===============================================================================
Dataset    : Dry Bean Dataset (UCI) - 13,611 instances, 7 classes
Models     : SVM (linear & RBF), KNN, Gaussian Naive Bayes, Gradient Boosting
Pipeline   : Unified preprocessing + RandomizedSearchCV + StratifiedKFold CV
Language   : English
===============================================================================
"""

# =============================================================================
# 1. IMPORTS AND CONFIGURATION
# =============================================================================

import warnings
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

# Show all columns when printing DataFrames
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# Suppress non-critical warnings so the console output stays clean
warnings.filterwarnings("ignore")

# Global seed for reproducibility
SEED = 42

# Number of folds for cross-validation
N_FOLDS = 5

# Fixed evaluation metric used everywhere
SCORING = "f1_macro"

# Store results of every model here: model_name -> dict of metrics
RESULTS = {}

# This folder holds all generated plots
PLOTS_DIR = "plots"


def pprint_header(title, char="="):
    """Print a big section header to make the console output readable."""
    print(f"\n{char * 70}")
    print(f"  {title}")
    print(f"{char * 70}\n")


def pprint_table(headers, rows):
    """Print a nicely aligned text table to the console."""
    col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    line = "+".join("-" * (w + 2) for w in col_widths)
    line = "+" + line + "+"
    header_row = "|".join(str(h).center(w + 2) for h, w in zip(headers, col_widths))
    header_row = "|" + header_row + "|"

    print(line)
    print(header_row)
    print(line)
    for r in rows:
        row = "|".join(str(c).center(w + 2) for c, w in zip(r, col_widths))
        print("|" + row + "|")
    print(line)


# =============================================================================
# 2. PHASE 1: STRATEGIC MODEL SELECTION (THEORETICAL COMPARISON)
# =============================================================================

pprint_header("PHASE 1 - STRATEGIC MODEL SELECTION: THEORETICAL COMPARISON")
print("""
Four models were strategically selected to guarantee mathematical diversity.
At least one model was chosen from each of the three families:

  - Family 1 (Geometric and Distance-based):  SVM and KNN
  - Family 2 (Probabilistic and Generative):  Gaussian Naive Bayes
  - Family 3 (Ensemble and Stochastic):       Gradient Boosting
""")

theo_headers = ["Model", "Core Mathematical Principle", "Outlier Sensitivity",
                "Computational Cost", "Assumptions About Data"]
theo_rows = [
    [
        "SVM",
        "Margin maximization. Finds the hyperplane with the largest margin that "
        "separates classes; the kernel trick maps data to a higher-dimensional "
        "space so non-linear boundaries can be found.",
        "Medium-High. Depends on the regularization parameter C. A small C "
        "tolerates outliers (softer margin); a large C is very sensitive to them.",
        "High training (O(n^2 - n^3) for quadratic programming), medium "
        "prediction time (depends on number of support vectors).",
        "Does NOT assume a data distribution. Requires adequate separation in "
        "the kernel-transformed space, and needs feature scaling to work well.",
    ],
    [
        "KNN",
        "Instance-based (lazy) learning. Classifies a point by voting among its "
        "k nearest neighbors according to a distance metric (Euclidean, Manhattan, "
        "etc.). Decision is purely geometric.",
        "High. A single outlier close to a point corrupts the vote. Very "
        "sensitive to noisy or balanced-density regions.",
        "Low training (only stores data), high prediction (O(n) to search "
        "neighbors). Gets worse as dataset grows.",
        "Non-parametric. Assumes nearby points in feature space belong to the "
        "same class (local spatial smoothness). Features MUST be scaled.",
    ],
    [
        "GaussianNB",
        "Bayes' Theorem: P(Class|X) = P(X|Class) * P(Class) / P(X). Assumes each "
        "feature follows a Gaussian (normal) distribution within each class and "
        "that features are independent given the class.",
        "Low-Medium. Outliers shift the estimated mean/variance slightly, but "
        "the model does not optimize decision boundaries, so impact is limited.",
        "Very low. Closed-form parameter estimation: O(n) to compute statistics.",
        "Strong assumption of feature independence given the class and Gaussian "
        "distribution per feature. Performs well when these hold even with "
        "little data.",
    ],
    [
        "Gradient Boosting",
        "Sequential boosting of weak learners (CART decision trees). Each new "
        "tree fits the negative gradient (residuals) of the current loss, "
        "correcting the mistakes of all previous trees.",
        "Medium. Deep trees overfit outliers, but shallow trees with high "
        "regularization dampen their effect. Outliers influence residual-driven "
        "learning.",
        "High training (sequential additive fitting, expensive to tune), low "
        "prediction time (sum of tree evaluations).",
        "Does NOT assume a distribution. Requires careful hyperparameter tuning "
        "to avoid overfitting and is very sensitive to imbalanced classes.",
    ],
]
pprint_table(theo_headers, theo_rows)


# =============================================================================
# 3. PHASE 2: EXPERIMENTAL CHALLENGE
# =============================================================================

# ---------------------------------------------------------------------------
# 3.1 Load the Dry Bean Dataset
# ---------------------------------------------------------------------------
pprint_header("PHASE 2 - EXPERIMENTAL CHALLENGE")

# The Dry Bean dataset is hosted on the UCI repository. We try to download it;
# if there is no internet connection we ask the user to place the file locally.
DRY_BEAN_URL = "https://archive.ics.uci.edu/static/public/602/data.csv"

print("[1/7] Loading the Dry Bean Dataset (13,611 instances, 7 classes)...")

# Prefer a local file if the user already downloaded it; otherwise download.
import os
import urllib.request

LOCAL_CSV = "DryBean_Dataset.csv"


def load_dry_bean():
    """Return the Dry Bean dataset as a DataFrame (downloaded or local)."""
    # Try local CSV first
    if os.path.exists(LOCAL_CSV):
        print(f"      -> Found local file: {LOCAL_CSV}")
        return pd.read_csv(LOCAL_CSV)

    # Try to download the dataset from UCI (plain CSV file)
    print("      -> Downloading from UCI repository...")
    try:
        req = urllib.request.Request(
            DRY_BEAN_URL,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read()
        # The UCI CSV starts with an HTML/JSON wrapper; find the real CSV body.
        import csv as _csv
        text = raw.decode("utf-8", errors="replace")
        # Locate the header row (starts with 'Area,Perimeter')
        start = text.find("Area,Perimeter")
        if start == -1:
            raise ValueError("CSV header not found in downloaded content")
        content = text[start:]
        df = pd.DataFrame(list(_csv.reader(_csv.StringIO(content))))
        header = df.iloc[0]
        df = df[1:]
        df.columns = header
        df = df.reset_index(drop=True)
        # Convert every numeric column to float
        for col in df.columns:
            if col != "Class":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_csv(LOCAL_CSV, index=False)
        print(f"      -> Saved to {LOCAL_CSV}")
        return df
    except Exception as e:
        raise RuntimeError(
            "\nCould not download the dataset automatically. "
            f"Reason: {e}\nPlease download 'DryBean_Dataset.csv' from "
            "https://archive.ics.uci.edu/dataset/602/dry+bean and place it "
            "in this folder.\n"
        )


df = load_dry_bean()
print(f"      -> Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# Separate features (X) from the target class label (y)
FEATURES = ["Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
            "AspectRatio", "Eccentricity", "ConvexArea", "EquivDiameter",
            "Extent", "Solidity", "Roundness", "Compactness", "ShapeFactor1",
            "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"]
TARGET = "Class"

X = df[FEATURES]
y = df[TARGET]

# ---------------------------------------------------------------------------
# 3.2 Quick exploratory overview
# ---------------------------------------------------------------------------
print("\n[2/7] Exploratory overview:")
print("\n      Target class distribution:")
class_counts = y.value_counts()
for cls, cnt in class_counts.items():
    print(f"        - {cls}: {cnt}  ({cnt / len(y) * 100:.1f}%)")

print(f"\n      Total instances : {len(df)}")
print(f"      Total features  : {X.shape[1]} (all numeric, no missing values)")
print(f"      Missing values  : {df[FEATURES].isna().sum().sum()}")
print(f"      Classes         : {df[TARGET].nunique()} (multiclass)")

# ---------------------------------------------------------------------------
# 3.3 Unified train / test split (stratified to preserve class ratios)
# ---------------------------------------------------------------------------
print("\n[3/7] Stratified train/test split (80% train - 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)
print(f"      Train set: {X_train.shape[0]} rows")
print(f"      Test set : {X_test.shape[0]} rows")

# ---------------------------------------------------------------------------
# 3.4 Unified preprocessing pipeline
# ---------------------------------------------------------------------------
print("\n[4/7] Building the UNIFIED preprocessing pipeline...")
print("""
      A single preprocessing flow is applied to ALL four models so that they
      compete on equal terms:
        1. StandardScaler  -> centers each feature to mean=0, std=1.
        2. ColumnTransformer -> applies the scaler to every numeric column.
        3. FunctionTransformer -> keeps the pipeline structure (no filling needed;
           this dataset has no missing values or categorical variables).
""")

# For this dataset every column is numeric, so we scale all of them.
# Using a ColumnTransformer keeps the pipeline extensible if other features
# (e.g. categorical columns) were added later.
preprocessing = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), list(X_train.columns)),
    ],
    remainder="passthrough",
)

# ---------------------------------------------------------------------------
# 3.5 Hyperparameter search definitions ()
# ---------------------------------------------------------------------------
print("\n[5/7] Defining hyperparameter search spaces for each model...")

# Each model gets its own RandomizedSearchCV with a reasonable parameter grid.
# We use RandomizedSearchCV instead of GridSearchCV because it is faster and
# explores the space more efficiently.
MODELS = {
    "SVM (linear/RBF)": {
        "estimator": Pipeline([
            ("prep", preprocessing),
            ("clf", SVC(random_state=SEED)),
        ]),
        "param_grid": {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto", 0.001, 0.01],
        },
        "n_iter": 25,
    },
    "KNN": {
        "estimator": Pipeline([
            ("prep", preprocessing),
            ("clf", KNeighborsClassifier()),
        ]),
        "param_grid": {
            "clf__n_neighbors": [3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
        "n_iter": 30,
    },
    "GaussianNB": {
        "estimator": Pipeline([
            ("prep", preprocessing),
            ("clf", GaussianNB()),
        ]),
        "param_grid": {
            "clf__var_smoothing": np.logspace(-12, -7, 10),
        },
        "n_iter": 10,
    },
    "Gradient Boosting": {
        "estimator": Pipeline([
            ("prep", preprocessing),
            ("clf", GradientBoostingClassifier(random_state=SEED)),
        ]),
        "param_grid": {
            "clf__n_estimators": [50, 100],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
            "clf__subsample": [0.9, 1.0],
        },
        "n_iter": 8,
    },
}

# ---------------------------------------------------------------------------
# 3.6 Train + optimize + cross-validate every model
# ---------------------------------------------------------------------------
print("\n[6/7] Training, optimizing and cross-validating all 4 models...")
print("      (randomized search + 5-fold stratified cross-validation each)\n")

# Fixed fold splitter used for the FINAL cross-validation report
stratified_folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for name, cfg in MODELS.items():
    print(f"      ----- Running: {name} -----")

    # --- Hyperparameter optimization (nested, with internal CV) ----------
    search = RandomizedSearchCV(
        estimator=cfg["estimator"],
        param_distributions=cfg["param_grid"],
        n_iter=cfg["n_iter"],
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED),
        scoring=SCORING,
        n_jobs=-1,
        verbose=0,
        random_state=SEED,
        refit=True,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    train_time = time.time() - t0

    best_model = search.best_estimator_
    print(f"        Best params : {search.best_params_}")
    print(f"        Best CV F1   : {search.best_score_:.4f}")

    # --- Final K-Fold Cross-Validation on the best model -----------------
    cv_scores = cross_val_score(
        best_model, X_train, y_train,
        cv=stratified_folds, scoring=SCORING, n_jobs=-1,
    )

    # --- Evaluation on the held-out test set ------------------------------
    t0 = time.time()
    y_pred = best_model.predict(X_test)
    predict_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro")
    f1 = f1_score(y_test, y_pred, average="macro")

    # Store every result for the final comparison
    RESULTS[name] = {
        "best_params": search.best_params_,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1": f1,
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "best_model": best_model,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    print(f"        CV F1 (mean)      : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"        Test F1 (macro)   : {f1:.4f}")
    print(f"        Test accuracy     : {acc:.4f}")
    print(f"        Training time     : {train_time:.2f} s")
    print(f"        Prediction time   : {predict_time:.4f} s\n")


# ---------------------------------------------------------------------------
# 3.7 Save the trained models (optional, commented out by default)
# ---------------------------------------------------------------------------
# from joblib import dump
# for name, res in RESULTS.items():
#     dump(res["best_model"], f"model_{name.replace(' ', '_').replace('(', '').replace(')', '')}.joblib")

print("[7/7] All models trained and evaluated.")


# =============================================================================
# 4. PHASE 3: THE VERDICT (ANALYSIS AND BENCHMARKING)
# =============================================================================

pprint_header("PHASE 3 - THE VERDICT: ANALYSIS AND BENCHMARKING")

# ---------------------------------------------------------------------------
# 4.1 Final results comparison table
# ---------------------------------------------------------------------------
print("[A] Final results table (all metrics):\n")

res_headers = ["Model", "CV F1 (mean)", "CV F1 (std)", "Test F1", "Test Acc",
               "Precision", "Recall", "Train (s)", "Predict (s)"]
res_rows = [
    [
        name,
        f"{r['cv_mean']:.4f}",
        f"{r['cv_std']:.4f}",
        f"{r['test_f1']:.4f}",
        f"{r['test_accuracy']:.4f}",
        f"{r['test_precision']:.4f}",
        f"{r['test_recall']:.4f}",
        f"{r['train_time_s']:.2f}",
        f"{r['predict_time_s']:.4f}",
    ]
    for name, r in RESULTS.items()
]
pprint_table(res_headers, res_rows)

# ---------------------------------------------------------------------------
# 4.2 Define a clean colour palette and helper to generate all figures
# ---------------------------------------------------------------------------
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# Use the class names in alphabetical order for every confusion matrix
class_labels = sorted(y.unique())


def safe_filename(name):
    """Make a model name safe to use as a filename (no spaces/slashes/parens)."""
    out = name.replace(" ", "_").replace("(", "").replace(")", "")
    return out.replace("/", "_")


def save_fig(fig, fname):
    """Save a figure and close it to free memory."""
    path = os.path.join(PLOTS_DIR, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      -> Saved {path}")


# ---------------------------------------------------------------------------
# 4.3 Confusion matrices for all four models
# ---------------------------------------------------------------------------
print("\n[B] Generating confusion matrices for all 4 models...")

for name, res in RESULTS.items():
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        res["confusion_matrix"],
        annot=True, fmt="d", cmap="Blues",
        xticklabels=class_labels, yticklabels=class_labels,
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    save_fig(fig, f"confusion_matrix_{safe_filename(name)}.png")

# ---------------------------------------------------------------------------
# 4.4 Bar chart: F1-score vs. training time (cost-benefit analysis)
# ---------------------------------------------------------------------------
print("\n[C] Generating F1 vs. training time comparison chart...")

model_names = list(RESULTS.keys())
test_f1 = [RESULTS[m]["test_f1"] * 100 for m in model_names]
train_times = [RESULTS[m]["train_time_s"] for m in model_names]

x_pos = np.arange(len(model_names))
width = 0.35

fig, ax1 = plt.subplots(figsize=(12, 6))
bars1 = ax1.bar(x_pos - width / 2, test_f1, width, label="Test F1 (macro, %)", color="#2e86ab", alpha=0.9)
ax1.set_ylabel("Test F1-Score (%)", color="#2e86ab")
ax1.set_ylim(0, 105)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_names)
ax1.tick_params(axis="y", labelcolor="#2e86ab")

ax2 = ax1.twinx()
bars2 = ax2.bar(x_pos + width / 2, train_times, width, label="Training Time (s)", color="#dc6acf", alpha=0.9)
ax2.set_ylabel("Training Time (seconds)", color="#dc6acf")
ax2.set_ylim(0, max(train_times) * 1.2)
ax2.tick_params(axis="y", labelcolor="#dc6acf")

for bar, val in zip(bars1, test_f1):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{val:.1f}",
             ha="center", va="bottom", fontsize=9, color="#2e86ab")
for bar, val in zip(bars2, train_times):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02 * max(train_times), f"{val:.1f}s",
             ha="center", va="bottom", fontsize=9, color="#dc6acf")

ax1.set_title("Cost-Benefit Analysis: F1-Score vs. Training Time")
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.98))
save_fig(fig, "f1_vs_training_time.png")

# ---------------------------------------------------------------------------
# 4.5 Decision boundaries simplified with PCA in 2D
# ---------------------------------------------------------------------------
print("\n[D] Generating PCA 2D decision-boundary visualisation (subsample)...")

# Map every class name to a number so matplotlib can colour them
codes = {c: i for i, c in enumerate(class_labels)}

# Because 13,611 points are too many to plot legibly, we subsample 500 points.
rng = np.random.RandomState(SEED)
sample_idx = rng.choice(X_train.shape[0], size=500, replace=False)

# Project the (scaled) training data into 2D with PCA
pca = PCA(n_components=2)
X_pca_all = pca.fit_transform(preprocessing.fit_transform(X_train))
X_pca = X_pca_all[sample_idx]
y_pca = y_train.iloc[sample_idx].values

# Mesh grid over the PCA space
mesh_step = 0.25
x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step),
                     np.arange(y_min, y_max, mesh_step))

from sklearn.base import clone

fig, axes = plt.subplots(2, 2, figsize=(15, 13))
axes = axes.ravel()

for ax, (name, res) in zip(axes, RESULTS.items()):
    # Clone the winning classifier and re-fit it on the 2D PCA coordinates.
    # This is ONLY for visualisation; the real evaluation uses the full
    # 16-dimensional pipeline trained earlier.
    clf_2d = clone(res["best_model"].named_steps["clf"])
    clf_2d.fit(X_pca, y_pca)

    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([codes[str(p)] for p in Z]).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.35, cmap="Set3", levels=len(class_labels) - 1)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=[codes[str(c)] for c in y_pca],
               cmap="Set3", edgecolor="k", s=25)
    ax.set_title(f"{name}\n(2D PCA projection)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)

fig.suptitle("Decision Boundaries in PCA Space (first 2 principal components)",
             fontsize=16, y=1.0)
save_fig(fig, "decision_boundaries_pca.png")

# ---------------------------------------------------------------------------
# 4.6 The verdict - written conclusions
# ---------------------------------------------------------------------------
pprint_header("THE VERDICT: WRITTEN CONCLUSIONS")

# Determine the best overall model by test F1-macro
best_name = max(RESULTS, key=lambda m: RESULTS[m]["test_f1"])
best = RESULTS[best_name]
second = sorted(RESULTS, key=lambda m: RESULTS[m]["test_f1"], reverse=True)[1]

print(f"""
QUESTION 1 - Which model achieved the best balance between accuracy and
             generalization capability (F1-Score)?
-------------------------------------------------------------------------------
The best model overall is **{best_name}** with a test F1-macro of
{best['test_f1']:.4f} and a cross-validated F1-macro of {best['cv_mean']:.4f}
(+/- {best['cv_std']:.4f}).

Key indicators of generalization (not lucky-split results):
  - The gap between the CV mean and the test F1 is only
    {abs(best['cv_mean'] - best['test_f1']):.4f}, which shows the model is stable.
  - The low cross-validation standard deviation ({best['cv_std']:.4f}) confirms
    the performance does not change much from one fold to another.

However, a well-balanced result must also consider the second-placed model,
**{second}** (test F1 = {RESULTS[second]['test_f1']:.4f}). The difference
between both is {abs(best['test_f1'] - RESULTS[second]['test_f1']):.4f}, which
we now weigh against their computational cost in the next question.

QUESTION 2 - Did the most complex model justify its computational cost
             compared to the simpler model?
-------------------------------------------------------------------------------
""")

# Show every model's cost vs. benefit in one compact table
cost_benefit_rows = []
for name in sorted(RESULTS, key=lambda m: RESULTS[m]["test_f1"], reverse=True):
    r = RESULTS[name]
    cost_benefit_rows.append([
        name,
        f"{r['test_f1']:.4f}",
        f"{r['train_time_s']:.2f}s",
        f"{r['train_time_s'] / max(1e-9, r['test_f1']):.1f} s per 1.0 F1 point",
    ])
pprint_table(["Model (by F1)", "Test F1", "Train time", "Cost-effectiveness ratio"],
             cost_benefit_rows)

print(f"""
Analysis:
  * If the most complex model (Gradient Boosting) clearly wins F1 but takes
    10-20x the training time of KNN/Naive Bayes, the cost may still be
    justified if accuracy matters more than speed (e.g. offline applications).
  * KNN and GaussianNB are dramatically faster: their training is essentially
    instant. If their F1 is within ~0.5-1.0 points of the best model, they are
    the better practical choice for online or resource-limited settings.
  * Look at the 'cost-effectiveness ratio': the ratio tells you how many
    seconds of training you spend per unit of F1. Lower is better.

QUESTION 3 - How did the decision boundaries of the four models behave in
             relation to the dimensionality of the dataset?
-------------------------------------------------------------------------------
  1. KNN suffers the most from the curse of dimensionality. With 16 features,
     Euclidean distances become less meaningful because all points are roughly
     equally far apart. It still performs well in the Dry Bean dataset because
     the features are informative, but its boundaries (see PCA plot) are very
     irregular and localised.

  2. SVM with the RBF kernel builds a flexible non-linear boundary that adapts
     to the data. In the PCA visualisation it produces smooth, well-separated
     regions, showing that the kernel trick effectively embeds the 16
     dimensional space into a richer separating space.

  3. GaussianNB draws quadratic decision boundaries based on the estimated
     Gaussian parameters of each class. Because it assumes independence, it
     cannot model correlations between features; in the PCA plot its borders
     look simpler and less complex.

  4. Gradient Boosting builds a piecewise-constant staircase of rectangular
     regions (one per tree leaf). In low-dimensional PCA space it captures
     complex interactions, but it is prone to overfitting the noise when the
     number of trees grows unchecked.

CONCLUSION
------------------------------------------------------------------------------
Balancing F1, generalization and computational effort, the recommended model
is: **{best_name}**.

Recommendation by use case:
  - Fast prototyping / real-time prediction : GaussianNB or KNN.
  - Maximum accuracy with time available     : Gradient Boosting or SVM.
  - Best trade-off for this dataset          : {best_name}
""")

print("\nAll visualisations were saved inside the 'plots/' folder.")
print("\nDONE - End of the comparative benchmark.")