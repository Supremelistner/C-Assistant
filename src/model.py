"""Dropout prediction — Step 3: train and cross-validate the models.

Run from the project root:   python src/model.py      (about 10 seconds; seeded, so results repeat exactly)

Every experiment is scored with 5-fold stratified cross-validation, repeated 5 times (25 test folds).
Metrics are computed in src/evaluate.ipynb from the saved predictions, so nothing here needs re-running
to try a new threshold or metric.

Outputs (data/models/):
    oof_predictions.csv   one row per (experiment, repeat, student): true label + predicted probability
    logreg_early.joblib   final logistic regression trained on all 500 students
    logreg_coefficients.csv
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ROOT / "data" / "features"
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "data" / "models"

N_SPLITS = 5
N_REPEATS = 5          # 5 x 5-fold = 25 test folds, so we can report mean ± std
SEED = 42

# Chosen from features.ipynb. Near-copies are left out so logistic-regression
# coefficients stay readable: first_score / best_score (r≈0.9 with mean_score)
# and retakes (r≈0.9 with n_attempts).
NUMERIC = [
    "mean_score",
    "mean_time",
    "days_to_first_attempt",
    "n_lessons",
    "n_attempts",
    "reached_L2",
    "days_active",
    "days_since_last_attempt",
    "score_trend",
    "no_attempts_in_window",
]
CATEGORICAL = ["device", "age_group", "country"]
# Deployed model drops country: it adds no accuracy (evaluate.ipynb, section 9)
# and is a fairness risk as a proxy for region / income.
FINAL_CATEGORICAL = ["device", "age_group"]
TARGET = "dropped"


# ---------------------------------------------------------------- data

def load_early():
    """Honest feature set: what we know 14 days after signup."""
    return pd.read_csv(FEATURES / "features_early.csv")


def load_leaky():
    """Whole-course features + refund_requested. Comparison only: shows how
    good a model *looks* when it is allowed to peek at the outcome."""
    full = pd.read_csv(FEATURES / "features_full.csv")
    refund = pd.read_csv(CLEAN / "students_clean.csv", usecols=["student_id", "refund_requested"])
    full = full.merge(refund, on="student_id")
    full["refund_requested"] = (full["refund_requested"] == "yes").astype(int)
    return full


# ---------------------------------------------------------------- models

def preprocessor(numeric, categorical):
    # Imputing inside the pipeline means each CV fold learns medians from its
    # own training rows only — no information leaks from the test fold.
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([("num", num, numeric), ("cat", cat, categorical)])


def logreg(numeric=NUMERIC, categorical=CATEGORICAL):
    # class_weight="balanced": dropouts are only 25% of students, so without it
    # the model is happy to predict "stays" for almost everyone.
    return Pipeline([
        ("prep", preprocessor(numeric, categorical)),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def gboost(numeric=NUMERIC, categorical=CATEGORICAL):
    # Small, shallow trees: 500 rows is not much data.
    return Pipeline([
        ("prep", preprocessor(numeric, categorical)),
        ("model", GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05,
                                             subsample=0.8, random_state=SEED)),
    ])


def majority(numeric=NUMERIC, categorical=CATEGORICAL):
    """Always predicts the dropout base rate. The floor every model must beat."""
    return DummyClassifier(strategy="prior")


def one_rule(numeric=NUMERIC, categorical=CATEGORICAL):
    """Baseline: one cut-off on mean_score ("flag if score < X"), learned per fold. A decision stump."""
    return Pipeline([
        ("score_only", ColumnTransformer([("score", SimpleImputer(strategy="median"), ["mean_score"])])),
        ("stump", DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=SEED)),
    ])


# ---------------------------------------------------------------- experiments
# name -> (data loader, model factory, numeric features, categorical features)

EXPERIMENTS = {
    "majority":          (load_early, majority, NUMERIC, CATEGORICAL),
    "one_rule":          (load_early, one_rule, NUMERIC, CATEGORICAL),
    "logreg":            (load_early, logreg, NUMERIC, CATEGORICAL),
    "gboost":            (load_early, gboost, NUMERIC, CATEGORICAL),
    # Fairness check: does the model actually need country?
    "logreg_no_country": (load_early, logreg, NUMERIC, ["device", "age_group"]),
    # Leak demo: whole-course activity + refund_requested
    "logreg_leaky":      (load_leaky, logreg, NUMERIC + ["refund_requested"], CATEGORICAL),
}


def cross_val_probs(make_model, df, numeric, categorical, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    """Out-of-fold dropout probabilities for every student, repeated n_repeats times.

    Each repeat reshuffles the folds, so we get n_repeats x n_splits test folds
    to measure how much the metrics move around.
    """
    X = df[numeric + categorical]
    y = df[TARGET].to_numpy()
    rows = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for fold, (train, test) in enumerate(skf.split(X, y)):
            model = make_model(numeric, categorical).fit(X.iloc[train], y[train])
            proba = model.predict_proba(X.iloc[test])[:, 1]
            rows.append(pd.DataFrame({
                "repeat": r, "fold": fold,
                "student_id": df["student_id"].iloc[test].to_numpy(),
                "y_true": y[test], "proba": proba,
            }))
    return pd.concat(rows, ignore_index=True)


def run_all(experiments=EXPERIMENTS):
    results = []
    for name, (load, make_model, numeric, categorical) in experiments.items():
        print(f"  {name} ...")
        oof = cross_val_probs(make_model, load(), numeric, categorical)
        oof.insert(0, "experiment", name)
        results.append(oof)
    return pd.concat(results, ignore_index=True)


def fit_final(df=None):
    """Logistic regression on all students — the model we'd actually deploy."""
    df = load_early() if df is None else df
    model = logreg(NUMERIC, FINAL_CATEGORICAL).fit(df[NUMERIC + FINAL_CATEGORICAL], df[TARGET])
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = pd.DataFrame({
        "feature": [n.split("__", 1)[1] for n in names],
        "coefficient": model.named_steps["model"].coef_[0],
    })
    coefs["odds_ratio"] = np.exp(coefs["coefficient"])
    return model, coefs.sort_values("coefficient", key=abs, ascending=False)


def main():
    OUT.mkdir(exist_ok=True)
    print(f"Cross-validating ({N_REPEATS} x {N_SPLITS}-fold):")
    oof = run_all()
    oof.to_csv(OUT / "oof_predictions.csv", index=False)

    model, coefs = fit_final()
    joblib.dump(model, OUT / "logreg_early.joblib")
    coefs.to_csv(OUT / "logreg_coefficients.csv", index=False)

    print(f"Saved predictions, final model and coefficients to {OUT}")
    print("Now open src/evaluate.ipynb for the metrics.")


if __name__ == "__main__":
    main()
