"""Part 3 bonus — TF-IDF + Logistic Regression on the 60 labelled explanations.

Compares a small classical model with the LLM grader on accuracy, cost and speed.

Two ways to test it, because the data is templated (many answers are the same text with a
different hedge on the end, e.g. "... i think" / "... not sure tho"):

  leave-one-out       train on 59, test on 1. Optimistic: a near-copy of the test answer,
                      with the same label, is usually in the training set.
  grouped (honest)    near-copies share a group and are always held out together, so the
                      model must grade an answer it has never seen any version of.

Run from the project root:
    python src/baseline_text_clf.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline

import grader as g

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "llm"


def make_model():
    # Character n-grams cope with typos and Hinglish; lesson id is prepended so the
    # model can tell which rubric an answer belongs to.
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1),
        LogisticRegression(max_iter=2000, C=10, class_weight="balanced"),
    )


def main():
    e = g.load_explanations()
    lab = e[e["label"].notna()].reset_index(drop=True)
    X = lab["lesson_id"] + " " + lab["explanation_text"]
    y = lab["label"]
    groups = lab["explanation_text"].map(g.core_text).factorize()[0]   # near-copies share a group

    results = {}
    start = time.perf_counter()
    results["leave-one-out (optimistic)"] = cross_val_predict(make_model(), X, y, cv=LeaveOneOut())
    loo_s = time.perf_counter() - start
    results["grouped 5-fold (honest)"] = cross_val_predict(make_model(), X, y, cv=GroupKFold(n_splits=5), groups=groups)

    # Speed: train once on all 60, time a prediction per answer
    model = make_model().fit(X, y)
    t0 = time.perf_counter()
    for _ in range(200):
        model.predict(X.iloc[:1])
    per_pred_ms = (time.perf_counter() - t0) / 200 * 1000

    # LLM on the same 60 rows, for the comparison
    graded = pd.read_csv(OUT / "graded.csv")
    qwen = graded[(graded["model"] == "qwen") & (graded["version"] == "rag") & (graded["set"] == "labelled")]
    qwen = qwen.drop_duplicates("explanation_id").set_index("explanation_id").loc[lab["explanation_id"]]

    rows = []
    for name, pred in results.items():
        rows.append({"method": f"TF-IDF + LR, {name}", "accuracy": (pred == y).mean(),
                     "wrong→passed": int(((y == "wrong") & (pred != "wrong")).sum())})
    rows.append({"method": "LLM: Qwen3.8-27B rag", "accuracy": (qwen["verdict"].values == y.values).mean(),
                 "wrong→passed": int(((y.values == "wrong") & (qwen["verdict"].values != "wrong")).sum())})
    table = pd.DataFrame(rows).set_index("method")

    print(f"Labelled rows: {len(lab)} | distinct answers after removing hedges: {groups.max() + 1}")
    print(f"Label mix: {y.value_counts().to_dict()}\n")
    print(table.round(3).to_string())
    print(f"\nTF-IDF + LR speed: ~{per_pred_ms:.2f} ms per answer on a laptop CPU "
          f"(training all 60 LOO folds took {loo_s:.1f} s)")
    print("Grouped CV confusion (rows = true, cols = predicted):")
    print(pd.crosstab(y, results["grouped 5-fold (honest)"]).to_string())
    table.round(4).to_csv(OUT / "baseline_text_clf.csv")


if __name__ == "__main__":
    main()
