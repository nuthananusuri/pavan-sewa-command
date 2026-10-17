"""
Train the PAVAN SEWA triage model.

* Splits incidents 70/15/15 (train/val/test), stratified by label.
* Compares three approaches on the held-out test set:
    1. Rule baseline  - what a control room does today with a checklist
    2. Logistic regression (multinomial) on engineered features  <- deployed
    3. Gradient boosting (benchmark for "how much accuracy do we give up
       for transparency?")
* Exports the deployed model as plain JSON (scaler + coefficients) so the
  inference engine is dependency-free and runs on any LinuxONE / IBM Z
  Python, and reports metrics to model/metrics.json.

Usage: python model/train.py --data data/incidents.csv
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from engine.features import FEATURE_NAMES, engineer  # noqa: E402

LABELS = ["STABLE", "URGENT", "CRITICAL"]


def rule_baseline(df):
    """Checklist triage: severe injury / medical need / deep water => critical."""
    out = np.zeros(len(df), dtype=int)
    urgent = (df.injured_count > 0) | (df.water_level_m > 1.2) | (df.vulnerable_count > 2)
    critical = (df.severe_injury == 1) | ((df.medical_need == 1) & (df.water_level_m > 1.0)) | (df.water_level_m > 2.2)
    out[urgent.values] = 1
    out[critical.values] = 2
    return out


def report(name, y, p):
    return {
        "model": name,
        "accuracy": round(accuracy_score(y, p), 4),
        "macro_f1": round(f1_score(y, p, average="macro"), 4),
        "critical_recall": round(recall_score(y, p, labels=[2], average="macro"), 4),
        "confusion_matrix": confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/incidents.csv")
    ap.add_argument("--out", default="model/triage_model.json")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    X = np.array([engineer(r) for r in df.to_dict("records")], dtype=float)
    y = df.label.values

    idx = np.arange(len(df))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=42)

    scaler = StandardScaler().fit(X[tr])
    Xs = scaler.transform(X)

    # pick C on validation set
    best = None
    for C in [0.03, 0.1, 0.3, 1.0, 3.0]:
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xs[tr], y[tr])
        f = f1_score(y[va], m.predict(Xs[va]), average="macro")
        if best is None or f > best[0]:
            best = (f, C, m)
    _, C, lr = best

    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0).fit(X[tr], y[tr])

    results = [
        report("rule_baseline_checklist", y[te], rule_baseline(df.iloc[te])),
        report("logistic_regression_deployed", y[te], lr.predict(Xs[te])),
        report("gradient_boosting_benchmark", y[te], gb.predict(X[te])),
    ]
    for r in results:
        print(f"{r['model']:32s} acc={r['accuracy']:.3f}  macroF1={r['macro_f1']:.3f}  critical_recall={r['critical_recall']:.3f}")
    print("\nDeployed model (test set):")
    print(classification_report(y[te], lr.predict(Xs[te]), target_names=LABELS, digits=3))

    model = {
        "name": "pavan-sewa-triage",
        "version": "1.0.0",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "type": "multinomial_logistic_regression",
        "labels": LABELS,
        "features": FEATURE_NAMES,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": lr.coef_.tolist(),
        "intercept": lr.intercept_.tolist(),
        "C": C,
        "train_rows": int(len(tr)),
    }
    with open(args.out, "w") as f:
        json.dump(model, f, indent=1)

    metrics = {"split": {"train": len(tr), "val": len(va), "test": len(te)}, "results": results}
    with open(os.path.join(os.path.dirname(args.out), "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)

    with open(os.path.join(os.path.dirname(args.out), "test_idx.json"), "w") as f:
        json.dump([int(i) for i in te], f)
    print(f"saved {args.out} (C={C})")


if __name__ == "__main__":
    main()
