"""
Impact evaluation: AI-priority dispatch vs first-come-first-served (FCFS).

Takes the held-out TEST incidents (never seen in training), spreads their
arrival over a surge window, scores them with the deployed model, and
replays both dispatch policies with the same fleet. Repeats over several
random arrival orders and reports the mean.

Usage: python scripts/evaluate_dispatch.py
"""
import json
import os
import random
import sys

from statistics import mean

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from engine.dataio import load_test_incidents  # noqa: E402
from engine.dispatch import simulate  # noqa: E402
from engine.triage import TriageModel  # noqa: E402

LABELS = ["STABLE", "URGENT", "CRITICAL"]


def main(n_runs=10, surge_hours=6.0, n_incidents=300):
    test = load_test_incidents()
    model = TriageModel()
    for r in test:
        p = model.predict(r)
        r["triage"], r["priority"] = p["triage"], p["priority"]
        r["true_label"] = LABELS[int(r["label"])]

    agg = {"ai": [], "fcfs": []}
    crit_counts = []
    for seed in range(n_runs):
        rng = random.Random(seed)
        surge = [dict(r) for r in rng.sample(test, n_incidents)]
        for r in surge:
            r["reported_at"] = rng.uniform(0, surge_hours * 60)
        crit_counts.append(sum(r["true_label"] == "CRITICAL" for r in surge))
        for order in ("ai", "fcfs"):
            agg[order].append(simulate([dict(r) for r in surge], order=order))

    summary = {}
    for order, runs in agg.items():
        summary[order] = {
            "critical_reached_in_time_pct": round(float(mean([x["reached_in_time"]["CRITICAL"]["pct"] for x in runs])), 1),
            "urgent_reached_in_time_pct": round(float(mean([x["reached_in_time"]["URGENT"]["pct"] for x in runs])), 1),
            "stable_reached_in_time_pct": round(float(mean([x["reached_in_time"]["STABLE"]["pct"] for x in runs])), 1),
            "critical_median_wait_min": round(float(mean([x["critical_median_wait_min"] for x in runs])), 1),
        }
    crit_total = float(mean(crit_counts))
    summary["setup"] = {"incidents_per_surge": n_incidents, "avg_critical_incidents": crit_total, "surge_hours": surge_hours,
                        "fleet": "9 medical drones + 9 rescue boats across 3 bases", "runs": n_runs}
    extra = (summary["ai"]["critical_reached_in_time_pct"] - summary["fcfs"]["critical_reached_in_time_pct"]) / 100 * crit_total
    summary["extra_critical_cases_reached_in_time"] = round(extra, 1)
    print(json.dumps(summary, indent=1))
    with open(os.path.join(ROOT, "model", "dispatch_eval.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
