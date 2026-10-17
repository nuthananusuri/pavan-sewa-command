"""Standard-library data loading, so the server runs on a bare LinuxONE Python
(no numpy / pandas / scikit-learn needed at runtime)."""
import csv
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TEXT_COLS = {"incident_id", "reporter_name", "reporter_phone", "source"}


def _num(v):
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v else f
    except (TypeError, ValueError):
        return v


def load_incidents(path=None):
    path = path or os.path.join(ROOT, "data", "incidents.csv")
    with open(path, newline="") as f:
        return [{k: (v if k in TEXT_COLS else _num(v)) for k, v in row.items()} for row in csv.DictReader(f)]


def load_test_incidents():
    rows = load_incidents()
    p = os.path.join(ROOT, "model", "test_idx.json")
    if not os.path.exists(p):
        return rows
    with open(p) as f:
        idx = json.load(f)
    return [rows[i] for i in idx]
