import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PSC_DATA_KEY", "dGVzdHMtb25seS1rZXktbm90LWZvci1wcm9kLTAwMDE=")

from engine.dispatch import default_fleet, plan, simulate  # noqa: E402
from engine.privacy import Vault  # noqa: E402
from engine.triage import TriageModel  # noqa: E402

BASE = dict(incident_id="T-1", lat=10.81, lon=78.69, people_count=3, injured_count=0, severe_injury=0,
            vulnerable_count=0, medical_need=0, water_level_m=0.2, water_rise_m_per_h=0.0, structure_risk=0.1,
            hours_since_report=0.5, has_food_water=1, phone_battery_pct=80, source_drone=0, report_confidence=0.8)


def test_probabilities_sum_to_one():
    p = TriageModel().predict(BASE)
    assert abs(sum(p["probabilities"].values()) - 1) < 1e-6
    assert 0 <= p["priority"] <= 100
    assert len(p["reasons"]) == 3


def test_danger_raises_priority():
    m = TriageModel()
    calm = m.predict(BASE)
    bad = m.predict({**BASE, "severe_injury": 1, "injured_count": 3, "water_level_m": 2.4,
                     "water_rise_m_per_h": 0.5, "medical_need": 1, "has_food_water": 0})
    assert bad["priority"] > calm["priority"]
    assert bad["triage"] == "CRITICAL" and calm["triage"] == "STABLE"


def test_encryption_roundtrip_and_masking():
    v = Vault()
    tok = v.encrypt("9876543210")
    assert "9876543210" not in tok
    assert v.decrypt(tok) == "9876543210"
    assert v.mask_phone("9876543210") == "98******10"
    assert v.pseudonym("9876543210") == v.pseudonym("9876543210")


def test_dispatch_prioritises_critical():
    m = TriageModel()
    incs = []
    for i in range(30):
        r = {**BASE, "incident_id": f"T-{i}", "reported_at": 0}
        if i % 10 == 0:
            r.update(severe_injury=1, injured_count=4, water_level_m=2.5, water_rise_m_per_h=0.5)
        incs.append({**r, **m.predict(r)})
    out = plan(incs, default_fleet())
    assert out[0]["triage"] == "CRITICAL"
    assert len(out) == len(incs)


def test_simulate_runs():
    m = TriageModel()
    incs = []
    for i in range(20):
        r = {**BASE, "incident_id": f"S-{i}", "reported_at": i * 3, "time_to_harm_min": 120}
        incs.append({**r, **m.predict(r), "true_label": "STABLE"})
    res = simulate(incs, order="ai")
    assert res["reached_in_time"]["STABLE"]["total"] == 20
