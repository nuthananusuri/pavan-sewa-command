"""
Latency / throughput benchmark of the triage engine on the current host.
Run it on your laptop and on LinuxONE and put both numbers in the deck.

    python scripts/benchmark.py
"""
import os
import platform
import statistics
import sys
import time

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from engine.dataio import load_incidents  # noqa: E402
from engine.dispatch import default_fleet, plan  # noqa: E402
from engine.privacy import Vault  # noqa: E402
from engine.triage import TriageModel  # noqa: E402


def main():
    rows = load_incidents()
    m = TriageModel()
    for r in rows[:200]:
        m.predict(r)  # warm-up
    lat = []
    t_all = time.perf_counter()
    for r in rows:
        t = time.perf_counter_ns()
        m.predict(r)
        lat.append((time.perf_counter_ns() - t) / 1000)
    total = time.perf_counter() - t_all
    lat.sort()

    v = Vault()
    t = time.perf_counter()
    for r in rows[:2000]:
        v.encrypt(r["reporter_phone"])
    enc_us = (time.perf_counter() - t) / 2000 * 1e6

    inc = []
    for r in rows[:150]:
        p = m.predict(r)
        inc.append({**r, **p, "reported_at": 0})
    t = time.perf_counter()
    plan(inc, default_fleet())
    plan_ms = (time.perf_counter() - t) * 1000

    print(f"host                 : {platform.machine()}  {platform.platform()}")
    print(f"python               : {platform.python_version()}")
    print(f"triage inferences    : {len(rows)}")
    print(f"latency p50 / p99    : {lat[len(lat)//2]:.1f} us / {lat[int(len(lat)*0.99)]:.1f} us")
    print(f"mean latency         : {statistics.mean(lat):.1f} us")
    print(f"throughput (1 core)  : {len(rows)/total:,.0f} incidents/s")
    print(f"AES encrypt per field: {enc_us:.1f} us")
    print(f"dispatch plan 150 inc: {plan_ms:.1f} ms")


if __name__ == "__main__":
    main()
