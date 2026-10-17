"""
Synthetic flood-disaster incident generator for PAVAN SEWA Command.

Scenario: monsoon flooding along the Cauvery / Kollidam rivers around
Tiruchirappalli, Tamil Nadu. Each row is one incident report: an SOS call,
a drone-scout detection (from the PAVAN SEWA / Aero Shield scout drone), or a
social-media / helpline relay.

Ground truth is produced by a hidden "latent risk" function plus noise, so
the model never sees the formula, only the observable features. Each incident
also gets a ground-truth `time_to_harm_min`: if no help arrives within that
window the outcome is counted as a failure in the dispatch evaluation.

Usage:  python data/generate.py --n 6000 --seed 7 --out data/incidents.csv
Only uses the Python standard library so it runs unchanged on s390x LinuxONE.
"""
import argparse
import csv
import math
import random

CENTER_LAT, CENTER_LON = 10.8155, 78.6965   # Tiruchirappalli
FIRST_NAMES = ["Arun", "Priya", "Karthik", "Meena", "Ravi", "Lakshmi", "Suresh",
               "Divya", "Murugan", "Anitha", "Selvam", "Kavya", "Ganesh", "Revathi"]
LAST_NAMES = ["Kumar", "Raj", "Devi", "Pandian", "Subramani", "Nair", "Iyer",
              "Rajan", "Velu", "Shankar"]
SOURCES = ["sos_call", "drone_scout", "helpline_relay"]

FEATURES = [
    "people_count", "injured_count", "severe_injury", "vulnerable_count",
    "medical_need", "water_level_m", "water_rise_m_per_h", "structure_risk",
    "hours_since_report", "has_food_water", "phone_battery_pct",
    "source_drone", "report_confidence",
]


def river_distance_km(lat, lon):
    """Rough distance to a river band running roughly east-west through the city."""
    river_lat = CENTER_LAT + 0.012 * math.sin((lon - CENTER_LON) * 40)
    return abs(lat - river_lat) * 111.0


def make_incident(i, rng):
    lat = CENTER_LAT + rng.gauss(0, 0.06)
    lon = CENTER_LON + rng.gauss(0, 0.08)
    d_river = river_distance_km(lat, lon)

    source = rng.choices(SOURCES, weights=[0.55, 0.30, 0.15])[0]
    people = max(1, int(rng.lognormvariate(1.3, 0.8)))
    people = min(people, 60)
    injured = min(people, int(rng.expovariate(1.6) * people * 0.25))
    severe = 1 if injured > 0 and rng.random() < 0.28 else 0
    vulnerable = min(people, int(rng.betavariate(1.5, 4) * people + (rng.random() < 0.3)))
    medical_need = 1 if rng.random() < 0.12 + 0.03 * vulnerable else 0  # insulin, dialysis, pregnancy...

    proximity = math.exp(-d_river / 2.5)
    water = max(0.0, rng.gauss(0.4 + 2.2 * proximity, 0.45))
    rise = max(-0.1, rng.gauss(0.05 + 0.35 * proximity, 0.12))
    structure = min(1.0, max(0.0, rng.betavariate(2, 5) + 0.25 * proximity * rng.random()))
    hours = round(rng.expovariate(1 / 5.0), 2)
    food = 1 if rng.random() < 0.65 - 0.04 * hours else 0
    battery = int(min(100, max(1, rng.gauss(55 - 3 * hours, 25))))
    confidence = round(min(1.0, max(0.2, rng.gauss(0.9 if source == "drone_scout" else 0.72, 0.12))), 2)

    # ---------- hidden ground truth ----------
    latent = (
        1.5 * severe
        + 0.30 * min(injured, 10)
        + 1.0 * medical_need
        + 0.22 * min(vulnerable, 10)
        + 1.25 * max(0.0, water - 0.8)
        + 3.0 * max(0.0, rise)
        + 1.8 * structure
        + 0.08 * min(hours, 24)
        - 0.7 * food
        + 0.015 * min(people, 40)
        + 0.6 * (1 if battery < 15 else 0)
    ) * (0.6 + 0.4 * confidence) + rng.gauss(0, 0.45)

    if latent > 3.6:
        label = 2   # CRITICAL
    elif latent > 2.2:
        label = 1   # URGENT
    else:
        label = 0   # STABLE

    time_to_harm = max(15.0, 600.0 * math.exp(-0.42 * latent) + rng.gauss(0, 15))

    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    phone = "9" + "".join(str(rng.randint(0, 9)) for _ in range(9))

    return {
        "incident_id": f"INC-{i:05d}",
        "reporter_name": name,
        "reporter_phone": phone,
        "lat": round(lat, 5),
        "lon": round(lon, 5),
        "source": source,
        "people_count": people,
        "injured_count": injured,
        "severe_injury": severe,
        "vulnerable_count": vulnerable,
        "medical_need": medical_need,
        "water_level_m": round(water, 2),
        "water_rise_m_per_h": round(rise, 3),
        "structure_risk": round(structure, 3),
        "hours_since_report": hours,
        "has_food_water": food,
        "phone_battery_pct": battery,
        "source_drone": 1 if source == "drone_scout" else 0,
        "report_confidence": confidence,
        "label": label,
        "time_to_harm_min": round(time_to_harm, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/incidents.csv")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [make_incident(i, rng) for i in range(args.n)]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    counts = [sum(r["label"] == k for r in rows) for k in range(3)]
    print(f"wrote {len(rows)} incidents to {args.out}  stable/urgent/critical = {counts}")


if __name__ == "__main__":
    main()
