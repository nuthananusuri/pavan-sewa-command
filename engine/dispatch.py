"""
Resource dispatcher: assigns medical drones and rescue boats to incidents.

Greedy, priority-first assignment with an ETA tie-break:
  1. Sort open incidents by AI priority (highest first).
  2. For each incident pick the eligible resource with the earliest arrival
     time (resource's next-free time + travel time).
  3. Drones carry a medical / survival kit (payload limit from the PAVAN SEWA
     drone design, 6 kg) and are preferred for medical-need cases and for
     triage while a boat is on the way; boats evacuate people (capacity 10).

`simulate()` replays a whole incident set and measures how many incidents are
reached before their (ground-truth) time-to-harm. It is used to compare the
AI ordering against first-come-first-served.
"""
import math
from dataclasses import dataclass, field

BASES = [
    {"name": "Srirangam Base", "lat": 10.8620, "lon": 78.6930},
    {"name": "Trichy Junction Base", "lat": 10.7950, "lon": 78.6850},
    {"name": "Thiruverumbur Base", "lat": 10.7800, "lon": 78.7780},
]


def haversine_km(a_lat, a_lon, b_lat, b_lon):
    r = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = p2 - p1, math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


@dataclass
class Resource:
    rid: str
    kind: str            # "drone" | "boat"
    lat: float
    lon: float
    speed_kmh: float
    service_min: float   # time on scene
    capacity: int = 0
    payload_kg: float = 0.0
    free_at: float = 0.0  # minutes from scenario start
    jobs: list = field(default_factory=list)


def default_fleet():
    fleet = []
    for bi, b in enumerate(BASES):
        for k in range(3):
            fleet.append(Resource(f"DRN-{bi+1}{k+1}", "drone", b["lat"], b["lon"], 60.0, 6.0, payload_kg=6.0))
        for k in range(3):
            fleet.append(Resource(f"BOAT-{bi+1}{k+1}", "boat", b["lat"], b["lon"], 22.0, 15.0, capacity=10))
    return fleet


def _needs_boat(inc):
    return inc.get("triage") in ("CRITICAL", "URGENT") and (
        float(inc.get("water_level_m", 0)) > 1.0 or int(inc.get("severe_injury", 0)) == 1
    )


def _eta(res, inc, now):
    d = haversine_km(res.lat, res.lon, float(inc["lat"]), float(inc["lon"]))
    travel = 60.0 * d / res.speed_kmh
    return max(now, res.free_at) + travel, travel


def plan(incidents, fleet, now=0.0, order="ai"):
    """Return a list of assignment dicts. Mutates fleet free_at times."""
    if order == "ai":
        queue = sorted(incidents, key=lambda i: (-float(i["priority"]), float(i.get("reported_at", 0))))
    else:  # first come, first served
        queue = sorted(incidents, key=lambda i: float(i.get("reported_at", 0)))

    out = []
    for inc in queue:
        start = max(now, float(inc.get("reported_at", 0)))
        if inc.get("triage") == "STABLE" and order == "ai":
            kinds = ["drone"]  # supply drop / check-in only
        elif _needs_boat(inc):
            kinds = ["boat"]
        else:
            kinds = ["drone", "boat"]
        if int(inc.get("medical_need", 0)) == 1 and "drone" not in kinds:
            kinds = kinds + ["drone"]

        best = None
        for r in fleet:
            if r.kind not in kinds:
                continue
            arrive, travel = _eta(r, inc, start)
            # a drone reaching first counts as first contact (medical kit / comms)
            if best is None or arrive < best[0]:
                best = (arrive, travel, r)
        if best is None:
            continue
        arrive, travel, r = best
        trips = 1
        if r.kind == "boat":
            trips = max(1, math.ceil(int(inc.get("people_count", 1)) / r.capacity))
        r.free_at = arrive + r.service_min + travel * (2 * trips - 1)
        r.lat, r.lon = (float(inc["lat"]), float(inc["lon"])) if r.kind == "drone" else (r.lat, r.lon)
        r.jobs.append(inc["incident_id"])
        out.append({
            "incident_id": inc["incident_id"],
            "resource": r.rid,
            "kind": r.kind,
            "eta_min": round(arrive - start, 1),
            "arrive_at_min": round(arrive, 1),
            "priority": inc.get("priority"),
            "triage": inc.get("triage"),
        })
    return out


def simulate(incidents, order="ai", fleet=None, step_min=2.0, horizon_min=24 * 60):
    """
    Event-driven replay: incidents arrive at `reported_at`; every `step_min`
    the control room assigns waiting incidents to resources that are free
    right now, in either AI-priority order or first-come-first-served order.
    Scores how many incidents are reached before their true time-to-harm.
    """
    fleet = fleet or default_fleet()
    pending_all = sorted(incidents, key=lambda i: float(i.get("reported_at", 0)))
    arrived, ptr, t = [], 0, 0.0
    reach = {}
    while t <= horizon_min and (ptr < len(pending_all) or arrived):
        while ptr < len(pending_all) and float(pending_all[ptr].get("reported_at", 0)) <= t:
            arrived.append(pending_all[ptr])
            ptr += 1
        if order == "ai":
            arrived.sort(key=lambda i: -float(i["priority"]))
        still = []
        for inc in arrived:
            kinds = ["boat"] if _needs_boat(inc) else ["drone", "boat"]
            free = [r for r in fleet if r.kind in kinds and r.free_at <= t]
            if not free:
                still.append(inc)
                continue
            r = min(free, key=lambda r: haversine_km(r.lat, r.lon, float(inc["lat"]), float(inc["lon"])))
            arrive, travel = _eta(r, inc, t)
            trips = max(1, math.ceil(int(inc.get("people_count", 1)) / r.capacity)) if r.kind == "boat" else 1
            r.free_at = arrive + r.service_min + travel * (2 * trips - 1)
            reach[inc["incident_id"]] = arrive
        arrived = still
        t += step_min

    saved = {"CRITICAL": [0, 0], "URGENT": [0, 0], "STABLE": [0, 0]}
    waits_critical = []
    for inc in incidents:
        truth = inc.get("true_label", inc.get("triage"))
        rep = float(inc.get("reported_at", 0))
        wait = reach.get(inc["incident_id"], horizon_min + 60) - rep
        saved[truth][1] += 1
        saved[truth][0] += int(wait <= float(inc["time_to_harm_min"]))
        if truth == "CRITICAL":
            waits_critical.append(wait)
    waits_critical.sort()
    return {
        "order": order,
        "reached_in_time": {k: {"ok": v[0], "total": v[1], "pct": round(100 * v[0] / v[1], 1) if v[1] else None} for k, v in saved.items()},
        "critical_median_wait_min": round(waits_critical[len(waits_critical) // 2], 1) if waits_critical else None,
    }
