"""
PAVAN SEWA Command - real-time disaster triage & dispatch on IBM LinuxONE.

Run:   python app.py            (http://<host>:8080)
Env:   PSC_PORT=8080  PSC_DB=pavan_sewa.db  PSC_SIM_SPEED=1.0 (sim minutes per real second)
"""
import copy
import json
import os
import platform
import random
import threading
import time

from flask import Flask, jsonify, request, send_from_directory

from engine.dataio import load_test_incidents
from engine.dispatch import BASES, default_fleet, plan
from engine.privacy import Vault
from engine.store import Store
from engine.triage import TriageModel

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(ROOT, "static"))

model = TriageModel()
vault = Vault()
store = Store(os.environ.get("PSC_DB", os.path.join(ROOT, "pavan_sewa.db")))
SIM_SPEED = float(os.environ.get("PSC_SIM_SPEED", "1.0"))

STATE = {"t0": time.time(), "busy_until": {}, "sim_running": False, "sim_thread": None, "latencies": []}
FLEET_TEMPLATE = default_fleet()


def sim_now():
    return (time.time() - STATE["t0"]) * SIM_SPEED


def load_json(name):
    p = os.path.join(ROOT, "model", name)
    return json.load(open(p)) if os.path.exists(p) else None


def ingest(inc):
    t = time.perf_counter_ns()
    pred = model.predict(inc)
    us = (time.perf_counter_ns() - t) / 1000.0
    inc.setdefault("reported_at", sim_now())
    store.insert(inc, pred, vault, us)
    STATE["latencies"] = (STATE["latencies"] + [us])[-2000:]
    return pred, us


# ---------------------------------------------------------------- API
@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "arch": platform.machine(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "model": model.meta,
        "sim_minutes": round(sim_now(), 1),
        "on_ibm_z": platform.machine() == "s390x",
    })


@app.post("/api/incidents")
def post_incident():
    inc = request.get_json(force=True)
    inc.setdefault("incident_id", f"LIVE-{int(time.time()*1000) % 10**8}")
    for k in ("lat", "lon"):
        if k not in inc:
            return jsonify({"error": f"missing {k}"}), 400
    pred, us = ingest(inc)
    return jsonify({"incident_id": inc["incident_id"], "inference_us": round(us, 1), **pred})


@app.get("/api/incidents")
def get_incidents():
    return jsonify(store.list(status=request.args.get("status")))


def _live_fleet():
    now = sim_now()
    fleet = copy.deepcopy(FLEET_TEMPLATE)
    for r in fleet:
        r.free_at = max(0.0, STATE["busy_until"].get(r.rid, 0.0) - now)
    return fleet


@app.get("/api/fleet")
def fleet():
    now = sim_now()
    out = []
    for r in FLEET_TEMPLATE:
        bu = STATE["busy_until"].get(r.rid, 0.0)
        out.append({"rid": r.rid, "kind": r.kind, "lat": r.lat, "lon": r.lon,
                    "busy": bu > now, "free_in_min": round(max(0, bu - now), 1)})
    return jsonify({"bases": BASES, "resources": out})


@app.get("/api/plan")
def get_plan():
    open_inc = store.list(status="open", limit=200)
    for i in open_inc:
        i["reported_at"] = 0
    proposals = plan(open_inc, _live_fleet(), now=0.0, order="ai")
    return jsonify(proposals[: int(request.args.get("limit", 15))])


@app.post("/api/approve")
def approve():
    body = request.get_json(force=True)
    actor = body.get("actor", "operator-1")
    ids = body.get("incident_ids") or [body.get("incident_id")]
    proposals = {p["incident_id"]: p for p in json.loads(get_plan().get_data())}
    done = []
    for iid in ids:
        p = proposals.get(iid)
        if not p:
            continue
        now = sim_now()
        busy = now + p["eta_min"] * 2 + (6 if p["kind"] == "drone" else 15)
        STATE["busy_until"][p["resource"]] = busy
        store.assign(iid, p["resource"], p["eta_min"], actor)
        done.append(p)
    return jsonify({"approved": done})


@app.post("/api/reveal")
def reveal():
    """Decrypt reporter contact - restricted to rescue-lead role, always audited."""
    body = request.get_json(force=True)
    role, iid = body.get("role"), body.get("incident_id")
    if role != "rescue-lead":
        store.audit(body.get("actor", "unknown"), "reveal_denied", iid, {"role": role})
        return jsonify({"error": "forbidden: only rescue-lead may view victim contact"}), 403
    raw = store.get_raw(iid)
    if not raw:
        return jsonify({"error": "not found"}), 404
    store.audit(body.get("actor", "rescue-lead"), "reveal_contact", iid, {"reason": body.get("reason", "")})
    return jsonify({"incident_id": iid, "name": vault.decrypt(raw["name_enc"]), "phone": vault.decrypt(raw["phone_enc"])})


@app.get("/api/stats")
def stats():
    s = store.stats()
    lat = sorted(STATE["latencies"])
    if lat:
        s["p50_inference_us"] = round(lat[len(lat) // 2], 1)
        s["p99_inference_us"] = round(lat[min(len(lat) - 1, int(len(lat) * 0.99))], 1)
    s["model_metrics"] = load_json("metrics.json")
    s["dispatch_eval"] = load_json("dispatch_eval.json")
    s["sim_running"] = STATE["sim_running"]
    s["sim_minutes"] = round(sim_now(), 1)
    return jsonify(s)


@app.get("/api/audit")
def audit():
    return jsonify(store.audit_tail())


# ---------------------------------------------------------------- simulator
def _sim_loop(rate_per_sec, total):
    rows = load_test_incidents()
    random.shuffle(rows)
    feed = rows[:total]
    for r in feed:
        if not STATE["sim_running"]:
            break
        r = dict(r)
        r.pop("label", None)
        r.pop("time_to_harm_min", None)
        r["incident_id"] = r["incident_id"].replace("INC", "SOS")
        r["reported_at"] = sim_now()
        ingest(r)
        time.sleep(random.expovariate(rate_per_sec))
    STATE["sim_running"] = False


@app.post("/api/simulate")
def simulate_start():
    body = request.get_json(silent=True) or {}
    if body.get("action") == "stop":
        STATE["sim_running"] = False
        return jsonify({"sim_running": False})
    if body.get("reset", True):
        STATE["sim_running"] = False
        time.sleep(0.2)
        store.reset()
        STATE.update(t0=time.time(), busy_until={}, latencies=[])
    STATE["sim_running"] = True
    th = threading.Thread(target=_sim_loop, args=(float(body.get("rate", 3.0)), int(body.get("total", 120))), daemon=True)
    STATE["sim_thread"] = th
    th.start()
    return jsonify({"sim_running": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PSC_PORT", 8080)), threaded=True)
