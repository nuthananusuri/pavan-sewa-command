"""
Storage layer (SQLite by default; schema is plain SQL so it ports directly to
Db2 for z/OS, Db2 on LinuxONE or PostgreSQL on LinuxONE).

Personal data (reporter name/phone) is stored ONLY as ciphertext. Every
human decision (approve dispatch, reveal contact) is written to an
append-only audit table.
"""
import json
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id TEXT PRIMARY KEY,
    reported_at REAL,
    pseudonym TEXT,
    phone_masked TEXT,
    name_enc TEXT,
    phone_enc TEXT,
    lat REAL, lon REAL,
    source TEXT,
    features TEXT,
    triage TEXT,
    priority REAL,
    probabilities TEXT,
    reasons TEXT,
    needs_review INTEGER,
    inference_us REAL,
    status TEXT DEFAULT 'open',
    assigned_resource TEXT,
    eta_min REAL
);
CREATE INDEX IF NOT EXISTS idx_open ON incidents(status, priority);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    actor TEXT,
    action TEXT,
    incident_id TEXT,
    detail TEXT
);
"""

FEATURE_KEYS = [
    "people_count", "injured_count", "severe_injury", "vulnerable_count", "medical_need",
    "water_level_m", "water_rise_m_per_h", "structure_risk", "hours_since_report",
    "has_food_water", "phone_battery_pct", "source_drone", "report_confidence",
]


class Store:
    def __init__(self, path="pavan_sewa.db"):
        self.path = path
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def reset(self):
        with self.lock:
            self.db.executescript("DELETE FROM incidents; DELETE FROM audit_log;")
            self.db.commit()

    def insert(self, inc, pred, vault, inference_us):
        feats = {k: inc.get(k) for k in FEATURE_KEYS}
        row = (
            inc["incident_id"], inc.get("reported_at", time.time()),
            vault.pseudonym(inc.get("reporter_phone", inc["incident_id"])),
            vault.mask_phone(inc.get("reporter_phone", "")),
            vault.encrypt(inc.get("reporter_name", "")),
            vault.encrypt(inc.get("reporter_phone", "")),
            float(inc["lat"]), float(inc["lon"]), inc.get("source", "sos_call"),
            json.dumps(feats), pred["triage"], pred["priority"],
            json.dumps(pred["probabilities"]), json.dumps(pred["reasons"]),
            int(pred["needs_human_review"]), inference_us,
        )
        with self.lock:
            self.db.execute(
                "INSERT OR REPLACE INTO incidents (incident_id, reported_at, pseudonym, phone_masked, name_enc, phone_enc,"
                " lat, lon, source, features, triage, priority, probabilities, reasons, needs_review, inference_us)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
            self.db.commit()

    def _row(self, r):
        d = dict(r)
        d.pop("name_enc", None)
        d.pop("phone_enc", None)
        for k in ("features", "probabilities", "reasons"):
            d[k] = json.loads(d[k]) if d[k] else None
        d.update(d.pop("features") or {})
        return d

    def list(self, status=None, limit=500):
        q = "SELECT * FROM incidents"
        args = []
        if status:
            q += " WHERE status = ?"
            args.append(status)
        q += " ORDER BY priority DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            return [self._row(r) for r in self.db.execute(q, args).fetchall()]

    def get_raw(self, incident_id):
        with self.lock:
            r = self.db.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,)).fetchone()
        return dict(r) if r else None

    def assign(self, incident_id, resource, eta, actor):
        with self.lock:
            self.db.execute("UPDATE incidents SET status='dispatched', assigned_resource=?, eta_min=? WHERE incident_id=?",
                            (resource, eta, incident_id))
            self.db.commit()
        self.audit(actor, "approve_dispatch", incident_id, {"resource": resource, "eta_min": eta})

    def audit(self, actor, action, incident_id, detail):
        with self.lock:
            self.db.execute("INSERT INTO audit_log (ts, actor, action, incident_id, detail) VALUES (?,?,?,?,?)",
                            (time.time(), actor, action, incident_id, json.dumps(detail)))
            self.db.commit()

    def audit_tail(self, n=30):
        with self.lock:
            rows = self.db.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [dict(r) for r in rows]

    def stats(self):
        with self.lock:
            rows = self.db.execute(
                "SELECT status, triage, COUNT(*) c FROM incidents GROUP BY status, triage").fetchall()
            lat = self.db.execute("SELECT AVG(inference_us), MAX(inference_us), COUNT(*) FROM incidents").fetchone()
        out = {"open": {"CRITICAL": 0, "URGENT": 0, "STABLE": 0}, "dispatched": 0, "total": lat[2] or 0,
               "avg_inference_us": round(lat[0] or 0, 1), "max_inference_us": round(lat[1] or 0, 1)}
        for r in rows:
            if r["status"] == "open":
                out["open"][r["triage"]] = r["c"]
            else:
                out["dispatched"] += r["c"]
        return out
