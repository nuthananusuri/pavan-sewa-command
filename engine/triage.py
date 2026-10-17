"""
Dependency-free triage inference + per-incident explanations.

The model is a multinomial logistic regression exported to JSON, so scoring
one incident is ~14 multiply-adds per class: microseconds on IBM Z / LinuxONE,
no GPU, no external service, and the data never leaves the platform.
"""
import json
import math
import os

from .features import FEATURE_NAMES, HUMAN_NAMES, engineer

DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), "..", "model", "triage_model.json")


class TriageModel:
    def __init__(self, path=DEFAULT_MODEL):
        with open(path) as f:
            m = json.load(f)
        assert m["features"] == FEATURE_NAMES, "feature list mismatch between model and engine"
        self.meta = {k: m[k] for k in ("name", "version", "trained_at", "type")}
        self.labels = m["labels"]
        self.mean = m["scaler_mean"]
        self.scale = m["scaler_scale"]
        self.coef = m["coef"]
        self.intercept = m["intercept"]

    def _scaled(self, x):
        return [(v - mu) / (s or 1.0) for v, mu, s in zip(x, self.mean, self.scale)]

    def predict(self, incident, top_k=3):
        xs = self._scaled(engineer(incident))
        logits = [b + sum(w * v for w, v in zip(ws, xs)) for ws, b in zip(self.coef, self.intercept)]
        mx = max(logits)
        exps = [math.exp(l - mx) for l in logits]
        z = sum(exps)
        probs = [e / z for e in exps]
        cls = max(range(len(probs)), key=probs.__getitem__)

        # priority 0-100: expected severity
        priority = round(100 * (probs[2] + 0.45 * probs[1]), 1)

        # explanation: contribution of each feature to the CRITICAL logit
        # relative to the STABLE logit (log-odds critical vs stable)
        contrib = [
            (FEATURE_NAMES[i], (self.coef[2][i] - self.coef[0][i]) * xs[i])
            for i in range(len(xs))
        ]
        contrib.sort(key=lambda t: abs(t[1]), reverse=True)
        reasons = [
            {
                "feature": name,
                "label": HUMAN_NAMES[name],
                "effect": "raises" if val > 0 else "lowers",
                "weight": round(val, 3),
            }
            for name, val in contrib[:top_k]
        ]
        confidence = max(probs)
        return {
            "triage": self.labels[cls],
            "priority": priority,
            "probabilities": {lbl: round(p, 4) for lbl, p in zip(self.labels, probs)},
            "confidence": round(confidence, 3),
            "needs_human_review": confidence < 0.6,
            "reasons": reasons,
        }
