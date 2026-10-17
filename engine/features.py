"""Feature engineering shared by training and live inference (pure Python)."""

FEATURE_NAMES = [
    "people_count_capped",
    "injured_count_capped",
    "severe_injury",
    "vulnerable_count_capped",
    "medical_need",
    "water_above_knee_m",
    "water_rise_m_per_h",
    "flood_momentum",          # water depth x rise rate
    "structure_risk",
    "hours_waiting_capped",
    "has_food_water",
    "phone_battery_low",
    "source_drone",
    "report_confidence",
]

HUMAN_NAMES = {
    "people_count_capped": "people at site",
    "injured_count_capped": "injured people",
    "severe_injury": "severe injury reported",
    "vulnerable_count_capped": "children / elderly",
    "medical_need": "urgent medical need",
    "water_above_knee_m": "water depth",
    "water_rise_m_per_h": "water rising fast",
    "flood_momentum": "deep AND rising water",
    "structure_risk": "building collapse risk",
    "hours_waiting_capped": "hours waiting",
    "has_food_water": "has food & water",
    "phone_battery_low": "phone battery almost dead",
    "source_drone": "confirmed by drone",
    "report_confidence": "report confidence",
}


def _f(r, k, default=0.0):
    try:
        return float(r.get(k, default))
    except (TypeError, ValueError):
        return float(default)


def engineer(r):
    water = _f(r, "water_level_m")
    rise = _f(r, "water_rise_m_per_h")
    return [
        min(_f(r, "people_count", 1), 40),
        min(_f(r, "injured_count"), 10),
        _f(r, "severe_injury"),
        min(_f(r, "vulnerable_count"), 10),
        _f(r, "medical_need"),
        max(0.0, water - 0.8),
        max(0.0, rise),
        max(0.0, water - 0.8) * max(0.0, rise),
        _f(r, "structure_risk"),
        min(_f(r, "hours_since_report"), 24),
        _f(r, "has_food_water"),
        1.0 if _f(r, "phone_battery_pct", 50) < 15 else 0.0,
        _f(r, "source_drone"),
        _f(r, "report_confidence", 0.7),
    ]
