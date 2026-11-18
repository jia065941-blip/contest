"""Pure, legal-observation M guidance used by the A--E native ablation.

The overlay never changes H/L orders, the frozen A00--A11 choice, satellite
timing, target identity, or the V12-R3 network.  It can only replace an M
target-track command with a bounded waypoint and/or clear M lateral acceleration.
"""
import copy
import math

LEVELS = {
    "a_current": 0,
    "b_sat_cache": 1,
    "c_h_relay": 2,
    "d_terminal": 3,
    "e_packaging": 4,
}


def _distance(a, b):
    lat = math.radians((a[1] + b[1]) / 2.0)
    dx = (a[0] - b[0]) * 111320.0 * math.cos(lat)
    dy = (a[1] - b[1]) * 110570.0
    return math.hypot(dx, dy)


def _append(history, track):
    if track is None:
        return
    stamp = float(track["timestamp_s"])
    if history and stamp <= history[-1]["timestamp_s"]:
        return
    history.append({"position": list(track["position"]), "timestamp_s": stamp})
    del history[:-12]


def _estimate(history):
    if not history:
        return None
    latest = history[-1]
    candidates = [row for row in history[:-1]
                  if 2.0 <= latest["timestamp_s"] - row["timestamp_s"] <= 20.0]
    velocity = None
    if candidates:
        prior = candidates[-1]
        dt = latest["timestamp_s"] - prior["timestamp_s"]
        mean_lat = math.radians((latest["position"][1] + prior["position"][1]) / 2.0)
        east = (latest["position"][0] - prior["position"][0]) * 111320.0 * math.cos(mean_lat) / dt
        north = (latest["position"][1] - prior["position"][1]) * 110570.0 / dt
        speed = math.hypot(east, north)
        # Public moving objectives are nominally 12 m/s.  Reject noisy jumps,
        # rather than inventing a velocity from a discontinuous source change.
        if 0.25 <= speed <= 20.0:
            velocity = [east, north]
    return {"position": list(latest["position"]),
            "timestamp_s": latest["timestamp_s"], "velocity_en_mps": velocity}


def _project(sample, now, bounds, max_age):
    if sample is None or sample["velocity_en_mps"] is None:
        return None
    age = max(0.0, float(now) - sample["timestamp_s"])
    if age > max_age:
        return None
    lon, lat = sample["position"][:2]
    east, north = sample["velocity_en_mps"]
    lat2 = lat + north * age / 110570.0
    lon2 = lon + east * age / max(1.0, 111320.0 * math.cos(math.radians((lat + lat2) / 2.0)))
    lon2 = min(max(lon2, bounds[0]), bounds[1])
    lat2 = min(max(lat2, bounds[2]), bounds[3])
    return [lon2, lat2]


def step(context, memory=None, level="a_current"):
    """Return an M-only command proposal and copied chronological memory."""
    if level not in LEVELS:
        raise ValueError("Unknown M-shadow level: " + str(level))
    memory = copy.deepcopy(memory or {})
    decision = {"state": "BASELINE", "source": None, "waypoint": None,
                "clear_maneuver": False, "target_id": context["target_id"],
                "target_override": None,
                "reason": "frozen current policy"}
    if context["role"] != "M" or LEVELS[level] == 0 or not context["binding_matches"]:
        return decision, memory

    track = context.get("own_track")
    if context.get("satellite_active") and track is not None:
        _append(memory.setdefault("satellite_history", []), track)
        memory["satellite_seen"] = True
    if context.get("relay_verified") and track is not None:
        _append(memory.setdefault("relay_history", []), track)

    sat = _estimate(memory.get("satellite_history", []))
    relay = _estimate(memory.get("relay_history", []))
    now = float(context["time_s"])
    track_age = None if track is None else max(0.0, now - float(track["timestamp_s"]))
    fresh = track_age is not None and track_age <= 2.0

    if context.get("satellite_active") and fresh:
        decision.update(state="SAT_CAPTURE", source="satellite", reason="fresh satellite track; retain action4")
        return decision, memory
    if LEVELS[level] >= 2 and context.get("relay_verified") and fresh:
        decision.update(state="H_RELAY", source="same_target_H", reason="fresh same-cluster H relay; retain action4")
        return decision, memory
    if fresh:
        decision.update(state="FRESH_TRACK", source="legal_current", reason="fresh legal track; retain action4")
        return decision, memory

    source, sample = None, None
    if LEVELS[level] >= 2 and relay is not None:
        source, sample = "same_target_H_memory", relay
    if sample is None and sat is not None:
        source, sample = "satellite_memory", sat
    max_age = 45.0 if LEVELS[level] <= 2 else (60.0 if LEVELS[level] == 3 else 75.0)
    point = _project(sample, now, context["map_bounds"], max_age)
    if point is None:
        decision["reason"] = "no bounded satellite/H velocity memory"
        return decision, memory

    distance = _distance(context["position"], point)
    terminal = LEVELS[level] >= 3 and distance <= (10000.0 if LEVELS[level] == 3 else 15000.0)
    packaged = LEVELS[level] >= 4 and context.get("same_target_m_count", 1) >= 2
    # Outside terminal flight, use sparse one-step corrections.  In terminal
    # flight use continuous predicted guidance until a fresh action-4 track is
    # recovered.  E only becomes more aggressive for an actual M package.
    period = 4 if LEVELS[level] <= 2 else (1 if terminal else (2 if packaged else 3))
    pulse = int(context["step"]) % period == 0
    decision.update(state="TERMINAL_COMMIT" if terminal else "COAST_PREDICT",
                    source=source, reason="bounded legal-memory extrapolation",
                    waypoint=point if pulse else None)
    if terminal:
        nearest = context.get("nearest_interceptor_m")
        decision["clear_maneuver"] = nearest is None or nearest > 8000.0
    return decision, memory

