from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, isfinite, radians
from statistics import median

from .priors import initial_targets_from_observation


_KINDS = {21000: "H", 21001: "M", 21002: "L"}
_SPEED_KM_S = {"H": 0.200, "M": 0.120, "L": 0.100}


@dataclass(frozen=True)
class SpatialWaveDeployment:
    actions: list[list[float]]
    assignments: dict[int, dict]
    report: dict


@dataclass(frozen=True)
class _Route:
    entity_id: int
    target_id: int | None
    kind: str
    destination: tuple[float, float]
    direction: tuple[float, float]
    earliest: float
    latest: float
    lane: int
    role: str
    package_id: int | None


def _finite(value, name):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _rectangle(coordinates):
    if not isinstance(coordinates, list) or len(coordinates) != 1 or len(coordinates[0]) != 4:
        raise ValueError("Spatial waves require one shared rectangular deployment area")
    points = [tuple(_finite(value, "deployment coordinate") for value in point) for point in coordinates[0]]
    if any(len(point) != 2 for point in points):
        raise ValueError("Deployment coordinates must be longitude/latitude pairs")
    west, east = min(p[0] for p in points), max(p[0] for p in points)
    south, north = min(p[1] for p in points), max(p[1] for p in points)
    if west >= east or south >= north or set(points) != {(west, south), (west, north), (east, south), (east, north)}:
        raise ValueError("Deployment area must be a non-degenerate axis-aligned rectangle")
    return west, east, south, north


def _public_lanes(targets, xy):
    ordered = sorted(targets, key=lambda target: (xy(target.position.lon, target.position.lat)[1], target.entity_id))
    gaps = [xy(b.position.lon, b.position.lat)[1] - xy(a.position.lon, a.position.lat)[1] for a, b in zip(ordered, ordered[1:])]
    positive = [gap for gap in gaps if gap > 0.1]
    threshold = max(3.0, 1.5 * median(positive)) if positive else 3.0
    lanes = [[]]
    for index, target in enumerate(ordered):
        if index and gaps[index - 1] > threshold:
            lanes.append([])
        lanes[-1].append(target)
    return lanes


def _ray_interval(destination, seed, rectangle):
    """Intersect the destination-to-seed ray with the inset deployment box."""
    dx, dy = destination[0] - seed[0], destination[1] - seed[1]
    length = hypot(dx, dy)
    if length < 1e-9:
        raise ValueError("A public destination coincides with its deployment ray origin")
    direction = dx / length, dy / length
    minimum, maximum = 0.0, float("inf")
    for value, component, lower, upper in zip(destination, direction, rectangle[::2], rectangle[1::2]):
        if abs(component) < 1e-12:
            if not lower <= value <= upper:
                raise ValueError("Deployment ray does not intersect the deployment area")
            continue
        first, second = (value - lower) / component, (value - upper) / component
        minimum, maximum = max(minimum, min(first, second)), min(maximum, max(first, second))
    if minimum >= maximum:
        raise ValueError("Deployment ray has no positive-length usable segment")
    return direction, minimum, maximum


def build_spatial_wave_deployment(entities, public_targets, coordinates, map_bounds):
    """Return legal Nx4 positions, per-platform destinations and auditable ETAs.

    Every ETA is measured from the common launch instant at nominal constant
    speed. No launch wait, hold, or hidden-site coordinate is used. When H/M
    time intervals do not overlap, the report explicitly identifies that
    physical limitation and schedules their reachable arrivals separately.
    """
    west, east, south, north = _rectangle(coordinates)
    if not entities or any(type(key) is not int or item.get("type") not in _KINDS for key, item in entities.items()):
        raise ValueError("Own-platform input must contain integer IDs and only red platform types")
    for item in entities.values():
        _finite(item["position"]["alt"], "platform altitude")
    if not public_targets:
        raise ValueError("Spatial waves require at least one public 9400/9600 target")
    for key, item in public_targets.items():
        if type(key) is not int or item.get("type") not in (9400, 9600):
            raise ValueError("Only public 9400/9600 targets may enter spatial deployment")
        _finite(item["position"]["lon"], "public target longitude")
        _finite(item["position"]["lat"], "public target latitude")
    bounds = {key: _finite(map_bounds[key], key) for key in ("lonMin", "lonMax", "latMin", "latMax")}
    if not bounds["lonMin"] <= west < east <= bounds["lonMax"] or not bounds["latMin"] <= south < north <= bounds["latMax"]:
        raise ValueError("Deployment area must lie within the public map bounds")
    if any(not bounds["lonMin"] <= item["position"]["lon"] <= bounds["lonMax"] or not bounds["latMin"] <= item["position"]["lat"] <= bounds["latMax"] for item in public_targets.values()):
        raise ValueError("Public targets must lie within the public map bounds")

    lon0, lat0 = (west + east) / 2, (south + north) / 2
    lon_scale, lat_scale = 111.32 * cos(radians(lat0)), 111.32
    if lon_scale < 1e-6:
        raise ValueError("Spatial wave geometry does not support a polar deployment area")
    def xy(lon, lat):
        return (lon - lon0) * lon_scale, (lat - lat0) * lat_scale
    def geographic(point):
        return {"lon": lon0 + point[0] / lon_scale, "lat": lat0 + point[1] / lat_scale}

    left, bottom = xy(west, south)
    right, top = xy(east, north)
    # A proportional inset leaves a real margin at all four legal boundaries.
    margin_x, margin_y = (right - left) * 0.02, (top - bottom) * 0.02
    rectangle = (left + margin_x, right - margin_x, bottom + margin_y, top - margin_y)
    targets = initial_targets_from_observation({"entities": public_targets})
    lanes = _public_lanes(targets, xy)
    lane_by_target = {target.entity_id: index for index, lane in enumerate(lanes) for target in lane}
    lane_centers = [sum(xy(t.position.lon, t.position.lat)[1] * t.value for t in lane) / sum(t.value for t in lane) for lane in lanes]
    kinds = {kind: sorted(key for key, item in entities.items() if _KINDS[item["type"]] == kind) for kind in _SPEED_KM_S}
    counts = {}
    target_for = {}
    pairs = list(zip(kinds["H"], kinds["M"]))
    paired_ids = {key for pair in pairs for key in pair}
    packages = [list(pair) for pair in pairs] + [[key] for kind in ("H", "M") for key in kinds[kind] if key not in paired_ids]
    # Allocate actual nominal damage, not an interchangeable unit count.
    committed = {t.entity_id: 0.0 for t in targets}
    durability = {t.entity_id: _finite(public_targets[t.entity_id].get("health", {9400: 24.0, 9600: 18.0}[t.entity_type]), "public target health") for t in targets}
    if any(value <= 0 for value in durability.values()):
        raise ValueError("Public target health must be positive")
    def damage(package, target):
        hit = 0.8 if target.entity_type == 9400 else 0.6
        return sum(20.0 if _KINDS[entities[key]["type"]] == "H" else 5.0 for key in package) * hit
    for package in packages:
        deficient = [t for t in targets if committed[t.entity_id] + 1e-9 < durability[t.entity_id]]
        if deficient:
            target = max(deficient, key=lambda t: (
                t.value * min(damage(package, t), durability[t.entity_id] - committed[t.entity_id]) / durability[t.entity_id],
                -hypot(*xy(t.position.lon, t.position.lat)), -t.entity_id))
        else:
            # Remaining capacity protects the least-covered target first.
            target = min(targets, key=lambda t: (committed[t.entity_id] / durability[t.entity_id], -t.value, t.entity_id))
        for key in package:
            target_for[key] = target
        committed[target.entity_id] += damage(package, target)

    nearest = [min(hypot(*[a - b for a, b in zip(xy(t.position.lon, t.position.lat), xy(other.position.lon, other.position.lat))]) for other in targets if other.entity_id != t.entity_id) for t in targets] if len(targets) > 1 else [6.0]
    separation = max(30.0, min(120.0, median(nearest) / _SPEED_KM_S["M"] / 2))
    routes = {}
    all_ids = sorted(entities)
    lane_weights = [sum(t.value for t in lane) for lane in lanes]
    total_weight = sum(lane_weights)
    # Most L escort the actual strike packages. Only a minority searches.
    search_count = min(len(kinds["L"]), max(2, len(kinds["L"]) // 4)) if packages else len(kinds["L"])
    search_ids = kinds["L"][:search_count]
    cover_ids = kinds["L"][search_count:]
    package_for = {key: index for index, package in enumerate(packages) for key in package}
    for index, key in enumerate(cover_ids):
        package_for[key] = min(len(packages) - 1, index * len(packages) // len(cover_ids))
    search_group_count = (len(search_ids) + 2) // 3
    # Greedily cover distinct legal horizontal corridors. Public nests add
    # weight, but unseen dark-nest positions never enter this selection.
    lower, upper = rectangle[2] + margin_y, rectangle[3] - margin_y
    cells = [lower + (upper-lower)*i/100 for i in range(101)]
    weights = [1.0 + sum(t.value for t in targets if abs(xy(t.position.lon, t.position.lat)[1]-y) <= 2.0) for y in cells]
    candidates = sorted(set(cells + [max(lower, min(upper, y)) for y in lane_centers]))
    covered = set()
    search_centers = []
    for _ in range(search_group_count):
        center = max(candidates, key=lambda y: (
            sum(w for i, (c, w) in enumerate(zip(cells, weights)) if i not in covered and abs(c-y) <= 2.0),
            min((abs(y-other) for other in search_centers), default=0), -abs(y), -y))
        search_centers.append(center)
        covered.update(i for i, c in enumerate(cells) if abs(c-center) <= 2.0)
        candidates.remove(center)
    search_lanes = [min(range(len(lanes)), key=lambda i: abs(lane_centers[i]-y)) for y in search_centers]
    search_ordinals = {key: index for index, key in enumerate(search_ids)}
    public_x = [xy(target.position.lon, target.position.lat)[0] for target in targets]
    map_left, map_bottom = xy(bounds["lonMin"], bounds["latMin"])
    map_right, map_top = xy(bounds["lonMax"], bounds["latMax"])
    forward = 1 if sum(public_x) >= 0 else -1
    search_x = map_right - margin_x if forward == 1 else map_left + margin_x
    entry_x = min(public_x) if forward == 1 else max(public_x)
    for ordinal, key in enumerate(all_ids):
        kind = _KINDS[entities[key]["type"]]
        package_id = package_for.get(key)
        if key in search_ordinals:
            search_index = search_ordinals[key]
            group = search_index // 3
            lane_id = search_lanes[group]
            search_y = search_centers[group]
            search_y += (search_index % 3 - 1) * 0.1
            search_y = max(rectangle[2] + margin_y, min(rectangle[3] - margin_y, search_y))
            destination, target_id = (search_x, search_y), None
            role, seed_y = "search", search_y
        else:
            target = target_for[packages[package_id][0]] if kind == "L" else target_for[key]
            destination = xy(target.position.lon, target.position.lat)
            target_id, lane_id = (None if kind == "L" else target.entity_id), lane_by_target[target.entity_id]
            role = "cover" if kind == "L" else "strike"
            # Cover and the package use the same approach, rather than fanning
            # L across unrelated map edges.
            fan = min(2.0, (rectangle[3] - rectangle[2]) / max(1, len(lanes)) * 0.08)
            offset = ((ordinal + 0.5) / len(all_ids) - 0.5) * fan
            seed_y = max(rectangle[2] + margin_y, min(rectangle[3] - margin_y, lane_centers[lane_id])) + offset
            seed_y = max(rectangle[2] + margin_y * 0.1, min(rectangle[3] - margin_y * 0.1, seed_y))
        direction, minimum, maximum = _ray_interval(destination, (0.0, seed_y), rectangle)
        routes[key] = _Route(key, target_id, kind, destination, direction, minimum / _SPEED_KM_S[kind], maximum / _SPEED_KM_S[kind], lane_id, role, package_id)

    def remaining_after_entry(route):
        return max(0.0, (route.destination[0] - entry_x) / route.direction[0]) / _SPEED_KM_S[route.kind]

    scheduled = []
    cover_entries = {}
    low_sequence = {key: index for index, key in enumerate(kinds["L"])}
    for key in kinds["L"]:
        route = routes[key]
        # All L start toward the front of the legal rectangle. Small depth
        # offsets keep positions distinct without introducing launch holds.
        offset = min((low_sequence[key] % 3) * 2.0, (route.latest - route.earliest) * 0.25)
        eta = route.earliest + offset
        scheduled.append((eta, [key]))
        if route.role == "cover":
            entry = eta - remaining_after_entry(route)
            cover_entries.setdefault(route.package_id, []).append(entry)

    strike_windows = []
    unaligned_pairs = []
    support_limits = []
    for package_id, package in enumerate(packages):
        selected = [routes[key] for key in package]
        earliest, latest = max(r.earliest for r in selected), min(r.latest for r in selected)
        units = [(package, earliest, latest)]
        if earliest > latest:
            if len(package) > 1:
                unaligned_pairs.append({"platform_ids": package, "target_id": selected[0].target_id, "minimum_nominal_skew_seconds": earliest - latest})
            units = [([r.entity_id], r.earliest, r.latest) for r in selected]
        escort_entry = max(cover_entries.get(package_id, [0.0]))
        for ids, first, last in units:
            wanted = max(escort_entry + 30.0 + remaining_after_entry(routes[key]) for key in ids)
            eta = min(last, max(first, wanted))
            strike_windows.append((eta, last, ids))
            if wanted > last:
                support_limits.append({"platform_ids": ids, "package_id": package_id, "shortfall_seconds": wanted - last})
    # Interval grouping: choose the earliest deadline, then assemble every
    # compatible package into one arrival wave. Bounds already include the
    # escort lead where physically feasible. No unit waits to launch.
    strike_batches = []
    pending_windows = sorted(strike_windows, key=lambda item: (item[1], item[2][0]))
    while pending_windows:
        arrival = pending_windows[0][1]
        group = [item for item in pending_windows if item[0] <= arrival <= item[1]]
        pending_windows = [item for item in pending_windows if item not in group]
        batch_ids = [key for _, _, ids in group for key in ids]
        scheduled.extend((arrival, ids) for _, _, ids in group)
        strike_batches.append({"arrival_seconds": arrival, "platform_ids": batch_ids})
    # Split escort support across the actual H/M arrival times, including
    # physically unalignable pairs. Repositioning replaces launch waiting.
    strike_eta = {key: eta for eta, ids in scheduled for key in ids if routes[key].role == "strike"}
    scheduled = [(eta, ids) for eta, ids in scheduled if routes[ids[0]].role != "cover"]
    escort_partners = {}
    escort_alignment = []
    ordinal_by_package = {}
    for key in cover_ids:
        route = routes[key]
        members = packages[route.package_id]
        ordinal = ordinal_by_package.get(route.package_id, 0)
        ordinal_by_package[route.package_id] = ordinal + 1
        partner = members[ordinal % len(members)]
        escort_partners[key] = partner
        entry = strike_eta[partner] - remaining_after_entry(routes[partner])
        desired = entry - 30.0 + remaining_after_entry(route)
        eta = min(route.latest, max(route.earliest, desired))
        scheduled.append((eta, [key]))
        escort_alignment.append({"cover_id": key, "strike_id": partner,
            "actual_lead_seconds": entry - (eta-remaining_after_entry(route)),
            "position_limit_error_seconds": eta-desired})
    # Search survives behind its nearest public-lane escort screen, instead
    # of becoming the first interceptor target. All still launch together.
    cover_entry_by_lane = {}
    for eta, ids in scheduled:
        for key in ids:
            route = routes[key]
            if route.role == "cover":
                cover_entry_by_lane.setdefault(route.lane, []).append(eta-remaining_after_entry(route))
    scheduled = [(eta, ids) for eta, ids in scheduled if routes[ids[0]].role != "search"]
    search_alignment = []
    for key in search_ids:
        route = routes[key]
        entries = cover_entry_by_lane.get(route.lane)
        if not entries:
            entries = [entry for lane in cover_entry_by_lane.values() for entry in lane]
        desired = median(entries)+30.0+remaining_after_entry(route) if entries else route.earliest
        eta = min(route.latest, max(route.earliest, desired))
        scheduled.append((eta, [key]))
        search_alignment.append({"platform_id": key, "position_limit_error_seconds": eta-desired})
    scheduled.sort(key=lambda item: (item[0], item[1][0]))

    actions, assignments, waves = [], {}, []
    occupied = set()
    for eta, ids in scheduled:
        if not waves or eta - waves[-1]["first_arrival_seconds"] > separation * 0.25:
            waves.append({"wave_id": len(waves) + 1, "first_arrival_seconds": eta, "last_arrival_seconds": eta, "platform_ids": [], "kind_counts": {kind: 0 for kind in _SPEED_KM_S}})
        wave = waves[-1]
        wave["last_arrival_seconds"] = eta
        for key in ids:
            route = routes[key]
            distance = eta * _SPEED_KM_S[route.kind]
            point = geographic(tuple(value - distance * component for value, component in zip(route.destination, route.direction)))
            position = (point["lon"], point["lat"])
            if position in occupied:
                raise ValueError("Geometry produced overlapping deployment points")
            occupied.add(position)
            actions.append([0, key, point["lon"], point["lat"]])
            assignments[key] = {"target_id": route.target_id, "destination": geographic(route.destination), "role": route.role, "package_id": route.package_id, "escort_partner_id": escort_partners.get(key), "nominal_entry_seconds": eta - remaining_after_entry(route), "wave_id": wave["wave_id"], "arrival_seconds": eta, "kind": route.kind, "lane_id": route.lane + 1}
            wave["platform_ids"].append(key)
            wave["kind_counts"][route.kind] += 1
    report = {
        "strategy": "r7_spatial_packages", "variant": "r7plus_coordinated_v2", "all_launch_together": True,
        "role_counts": {"cover": len(cover_ids), "search": len(search_ids), "strike": len(kinds["H"]) + len(kinds["M"])},
        "search_stripe_spacing_km": 4.0, "search_coverage_note": "Limited public-route search; 3 km slant range with observed ~2 km cruise altitude; not full-map coverage.",
        "escort_lead_seconds": 30.0, "entry_reference_longitude": lon0 + entry_x / lon_scale,
        "escort_timing_shortfalls": support_limits,
        "escort_alignment": escort_alignment,
        "search_after_cover_seconds": 30.0, "search_alignment": search_alignment,
        "search_stripe_centers_latitude": [lat0+y/lat_scale for y in search_centers],
        "escort_note": "Position-derived arrival at public frontline is a planning proxy, not proof of clearance or that blue will target escorts.",
        "eta_basis": "seconds after common launch; straight path at nominal speed; reactive maneuvers and interception can change actual arrivals",
        "nominal_speed_m_s": {kind: speed * 1000 for kind, speed in _SPEED_KM_S.items()},
        "strike_arrival_batches": strike_batches,
        "damage_budget": [{"target_id": t.entity_id, "health": durability[t.entity_id], "nominal_committed_damage": committed[t.entity_id]} for t in targets],
        "public_target_ids": sorted(public_targets), "lane_count": len(lanes),
        "lanes": [{"lane_id": index + 1, "target_ids": [t.entity_id for t in lane], "center_latitude": lat0 + lane_centers[index] / lat_scale} for index, lane in enumerate(lanes)],
        "derived_wave_spacing_seconds": separation, "wave_count": len(waves), "waves": waves,
        "hm_package_count": len(pairs), "hm_common_arrival_packages": len(pairs) - len(unaligned_pairs),
        "hm_packages_without_common_arrival": unaligned_pairs,
    }
    return SpatialWaveDeployment(sorted(actions, key=lambda action: action[1]), assignments, report)
