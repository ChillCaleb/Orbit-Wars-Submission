import math
from collections import Counter, defaultdict, namedtuple


CENTER_X = 50.0
CENTER_Y = 50.0
ROTATION_RADIUS_LIMIT = 50.0
CORNER_CLUSTER_RADIUS = 26.0
LAUNCH_CLEARANCE = 0.1

QUADRANT_NAMES = ("Q0_SE", "Q1_SW", "Q2_NW", "Q3_NE")
QUADRANT_CORNERS = ((100.0, 100.0), (0.0, 100.0), (0.0, 0.0), (100.0, 0.0))

Planet = namedtuple("Planet", "id owner x y radius ships production")
Fleet = namedtuple("Fleet", "id owner x y angle from_planet_id ships")

TACTICAL_WEIGHTS = {
    "version": 2,
    "anchor": {
        "static": 1.5,
        "big": 3.0,
        "corner": 2.0,
        "support_small": 0.8,
        "safety": 2.0,
        "production": 1.0,
        "ships": 0.6,
    },
    "feeder": {
        "established_quadrant": 2.5,
        "static": 1.2,
        "meridian": 1.6,
        "surplus": 1.4,
        "safety": 1.8,
        "production": 0.8,
    },
    "sweeper": {
        "small_static": 2.2,
        "inner": 1.6,
        "equator": 1.2,
        "safety": 0.8,
        "ships": 0.5,
    },
    "strike_stage": {
        "surplus": 1.4,
        "frontier": 1.6,
        "safety": 0.9,
        "production": 0.6,
    },
}

ROLE_SCORE_NAMES = (
    "anchor",
    "feeder",
    "sweeper",
    "shield",
    "battery",
    "attacker",
    "expander",
)

ROLE_NAMES = (
    "anchor",
    "feeder",
    "sweeper",
    "shield",
    "battery",
    "attacker",
    "expander",
    "unknown",
)

PHASE_NAMES = (
    "initiation",
    "expansion",
    "established",
    "attack",
)

TREND_NAMES = (
    "chasing_leader",
    "pressured",
    "overtake_window",
    "cash_in",
    "neutral",
)

PRESSURE_ARRIVAL_WINDOW = 9.0
PRESSURE_MIN_SOURCE_COUNT = 2
PRESSURE_MIN_DEFENSE_RATIO = 0.45
PRESSURE_MIN_HOSTILE_SHIPS = 10
PRESSURE_ACTION_FEATURE_NAMES = (
    "pressure_direct_reinforcement",
    "pressure_arrival_slack",
    "pressure_need_coverage",
    "pressure_source_drain",
    "pressure_counter_source",
)

ACTION_FEATURE_SCALES = {
    "source_ships": 160.0,
    "source_prod": 5.0,
    "source_margin": 1200.0,
    "source_safety": 1.0,
    "source_corner": 100.0,
    "target_ships": 160.0,
    "target_prod": 5.0,
    "target_margin": 1200.0,
    "target_safety": 1.0,
    "target_corner": 100.0,
    "distance": 100.0,
    "ships_sent": 160.0,
    "send_ratio": 1.0,
    "need_gap": 160.0,
    "enemy_ships": 1800.0,
    "enemy_prod": 120.0,
    "enemy_target_ships": 1800.0,
    "our_target_ships": 1200.0,
    "neutral_target_ships": 1800.0,
    "enemy_fleet_target": 1800.0,
    "our_fleet_target": 1200.0,
    "eta": 12.0,
    "opportunity_value": 12.0,
    "opportunity_cost": 1.0,
    "ship_gap": 40.0,
}

LONG_FLIGHT_SECONDS = 5.0
STATIC_LONG_FLIGHT_SECONDS = 3.75
SUN_PENALTY = 0.95
MISS_PENALTY = 0.18
LONG_FLIGHT_PENALTY_CAP = 0.18
OPPORTUNITY_PENALTY_CAP = 0.40
SUN_DANGER_RADIUS = 11.0
SUN_CORRIDOR_RADIUS = 18.0
STATIC_LONG_FLIGHT_PENALTY_CAP = 0.28
STATIC_REMOTE_PENALTY_CAP = 0.24


def obs_get(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _raw_get(item, index, name):
    if isinstance(item, dict):
        return item[name]
    if isinstance(item, (list, tuple)):
        return item[index]
    return getattr(item, name)


def parse_planet(raw):
    return Planet(
        int(_raw_get(raw, 0, "id")),
        int(_raw_get(raw, 1, "owner")),
        float(_raw_get(raw, 2, "x")),
        float(_raw_get(raw, 3, "y")),
        float(_raw_get(raw, 4, "radius")),
        int(_raw_get(raw, 5, "ships")),
        int(_raw_get(raw, 6, "production")),
    )


def parse_fleet(raw):
    return Fleet(
        int(_raw_get(raw, 0, "id")),
        int(_raw_get(raw, 1, "owner")),
        float(_raw_get(raw, 2, "x")),
        float(_raw_get(raw, 3, "y")),
        float(_raw_get(raw, 4, "angle")),
        int(_raw_get(raw, 5, "from_planet_id")),
        int(_raw_get(raw, 6, "ships")),
    )


def planets_from_obs(obs):
    return [parse_planet(raw) for raw in obs_get(obs, "planets", []) or []]


def fleets_from_obs(obs):
    return [parse_fleet(raw) for raw in obs_get(obs, "fleets", []) or []]


def player_from_obs(obs):
    return int(obs_get(obs, "player", 0) or 0)


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def distance_xy(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def distance(a, b):
    return distance_xy(a.x, a.y, b.x, b.y)


def orbital_radius(planet):
    return distance_xy(planet.x, planet.y, CENTER_X, CENTER_Y)


def is_static(planet):
    return orbital_radius(planet) + planet.radius >= ROTATION_RADIUS_LIMIT


def is_rotating(planet):
    return not is_static(planet)


def size_label(planet):
    return "big" if int(planet.production) == 5 else "small"


def quadrant_index_xy(x, y):
    angle = math.atan2(y - CENTER_Y, x - CENTER_X)
    if angle < 0:
        angle += 2.0 * math.pi
    return int(angle // (math.pi / 2.0)) % 4


def quadrant_index(planet):
    return quadrant_index_xy(planet.x, planet.y)


def quadrant_name(index):
    return QUADRANT_NAMES[int(index) % 4]


def corner_distance(planet, quadrant=None):
    q = quadrant_index(planet) if quadrant is None else int(quadrant)
    cx, cy = QUADRANT_CORNERS[q % 4]
    return distance_xy(planet.x, planet.y, cx, cy)


def is_corner_node(planet):
    return is_static(planet) and corner_distance(planet) <= CORNER_CLUSTER_RADIUS


def planet_label(planet):
    q = quadrant_index(planet)
    return {
        "id": planet.id,
        "owner": planet.owner,
        "quadrant": quadrant_name(q),
        "quadrant_index": q,
        "static": is_static(planet),
        "rotating": is_rotating(planet),
        "size": size_label(planet),
        "ships": planet.ships,
        "production": planet.production,
        "corner_node": is_corner_node(planet),
        "corner_distance": round(corner_distance(planet, q), 3),
        "orbital_radius": round(orbital_radius(planet), 3),
    }


def owned_corner_groups(planets, owner, quadrant):
    owned = [
        planet
        for planet in planets
        if planet.owner == owner and is_corner_node(planet) and quadrant_index(planet) == quadrant
    ]
    big = [planet for planet in owned if size_label(planet) == "big"]
    small = [planet for planet in owned if size_label(planet) == "small"]
    return big, small


def anchor_corner_planet(planets, quadrant):
    quadrant = int(quadrant)
    corner_nodes = [planet for planet in planets if is_corner_node(planet) and quadrant_index(planet) == quadrant]
    candidates = corner_nodes or [planet for planet in planets if is_static(planet) and quadrant_index(planet) == quadrant]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda planet: (
            corner_distance(planet, quadrant),
            -int(planet.production),
            int(planet.ships),
            int(planet.id),
        ),
    )


def role_ready_quadrants(planets, owner):
    ready = set()
    for quadrant in range(4):
        big, small = owned_corner_groups(planets, owner, quadrant)
        if len(big) >= 1 and len(small) >= 2:
            ready.add(quadrant)
    return ready


def established_quadrants(planets, owner):
    established = set()
    for quadrant in range(4):
        anchor = anchor_corner_planet(planets, quadrant)
        if anchor is not None and int(anchor.owner) == int(owner):
            established.add(quadrant)
    return established


def _margin_score(margin, scale=900.0):
    return clamp((float(margin) + float(scale)) / (2.0 * float(scale)))


def _role_targets_for_planet(planets, player, source, control_half=None):
    same_side_targets = [
        planet
        for planet in planets
        if int(planet.id) != int(source.id)
        and int(planet.owner) != int(player)
        and same_equator_side(source, planet)
    ]
    if control_half:
        filtered = [planet for planet in same_side_targets if quadrant_index(planet) in control_half]
        same_side_targets = filtered or same_side_targets
    return same_side_targets


def _best_role_opportunity(source, targets, player, phase_name, control_half, planets):
    best = None
    for candidate in targets:
        ships = max(int(source.ships), _candidate_capture_need(candidate))
        eta = estimate_travel_eta(source, candidate, ships)
        if is_static(source) and eta > STATIC_LONG_FLIGHT_SECONDS + 2.0:
            continue
        value = _candidate_opportunity_value(source, candidate, ships, player, phase_name, control_half, planets)
        score = (value, -eta, -int(candidate.production), -int(candidate.ships), -int(candidate.id))
        if best is None or score > best["score"]:
            best = {
                "candidate": candidate,
                "value": value,
                "eta": eta,
                "need": _candidate_capture_need(candidate),
                "score": score,
            }
    return best


def _role_context_for_planet(
    planets,
    fleets,
    player,
    source,
    phase_name=None,
    anchor_quadrant=None,
    control_half=None,
    established=None,
    role_ready=None,
    pressure_cache=None,
    margin_cache=None,
):
    player = int(player)
    q = quadrant_index(source)
    static = is_static(source)
    big = size_label(source) == "big"
    pressure_cache = pressure_cache or {}
    pressure = pressure_cache.get(int(source.id)) or pressure_profile(source, planets, player)
    ships_norm = clamp(int(source.ships) / 120.0)
    prod_norm = clamp(int(source.production) / 5.0)
    surplus_floor = 40 if big else 20
    surplus_score = clamp(max(0, int(source.ships) - surplus_floor) / 120.0)
    margin_cache = margin_cache or {}
    local_margin = margin_cache.get(int(q))
    if local_margin is None:
        local_margin = quadrant_control_margin(planets, fleets, player, q)
    established = established_quadrants(planets, player) if established is None else established
    role_ready = role_ready_quadrants(planets, player) if role_ready is None else role_ready
    if anchor_quadrant is None:
        anchor_quadrant = best_anchor_quadrant(planets, fleets, player)
    if control_half is None:
        control_half = current_control_half(planets, fleets, player, anchor_quadrant=anchor_quadrant)
    open_quadrants = (set(control_half) if control_half else set(range(4))) - established

    same_side_targets = _role_targets_for_planet(planets, player, source, control_half=control_half)
    same_quadrant_targets = [planet for planet in same_side_targets if quadrant_index(planet) == q]
    open_targets = [planet for planet in same_side_targets if quadrant_index(planet) in open_quadrants]
    enemy_targets = [planet for planet in same_side_targets if int(planet.owner) not in (-1, player)]
    nearby_friendlies = [
        planet
        for planet in planets
        if int(planet.owner) == player and int(planet.id) != int(source.id) and same_equator_side(source, planet)
    ]

    best_same_quadrant = _best_role_opportunity(source, same_quadrant_targets, player, phase_name, control_half, planets)
    best_open = _best_role_opportunity(source, open_targets or same_side_targets, player, phase_name, control_half, planets)
    best_enemy = _best_role_opportunity(source, enemy_targets, player, phase_name, control_half, planets)

    cheap_targets = sum(1 for planet in same_side_targets if _candidate_capture_need(planet) <= int(source.ships))
    cheap_same_quadrant = sum(1 for planet in same_quadrant_targets if _candidate_capture_need(planet) <= int(source.ships))
    cheap_small_statics = sum(
        1
        for planet in same_quadrant_targets
        if is_static(planet) and size_label(planet) == "small" and _candidate_capture_need(planet) <= int(source.ships)
    )

    friendly_support_need = 0.0
    for friendly in nearby_friendlies:
        if control_half and quadrant_index(friendly) not in control_half:
            continue
        margin = quadrant_control_margin(planets, fleets, player, quadrant_index(friendly))
        friendly_pressure = pressure_cache.get(int(friendly.id)) or pressure_profile(friendly, planets, player)
        threat = 1.0 - float(friendly_pressure["safety"])
        if margin < 0:
            friendly_support_need += 0.6
        friendly_support_need += max(0.0, threat - 0.45)
    friendly_support_need = clamp(friendly_support_need / 2.5)

    if enemy_targets:
        nearest_enemy = min(distance(source, enemy) for enemy in enemy_targets)
    else:
        nearest_enemy = 100.0

    return {
        "quadrant": q,
        "static": static,
        "rotating": not static,
        "big": big,
        "corner": is_corner_node(source),
        "safety": float(pressure["safety"]),
        "ships_norm": ships_norm,
        "prod_norm": prod_norm,
        "surplus_score": surplus_score,
        "inner_score": clamp(1.0 - orbital_radius(source) / ROTATION_RADIUS_LIMIT),
        "outer_score": clamp(corner_distance(source, q) / 55.0),
        "equator_score": clamp(1.0 - abs(float(source.y) - CENTER_Y) / 50.0),
        "meridian_score": clamp(
            1.0 - min(abs(float(source.x) - CENTER_X), abs(float(source.y) - CENTER_Y)) / 50.0
        ),
        "local_margin_score": _margin_score(local_margin),
        "in_established": 1.0 if q in established else 0.0,
        "role_ready": 1.0 if q in role_ready else 0.0,
        "in_control_half": 1.0 if control_half and q in control_half else 0.0,
        "open_quadrant_score": 1.0 if q in open_quadrants else 0.0,
        "best_same_quadrant_value": 0.0 if best_same_quadrant is None else clamp(best_same_quadrant["value"] / 8.0),
        "best_open_value": 0.0 if best_open is None else clamp(best_open["value"] / 8.0),
        "best_enemy_value": 0.0 if best_enemy is None else clamp(best_enemy["value"] / 8.0),
        "best_same_quadrant_eta": 1.0 if best_same_quadrant is None else clamp(best_same_quadrant["eta"] / 8.0),
        "cheap_targets": clamp(cheap_targets / 4.0),
        "cheap_same_quadrant": clamp(cheap_same_quadrant / 4.0),
        "cheap_small_statics": clamp(cheap_small_statics / 3.0),
        "friendly_support_need": friendly_support_need,
        "frontier_score": clamp(1.0 - nearest_enemy / 80.0),
        "anchor_quadrant_match": 1.0 if anchor_quadrant is not None and q == int(anchor_quadrant) else 0.0,
        "has_open_quadrants": 1.0 if open_quadrants else 0.0,
        "pressure": pressure,
    }


def _normalize_role_scores(raw_scores):
    positive = {name: max(0.0, float(raw_scores.get(name, 0.0))) for name in ROLE_SCORE_NAMES}
    total = sum(positive.values())
    if total <= 1e-6:
        positive["expander"] = 1.0
        total = 1.0
    scores = {name: positive[name] / total for name in ROLE_SCORE_NAMES}
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    primary_role, primary_score = ordered[0]
    secondary_score = ordered[1][1] if len(ordered) > 1 else 0.0
    scores["primary_role"] = primary_role
    scores["primary_score"] = primary_score
    scores["role_certainty"] = clamp(primary_score - secondary_score)
    scores["raw_scores"] = positive
    return scores


def _role_summary_from_profiles(owned, profiles, established, top_n=5):
    rows = []
    for planet in owned:
        q = quadrant_index(planet)
        profile = profiles.get(int(planet.id), {})
        rows.append(
            {
                "planet_id": int(planet.id),
                "quadrant": quadrant_name(q),
                "anchor_score": round(float(profile.get("anchor", 0.0)) * 12.0, 4),
                "feeder_score": round(float(profile.get("feeder", 0.0)) * 12.0, 4),
                "sweeper_score": round(float(profile.get("sweeper", 0.0)) * 12.0, 4),
                "strike_stage_score": round(float(profile.get("attacker", 0.0)) * 12.0, 4),
                "primary_role": profile.get("primary_role", "expander"),
                "role_certainty": round(float(profile.get("role_certainty", 0.0)), 4),
                "labels": planet_label(planet),
                "pressure": profile.get("pressure", {}),
            }
        )

    def top_by(key):
        return sorted(rows, key=lambda item: (-item[key], item["planet_id"]))[:top_n]

    return {
        "weights_version": TACTICAL_WEIGHTS["version"],
        "established_quadrants": [quadrant_name(q) for q in sorted(established)],
        "anchor_candidates": top_by("anchor_score"),
        "feeder_candidates": top_by("feeder_score"),
        "sweeper_candidates": top_by("sweeper_score"),
        "strike_stage_candidates": top_by("strike_stage_score"),
    }


def role_confidence_map_from_state(
    planets,
    fleets,
    player,
    phase_name=None,
    anchor_planet_id=None,
    attacker_planet_id=None,
    feeder_planet_id=None,
):
    player = int(player)
    owned = [planet for planet in planets if int(planet.owner) == player]
    if not owned:
        return {}

    phase_name = phase_name or infer_phase_from_state(planets, fleets, player)
    established = established_quadrants(planets, player)
    role_ready = role_ready_quadrants(planets, player)
    anchor_quadrant = best_anchor_quadrant(planets, fleets, player)
    control_half = current_control_half(planets, fleets, player, anchor_quadrant=anchor_quadrant)
    pressure_cache = {int(planet.id): pressure_profile(planet, planets, player) for planet in owned}
    margin_cache = {quadrant: quadrant_control_margin(planets, fleets, player, quadrant) for quadrant in range(4)}

    profiles = {}
    for planet in owned:
        context = _role_context_for_planet(
            planets,
            fleets,
            player,
            planet,
            phase_name=phase_name,
            anchor_quadrant=anchor_quadrant,
            control_half=control_half,
            established=established,
            role_ready=role_ready,
            pressure_cache=pressure_cache,
            margin_cache=margin_cache,
        )
        q = context["quadrant"]
        raw = {
            "anchor": (
                (2.4 if context["static"] else 0.0)
                + (2.1 if context["corner"] else 0.0)
                + (1.9 if context["big"] else 0.0)
                + 1.6 * context["safety"]
                + 0.8 * context["prod_norm"]
                + 0.5 * context["ships_norm"]
                + 0.9 * context["local_margin_score"]
                + 1.0 * context["anchor_quadrant_match"]
                + 0.6 * context["in_established"]
            ),
            "feeder": (
                (1.6 if context["static"] else 0.0)
                + 1.8 * context["surplus_score"]
                + 1.2 * context["safety"]
                + 0.9 * context["meridian_score"]
                + 1.3 * context["friendly_support_need"]
                + 0.9 * context["best_open_value"]
                + 0.7 * context["in_established"]
            ),
            "sweeper": (
                (1.8 if context["static"] and not context["big"] else 0.0)
                + 1.2 * context["inner_score"]
                + 1.1 * context["equator_score"]
                + 0.8 * context["safety"]
                + 1.4 * context["cheap_small_statics"]
                + 0.8 * context["cheap_same_quadrant"]
                + 0.5 * (1.0 - context["best_same_quadrant_eta"])
            ),
            "shield": (
                (1.4 if context["static"] and not context["big"] else 0.0)
                + 1.4 * context["outer_score"]
                + 1.1 * context["safety"]
                + 0.8 * context["ships_norm"]
                + 0.9 * context["frontier_score"]
                + 0.6 * context["in_established"]
            ),
            "battery": (
                (2.3 if context["static"] and context["big"] else 0.0)
                + 1.7 * context["surplus_score"]
                + 1.0 * context["safety"]
                + 1.0 * context["prod_norm"]
                + 0.8 * context["ships_norm"]
                + 0.7 * context["role_ready"]
                + 0.5 * context["in_established"]
            ),
            "attacker": (
                (2.1 if context["rotating"] else 0.0)
                + 1.5 * context["surplus_score"]
                + 1.3 * context["frontier_score"]
                + 1.2 * context["best_open_value"]
                + 0.9 * context["best_enemy_value"]
                + 0.7 * context["in_control_half"]
                + 0.5 * context["has_open_quadrants"]
            ),
            "expander": (
                1.2
                + 1.5 * context["best_same_quadrant_value"]
                + 1.3 * context["best_open_value"]
                + 1.1 * context["cheap_targets"]
                + 0.7 * context["cheap_same_quadrant"]
                + 0.6 * context["open_quadrant_score"]
                + 0.5 * context["surplus_score"]
            ),
        }

        if phase_name == "initiation":
            raw["expander"] += 1.4
            raw["anchor"] *= 0.75
            raw["feeder"] *= 0.55
            raw["battery"] *= 0.55
        elif phase_name == "expansion":
            raw["expander"] += 1.0
            raw["attacker"] += 0.4
            raw["feeder"] *= 0.7
        elif phase_name == "established":
            raw["anchor"] += 0.9
            raw["feeder"] += 0.8
            raw["battery"] += 0.7
        elif phase_name == "attack":
            raw["attacker"] += 1.0
            raw["battery"] += 0.5
            raw["sweeper"] += 0.3

        if q not in established:
            raw["anchor"] *= 0.75
            raw["battery"] *= 0.8
        if q not in role_ready:
            raw["shield"] *= 0.7
            raw["battery"] *= 0.8
        if anchor_planet_id is not None and int(planet.id) == int(anchor_planet_id):
            raw["anchor"] += 2.0
        if feeder_planet_id is not None and int(planet.id) == int(feeder_planet_id):
            raw["feeder"] += 1.8
        if attacker_planet_id is not None and int(planet.id) == int(attacker_planet_id):
            raw["attacker"] += 1.8

        scores = _normalize_role_scores(raw)
        scores["quadrant"] = quadrant_name(q)
        scores["pressure"] = context["pressure"]
        profiles[int(planet.id)] = scores
    return profiles


def pressure_profile(planet, planets, player):
    enemies = [p for p in planets if p.owner not in (-1, player)]
    if not enemies:
        return {"nearest_enemy_distance": 100.0, "same_quadrant_enemy_ships": 0, "safety": 1.0}

    nearest = min(distance(planet, enemy) for enemy in enemies)
    same_quadrant_ships = sum(
        enemy.ships for enemy in enemies if quadrant_index(enemy) == quadrant_index(planet)
    )
    nearest_safety = clamp(nearest / 65.0)
    local_safety = clamp(1.0 - same_quadrant_ships / 260.0)
    safety = clamp(0.55 * nearest_safety + 0.45 * local_safety)
    return {
        "nearest_enemy_distance": round(nearest, 3),
        "same_quadrant_enemy_ships": int(same_quadrant_ships),
        "safety": round(safety, 4),
    }


def role_scores(obs, player=None, top_n=5):
    planets = planets_from_obs(obs)
    fleets = fleets_from_obs(obs)
    player = player_from_obs(obs) if player is None else int(player)
    established = established_quadrants(planets, player)
    owned = [planet for planet in planets if int(planet.owner) == player]
    profiles = role_confidence_map_from_state(planets, fleets, player)
    return _role_summary_from_profiles(owned, profiles, established, top_n=top_n)


def build_quadrant_array(obs, player_count=None):
    planets = planets_from_obs(obs)
    fleets = fleets_from_obs(obs)
    max_owner = max([p.owner for p in planets if p.owner >= 0] + [f.owner for f in fleets if f.owner >= 0] + [0])
    player_count = max(player_count or (max_owner + 1), max_owner + 1)
    rows = []

    for q in range(4):
        row = {
            "quadrant": quadrant_name(q),
            "neutral": {
                "ships": 0,
                "production": 0,
                "big_static": 0,
                "small_static": 0,
                "rotating_ships": 0,
            },
            "players": {
                str(owner): {
                    "ships": 0,
                    "production": 0,
                    "big_static": 0,
                    "small_static": 0,
                    "rotating_ships": 0,
                    "fleet_ships": 0,
                    "fleet_count": 0,
                    "established": False,
                }
                for owner in range(player_count)
            },
        }
        rows.append(row)

    for planet in planets:
        q = quadrant_index(planet)
        target = rows[q]["neutral"] if planet.owner == -1 else rows[q]["players"].setdefault(str(planet.owner), {})
        target["ships"] = target.get("ships", 0) + planet.ships
        target["production"] = target.get("production", 0) + planet.production
        if is_static(planet):
            key = "big_static" if size_label(planet) == "big" else "small_static"
            target[key] = target.get(key, 0) + 1
        else:
            target["rotating_ships"] = target.get("rotating_ships", 0) + planet.ships

    for fleet in fleets:
        q = quadrant_index_xy(fleet.x, fleet.y)
        player_row = rows[q]["players"].setdefault(str(fleet.owner), {})
        player_row["fleet_ships"] = player_row.get("fleet_ships", 0) + fleet.ships
        player_row["fleet_count"] = player_row.get("fleet_count", 0) + 1

    for owner in range(player_count):
        established = established_quadrants(planets, owner)
        for q in established:
            rows[q]["players"][str(owner)]["established"] = True

    return rows


def numeric_quadrant_array(obs, player=None, player_count=None):
    player = player_from_obs(obs) if player is None else int(player)
    rows = build_quadrant_array(obs, player_count=player_count)
    numeric_rows = []
    for row in rows:
        our = row["players"].get(str(player), {})
        enemy_players = [values for owner, values in row["players"].items() if int(owner) != player]
        enemy_ships = sum(values.get("ships", 0) for values in enemy_players)
        enemy_prod = sum(values.get("production", 0) for values in enemy_players)
        enemy_fleet_ships = sum(values.get("fleet_ships", 0) for values in enemy_players)
        enemy_established = sum(1 for values in enemy_players if values.get("established"))
        neutral = row["neutral"]
        numeric_rows.append(
            [
                our.get("ships", 0),
                our.get("production", 0),
                our.get("big_static", 0),
                our.get("small_static", 0),
                our.get("rotating_ships", 0),
                our.get("fleet_ships", 0),
                1 if our.get("established") else 0,
                enemy_ships,
                enemy_prod,
                enemy_fleet_ships,
                enemy_established,
                neutral.get("ships", 0),
                neutral.get("production", 0),
                neutral.get("big_static", 0),
                neutral.get("small_static", 0),
                neutral.get("rotating_ships", 0),
            ]
        )
    return numeric_rows


def _ray_circle_hit(start_x, start_y, angle, cx, cy, radius):
    dx = math.cos(angle)
    dy = math.sin(angle)
    fx = start_x - cx
    fy = start_y - cy
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0:
        return None
    root = math.sqrt(disc)
    first = (-b - root) / 2.0
    second = (-b + root) / 2.0
    hits = [value for value in (first, second) if value >= 0]
    return min(hits) if hits else None


def _angle_delta(first, second):
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def infer_action_target(action, planets):
    if not action or len(action) < 3:
        return None
    source_id = int(action[0])
    angle = float(action[1])
    by_id = {planet.id: planet for planet in planets}
    source = by_id.get(source_id)
    if source is None:
        return None

    start_x = source.x + math.cos(angle) * (source.radius + LAUNCH_CLEARANCE)
    start_y = source.y + math.sin(angle) * (source.radius + LAUNCH_CLEARANCE)
    best = None
    fallback = None
    for planet in planets:
        if planet.id == source_id:
            continue
        hit = _ray_circle_hit(start_x, start_y, angle, planet.x, planet.y, planet.radius)
        if hit is None:
            to_x = planet.x - start_x
            to_y = planet.y - start_y
            along = to_x * math.cos(angle) + to_y * math.sin(angle)
            if along <= 0:
                continue
            aim_angle = math.atan2(to_y, to_x)
            angle_gap = _angle_delta(angle, aim_angle)
            cross_track = abs(to_x * math.sin(angle) - to_y * math.cos(angle))
            if angle_gap > 0.28 or cross_track > max(planet.radius + 9.0, 12.0):
                continue
            score = angle_gap * 25.0 + cross_track / max(planet.radius, 1.0) + along * 0.002
            if fallback is None or score < fallback[0]:
                fallback = (score, planet)
            continue
        if best is None or hit < best[0]:
            best = (hit, planet)
    if best:
        return best[1]
    return fallback[1] if fallback else None


def _fleet_target_estimate(fleet, planets):
    best = None
    fallback = None
    for planet in planets:
        if int(planet.id) == int(fleet.from_planet_id):
            continue
        hit = _ray_circle_hit(
            float(fleet.x),
            float(fleet.y),
            float(fleet.angle),
            float(planet.x),
            float(planet.y),
            float(planet.radius),
        )
        if hit is not None:
            if best is None or hit < best[0]:
                best = (hit, planet)
            continue

        to_x = float(planet.x) - float(fleet.x)
        to_y = float(planet.y) - float(fleet.y)
        along = to_x * math.cos(float(fleet.angle)) + to_y * math.sin(float(fleet.angle))
        if along <= 0.0:
            continue
        aim_angle = math.atan2(to_y, to_x)
        angle_gap = _angle_delta(float(fleet.angle), aim_angle)
        cross_track = abs(
            to_x * math.sin(float(fleet.angle)) - to_y * math.cos(float(fleet.angle))
        )
        if angle_gap > 0.22 or cross_track > max(float(planet.radius) + 7.0, 10.0):
            continue
        score = angle_gap * 25.0 + cross_track / max(float(planet.radius), 1.0)
        if fallback is None or score < fallback[0]:
            fallback = (score, along, planet)

    if best is not None:
        travel_distance, target = best
    elif fallback is not None:
        _, along, target = fallback
        travel_distance = max(0.0, along - float(target.radius))
    else:
        return None, None
    eta = travel_distance / max(1.0, fleet_speed(int(fleet.ships)))
    return target, max(1.0, eta)


def concentrated_pressure_profile(planets, fleets, player):
    player = int(player)
    owned = [planet for planet in planets if int(planet.owner) == player]
    empty = {
        "flagged": False,
        "reason": "none",
        "target_id": None,
        "quadrant": None,
        "source_ids": (),
        "source_count": 0,
        "fleet_count": 0,
        "hostile_ships": 0,
        "projected_defense": 0.0,
        "defense_ratio": 0.0,
        "first_eta": 0.0,
        "last_eta": 0.0,
        "eta_spread": 0.0,
    }
    if not owned:
        return empty

    hostile_by_target = defaultdict(list)
    friendly_by_target = defaultdict(list)
    for fleet in fleets:
        if int(fleet.owner) < 0:
            continue
        target, eta = _fleet_target_estimate(fleet, planets)
        if target is None or int(target.owner) != player:
            continue
        row = (float(eta), int(fleet.from_planet_id), int(fleet.ships), int(fleet.owner))
        if int(fleet.owner) == player:
            friendly_by_target[int(target.id)].append(row)
        else:
            hostile_by_target[int(target.id)].append(row)

    strongest = None
    for target in owned:
        arrivals = sorted(hostile_by_target.get(int(target.id), []))
        for start in range(len(arrivals)):
            window = [
                row
                for row in arrivals[start:]
                if float(row[0]) - float(arrivals[start][0]) <= PRESSURE_ARRIVAL_WINDOW
            ]
            source_ids = {row[1] for row in window if row[1] >= 0}
            if len(source_ids) < PRESSURE_MIN_SOURCE_COUNT:
                continue

            first_eta = float(window[0][0])
            last_eta = float(window[-1][0])
            hostile_ships = sum(int(row[2]) for row in window)
            friendly_ships = sum(
                int(row[2])
                for row in friendly_by_target.get(int(target.id), [])
                if float(row[0]) <= last_eta
            )
            projected_defense = (
                float(target.ships)
                + float(target.production) * math.ceil(last_eta)
                + float(friendly_ships)
            )
            defense_ratio = float(hostile_ships) / max(1.0, projected_defense)
            meaningful_mass = hostile_ships >= max(
                PRESSURE_MIN_HOSTILE_SHIPS,
                int(math.ceil(projected_defense * PRESSURE_MIN_DEFENSE_RATIO)),
            )
            if not meaningful_mass:
                continue

            profile = {
                "flagged": True,
                "reason": "concentrated_inbound",
                "target_id": int(target.id),
                "quadrant": quadrant_name(quadrant_index(target)),
                "source_ids": tuple(sorted(source_ids)),
                "source_count": len(source_ids),
                "fleet_count": len(window),
                "hostile_ships": int(hostile_ships),
                "projected_defense": round(projected_defense, 3),
                "defense_ratio": round(defense_ratio, 4),
                "first_eta": round(first_eta, 3),
                "last_eta": round(last_eta, 3),
                "eta_spread": round(last_eta - first_eta, 3),
            }
            score = (
                float(profile["source_count"]),
                min(2.0, defense_ratio),
                float(target.production),
                float(hostile_ships),
            )
            if strongest is None or score > strongest[0]:
                strongest = (score, profile)

    return strongest[1] if strongest is not None else empty


def pressure_conditioned_action_profile(
    planets,
    fleets,
    player,
    source,
    target,
    ships,
    action_eta=None,
    pressure_profile=None,
):
    pressure = pressure_profile or concentrated_pressure_profile(planets, fleets, player)
    empty = {
        "pressure_direct_reinforcement": 0.0,
        "pressure_arrival_slack": 0.0,
        "pressure_need_coverage": 0.0,
        "pressure_source_drain": 0.0,
        "pressure_counter_source": 0.0,
    }
    if not pressure["flagged"] or source is None or target is None or int(ships) <= 0:
        return empty

    pressure_target_id = int(pressure["target_id"])
    direct_reinforcement = int(target.id) == pressure_target_id
    source_drain = int(source.id) == pressure_target_id
    counter_source = int(target.id) in set(pressure.get("source_ids", ()))

    if action_eta is None:
        travel_distance = max(
            0.0,
            distance(source, target) - float(source.radius) - float(target.radius),
        )
        action_eta = travel_distance / max(1.0, fleet_speed(int(ships)))

    arrival_slack = 0.0
    need_coverage = 0.0
    if direct_reinforcement:
        final_pressure_eta = max(1.0, float(pressure["last_eta"]))
        arrival_slack = clamp(
            (final_pressure_eta - float(action_eta) + 1.0) / final_pressure_eta
        )
        projected_gap = max(
            1.0,
            float(pressure["hostile_ships"])
            - float(pressure["projected_defense"])
            + 1.0,
            math.ceil(float(pressure["hostile_ships"]) * 0.15),
        )
        need_coverage = clamp(float(ships) / projected_gap)

    return {
        "pressure_direct_reinforcement": 1.0 if direct_reinforcement else 0.0,
        "pressure_arrival_slack": arrival_slack,
        "pressure_need_coverage": need_coverage,
        "pressure_source_drain": 1.0 if source_drain else 0.0,
        "pressure_counter_source": 1.0 if counter_source else 0.0,
    }


def action_features(action, obs, player=None):
    planets = planets_from_obs(obs)
    player = player_from_obs(obs) if player is None else int(player)
    source = next((planet for planet in planets if planet.id == int(action[0])), None) if action else None
    target = infer_action_target(action, planets)
    ships = int(action[2]) if action and len(action) >= 3 else 0

    if target is None:
        target_owner = "unknown"
        target_label = None
    elif target.owner == -1:
        target_owner = "neutral"
        target_label = planet_label(target)
    elif target.owner == player:
        target_owner = "friendly"
        target_label = planet_label(target)
    else:
        target_owner = "enemy"
        target_label = planet_label(target)

    source_label = planet_label(source) if source is not None else None
    target_kind = "unknown"
    target_size = "unknown"
    target_quadrant = "unknown"
    target_ships = None
    if target is not None:
        target_kind = "static" if is_static(target) else "rotating"
        target_size = size_label(target)
        target_quadrant = quadrant_name(quadrant_index(target))
        target_ships = target.ships

    return {
        "source_id": int(action[0]) if action else None,
        "ships": ships,
        "angle": float(action[1]) if action and len(action) >= 2 else None,
        "source": source_label,
        "target": target_label,
        "target_id": target.id if target is not None else None,
        "target_owner_group": target_owner,
        "target_kind": target_kind,
        "target_size": target_size,
        "target_quadrant": target_quadrant,
        "target_ships": target_ships,
        "central_rotating_big": bool(target is not None and is_rotating(target) and target.production == 5),
        "central_rotating_small": bool(target is not None and is_rotating(target) and target.production < 5),
    }


def capture_events(before_obs, after_obs):
    before = {planet.id: planet for planet in planets_from_obs(before_obs)}
    after = {planet.id: planet for planet in planets_from_obs(after_obs)}
    events = []
    for planet_id, new_planet in after.items():
        old_planet = before.get(planet_id)
        if old_planet is None or old_planet.owner == new_planet.owner:
            continue
        events.append(
            {
                "planet_id": planet_id,
                "from_owner": old_planet.owner,
                "to_owner": new_planet.owner,
                "ships_before": old_planet.ships,
                "ships_after": new_planet.ships,
                "production": new_planet.production,
                "quadrant": quadrant_name(quadrant_index(new_planet)),
                "static": is_static(new_planet),
                "size": size_label(new_planet),
            }
        )
    return events


def empty_counter_map():
    return defaultdict(Counter)


def player_count_from_state(planets, fleets, player=None):
    owners = []
    if player is not None:
        owners.append(int(player))
    owners.extend(int(planet.owner) for planet in planets if int(planet.owner) >= 0)
    owners.extend(int(fleet.owner) for fleet in fleets if int(fleet.owner) >= 0)
    return max(2, max(owners, default=0) + 1)


def obs_from_state(planets, fleets, player=0, step=0, angular_velocity=0.0):
    return {
        "player": int(player),
        "step": int(step),
        "angular_velocity": float(angular_velocity),
        "planets": [
            [int(planet.id), int(planet.owner), float(planet.x), float(planet.y), float(planet.radius), int(planet.ships), int(planet.production)]
            for planet in planets
        ],
        "fleets": [
            [int(fleet.id), int(fleet.owner), float(fleet.x), float(fleet.y), float(fleet.angle), int(fleet.from_planet_id), int(fleet.ships)]
            for fleet in fleets
        ],
        "comet_planet_ids": [],
    }


def quadrant_totals(planets, fleets, owner, quadrant):
    ours = 0
    enemy = 0
    quadrant = int(quadrant)
    owner = int(owner)
    for planet in planets:
        if quadrant_index(planet) != quadrant:
            continue
        weight = int(planet.ships) + 3 * int(planet.production)
        if int(planet.owner) == owner:
            ours += weight
        elif int(planet.owner) not in (-1, owner):
            enemy += weight

    for fleet in fleets:
        if quadrant_index_xy(float(fleet.x), float(fleet.y)) != quadrant:
            continue
        if int(fleet.owner) == owner:
            ours += int(fleet.ships)
        elif int(fleet.owner) >= 0 and int(fleet.owner) != owner:
            enemy += int(fleet.ships)
    return ours, enemy


def quadrant_control_margin(planets, fleets, owner, quadrant):
    ours, enemy = quadrant_totals(planets, fleets, owner, quadrant)
    return ours - enemy


def adjacent_quadrants(quadrant):
    quadrant = int(quadrant)
    return ((quadrant - 1) % 4, (quadrant + 1) % 4)


def half_control_margin(planets, fleets, owner, quadrants):
    return sum(quadrant_control_margin(planets, fleets, owner, quadrant) for quadrant in quadrants)


def best_anchor_quadrant(planets, fleets, owner):
    owner = int(owner)
    best = None
    best_score = None
    for quadrant in range(4):
        anchor = anchor_corner_planet(planets, quadrant)
        if anchor is None or int(anchor.owner) != owner:
            continue
        local_margin = quadrant_control_margin(planets, fleets, owner, quadrant)
        half_margin = max(
            half_control_margin(planets, fleets, owner, (quadrant, adjacent))
            for adjacent in adjacent_quadrants(quadrant)
        )
        score = (
            half_margin,
            local_margin,
            int(anchor.production),
            int(anchor.ships),
            -corner_distance(anchor, quadrant),
        )
        if best_score is None or score > best_score:
            best = quadrant
            best_score = score
    return best


def current_control_half(planets, fleets, owner, anchor_quadrant=None):
    owner = int(owner)
    if anchor_quadrant is None:
        anchor_quadrant = best_anchor_quadrant(planets, fleets, owner)
    if anchor_quadrant is None:
        return None
    best_half = None
    best_score = None
    for adjacent in adjacent_quadrants(anchor_quadrant):
        score = (
            half_control_margin(planets, fleets, owner, (anchor_quadrant, adjacent)),
            quadrant_control_margin(planets, fleets, owner, adjacent),
        )
        if best_score is None or score > best_score:
            best_half = (int(anchor_quadrant), int(adjacent))
            best_score = score
    return best_half


def owner_control_scores(planets, fleets, player_count=None):
    player_count = player_count_from_state(planets, fleets) if player_count is None else int(player_count)
    scores = {owner: 0.0 for owner in range(player_count)}
    for planet in planets:
        owner = int(planet.owner)
        if owner < 0:
            continue
        value = float(planet.ships) + 14.0 * float(planet.production)
        if is_static(planet):
            value += 5.0
        if is_corner_node(planet):
            value += 6.0
        if size_label(planet) == "big":
            value += 4.0
        scores[owner] = scores.get(owner, 0.0) + value

    for fleet in fleets:
        owner = int(fleet.owner)
        if owner < 0:
            continue
        scores[owner] = scores.get(owner, 0.0) + 0.7 * float(fleet.ships)
    return scores


def _target_control_swing(target):
    if target is None:
        return 0.0
    value = 0.30 * float(target.ships) + 18.0 * float(target.production)
    if is_static(target):
        value += 6.0
    if is_corner_node(target):
        value += 8.0
    if size_label(target) == "big":
        value += 5.0
    return value


def overtake_profile_for_target(planets, fleets, player, target, owner_scores=None, player_count=None):
    player = int(player)
    if target is None:
        return {
            "our_score_norm": 0.0,
            "leader_gap_norm": 0.0,
            "nearest_ahead_gap_norm": 0.0,
            "projected_gain_norm": 0.0,
            "leader_gap_improvement": 0.0,
            "nearest_ahead_improvement": 0.0,
            "projected_overtake_count": 0.0,
            "leader_target": 0.0,
            "ahead_owner_target": 0.0,
            "quadrant_flip_pressure": 0.0,
            "board_ownership_bonus": 0.0,
            "overtake_bonus": 0.0,
        }

    player_count = player_count_from_state(planets, fleets, player) if player_count is None else int(player_count)
    owner_scores = owner_control_scores(planets, fleets, player_count=player_count) if owner_scores is None else owner_scores
    our_score = float(owner_scores.get(player, 0.0))
    opponents = [(owner, score) for owner, score in owner_scores.items() if int(owner) != player]
    if opponents:
        leader_owner, leader_score = max(opponents, key=lambda item: item[1])
    else:
        leader_owner, leader_score = player, our_score
    ahead = sorted((score for owner, score in opponents if score > our_score))
    nearest_ahead_gap = ahead[0] - our_score if ahead else 0.0
    leader_gap = max(0.0, float(leader_score) - our_score)

    projected_gain = _target_control_swing(target)
    deny_value = 0.0
    target_owner = int(target.owner)
    if target_owner >= 0 and target_owner != player:
        deny_value = 0.85 * projected_gain

    projected_scores = dict(owner_scores)
    projected_scores[player] = projected_scores.get(player, 0.0) + projected_gain
    if target_owner >= 0 and target_owner != player:
        projected_scores[target_owner] = max(0.0, projected_scores.get(target_owner, 0.0) - deny_value)

    projected_our = float(projected_scores.get(player, 0.0))
    ahead_after = sorted((score for owner, score in projected_scores.items() if int(owner) != player and score > projected_our))
    nearest_ahead_gap_after = ahead_after[0] - projected_our if ahead_after else 0.0
    leader_after = max((score for owner, score in projected_scores.items() if int(owner) != player), default=projected_our)
    leader_gap_after = max(0.0, float(leader_after) - projected_our)
    overtaken_count = sum(
        1
        for owner, score in owner_scores.items()
        if int(owner) != player and score > our_score and projected_our >= float(projected_scores.get(owner, score))
    )

    target_quadrant = quadrant_index(target)
    quadrant_margin = quadrant_control_margin(planets, fleets, player, target_quadrant)
    quadrant_flip_pressure = clamp(max(0.0, -float(quadrant_margin)) / 900.0)
    leader_target = 1.0 if target_owner >= 0 and target_owner == int(leader_owner) and leader_owner != player else 0.0
    ahead_owner_target = 1.0 if target_owner >= 0 and target_owner != player and float(owner_scores.get(target_owner, 0.0)) > our_score else 0.0
    board_ownership_bonus = clamp(
        0.42 * min(1.0, projected_gain / 150.0)
        + 0.34 * quadrant_flip_pressure
        + 0.14 * (1.0 if target_owner == -1 else 0.0)
        + 0.10 * (1.0 if is_static(target) else 0.0)
    )
    overtake_bonus = clamp(
        0.20 * min(1.0, projected_gain / 150.0)
        + 0.22 * min(1.0, max(0.0, leader_gap - leader_gap_after) / 280.0)
        + 0.20 * min(1.0, max(0.0, nearest_ahead_gap - nearest_ahead_gap_after) / 220.0)
        + 0.18 * (float(overtaken_count) / max(1.0, float(player_count - 1)))
        + 0.10 * leader_target
        + 0.10 * ahead_owner_target
        + 0.12 * quadrant_flip_pressure
        + 0.10 * board_ownership_bonus
    )

    return {
        "our_score_norm": clamp(our_score / 1200.0),
        "leader_gap_norm": clamp(leader_gap / 900.0),
        "nearest_ahead_gap_norm": clamp(nearest_ahead_gap / 700.0),
        "projected_gain_norm": clamp(projected_gain / 180.0),
        "leader_gap_improvement": clamp(max(0.0, leader_gap - leader_gap_after) / 320.0),
        "nearest_ahead_improvement": clamp(max(0.0, nearest_ahead_gap - nearest_ahead_gap_after) / 260.0),
        "projected_overtake_count": clamp(float(overtaken_count) / max(1.0, float(player_count - 1))),
        "leader_target": leader_target,
        "ahead_owner_target": ahead_owner_target,
        "quadrant_flip_pressure": quadrant_flip_pressure,
        "board_ownership_bonus": board_ownership_bonus,
        "overtake_bonus": overtake_bonus,
    }


def trend_identity_details_for_target(
    planets,
    fleets,
    player,
    target,
    owner_scores=None,
    player_count=None,
    tendency=None,
):
    player = int(player)
    overtake_profile = overtake_profile_for_target(
        planets,
        fleets,
        player,
        target,
        owner_scores=owner_scores,
        player_count=player_count,
    )
    tendency = tendency or {}
    captures = float(tendency.get("captures", 0))
    losses = float(tendency.get("losses", 0))
    launches = float(tendency.get("launches", 0))
    aggression = captures + launches
    pressure = losses - captures
    concentrated = concentrated_pressure_profile(planets, fleets, player)
    if concentrated["flagged"]:
        return {
            **concentrated,
            "identity": "pressured",
            "reason": concentrated["reason"],
        }
    if float(overtake_profile["overtake_bonus"]) >= 0.55:
        return {**concentrated, "identity": "overtake_window", "reason": "overtake_window"}
    if pressure >= 2.0:
        return {**concentrated, "identity": "pressured", "reason": "loss_trend"}
    if aggression >= 6.0 and float(overtake_profile["board_ownership_bonus"]) >= 0.35:
        return {**concentrated, "identity": "cash_in", "reason": "cash_in"}
    if float(overtake_profile["leader_target"]) > 0.0 or float(overtake_profile["ahead_owner_target"]) > 0.0:
        return {**concentrated, "identity": "chasing_leader", "reason": "leader_target"}
    return {**concentrated, "identity": "neutral", "reason": "neutral"}


def trend_identity_for_target(planets, fleets, player, target, owner_scores=None, player_count=None, tendency=None):
    return trend_identity_details_for_target(
        planets,
        fleets,
        player,
        target,
        owner_scores=owner_scores,
        player_count=player_count,
        tendency=tendency,
    )["identity"]


def infer_role_assignments_from_state(
    planets,
    fleets,
    player,
    anchor_planet_id=None,
    attacker_planet_id=None,
    feeder_planet_id=None,
    phase_name=None,
    role_profiles=None,
):
    player = int(player)
    owned = [planet for planet in planets if int(planet.owner) == player]
    if not owned:
        return {}

    phase_name = phase_name or infer_phase_from_state(planets, fleets, player)
    profiles = role_profiles or role_confidence_map_from_state(
        planets,
        fleets,
        player,
        phase_name=phase_name,
        anchor_planet_id=anchor_planet_id,
        attacker_planet_id=attacker_planet_id,
        feeder_planet_id=feeder_planet_id,
    )
    assignments = {}
    established = established_quadrants(planets, player)
    role_ready = role_ready_quadrants(planets, player)
    anchor = None
    anchor_candidates = [planet for planet in owned if is_static(planet)]
    if anchor_candidates:
        if anchor_planet_id is not None:
            anchor = next((planet for planet in anchor_candidates if int(planet.id) == int(anchor_planet_id)), None)
        if anchor is None:
            anchor = max(
                anchor_candidates,
                key=lambda planet: (
                    float(profiles.get(int(planet.id), {}).get("anchor", 0.0)),
                    int(is_corner_node(planet)),
                    int(size_label(planet) == "big"),
                    int(planet.production),
                    int(planet.ships),
                    -int(planet.id),
                ),
            )
        assignments[int(anchor.id)] = "anchor"

    for quadrant in sorted(role_ready):
        bigs, smalls = owned_corner_groups(planets, player, quadrant)
        bigs = sorted(
            bigs,
            key=lambda planet: (
                -float(profiles.get(int(planet.id), {}).get("battery", 0.0)),
                -int(planet.ships),
                int(planet.id),
            ),
        )
        smalls = sorted(
            smalls,
            key=lambda planet: (
                -float(profiles.get(int(planet.id), {}).get("sweeper", 0.0)),
                corner_distance(planet, quadrant),
                int(planet.id),
            ),
        )
        if bigs:
            assignments.setdefault(int(bigs[0].id), "battery")
        if smalls:
            assignments.setdefault(int(smalls[0].id), "sweeper")
        if len(smalls) > 1:
            shield = max(
                [planet for planet in smalls if int(planet.id) != int(smalls[0].id)],
                key=lambda planet: (
                    float(profiles.get(int(planet.id), {}).get("shield", 0.0)),
                    int(planet.ships),
                    -corner_distance(planet, quadrant),
                    -int(planet.id),
                ),
            )
            assignments[int(shield.id)] = "shield"

    if established:
        candidates = [
            planet
            for planet in owned
            if is_static(planet)
            and quadrant_index(planet) in established
            and (anchor is None or int(planet.id) != int(anchor.id))
        ]
        if candidates:
            feeder = None
            if feeder_planet_id is not None:
                feeder = next((planet for planet in candidates if int(planet.id) == int(feeder_planet_id)), None)
            if feeder is None:
                feeder = max(
                    candidates,
                    key=lambda planet: (
                        float(profiles.get(int(planet.id), {}).get("feeder", 0.0)),
                        int(planet.ships),
                        int(planet.production),
                        -int(planet.id),
                    ),
                )
            if feeder is not None:
                assignments[int(feeder.id)] = "feeder"

    if established:
        rotating = [planet for planet in owned if is_rotating(planet) and int(planet.id) not in assignments]
        if rotating:
            attacker = None
            if attacker_planet_id is not None:
                attacker = next((planet for planet in rotating if int(planet.id) == int(attacker_planet_id)), None)
            if attacker is None:
                attacker = max(
                    rotating,
                    key=lambda planet: (
                        float(profiles.get(int(planet.id), {}).get("attacker", 0.0)),
                        int(planet.ships),
                        int(planet.production),
                        -int(planet.id),
                    ),
                )
            if attacker is not None:
                assignments[int(attacker.id)] = "attacker"

    for planet in owned:
        if int(planet.id) in assignments:
            continue
        fallback = profiles.get(int(planet.id), {}).get("primary_role", "expander")
        if fallback in ("anchor", "feeder", "attacker"):
            fallback = "expander"
        assignments[int(planet.id)] = fallback if fallback in ROLE_NAMES else "expander"
    return assignments


def infer_phase_from_state(planets, fleets, player, target=None, anchor_quadrant=None):
    player = int(player)
    owned = [planet for planet in planets if int(planet.owner) == player]
    if len(owned) < 2:
        return "initiation"

    established = established_quadrants(planets, player)
    if not established:
        return "expansion"

    if target is not None and int(target.owner) not in (-1, player):
        return "attack"

    control_half = current_control_half(planets, fleets, player, anchor_quadrant=anchor_quadrant)
    enemy_in_half = any(
        int(planet.owner) not in (-1, player)
        and (control_half is None or quadrant_index(planet) in control_half)
        for planet in planets
    )
    if enemy_in_half:
        return "attack"
    return "established"


def flatten_tendency_features(tendency):
    tendency = tendency or {}
    launches = max(1, int(tendency.get("launches", 0)))
    return [
        float(tendency.get("launches", 0)) / 400.0,
        float(tendency.get("ships_launched", 0)) / 20000.0,
        float(tendency.get("neutral_targets", 0)) / launches,
        float(tendency.get("enemy_targets", 0)) / launches,
        float(tendency.get("friendly_targets", 0)) / launches,
        float(tendency.get("static_targets", 0)) / launches,
        float(tendency.get("rotating_targets", 0)) / launches,
        float(tendency.get("central_rotating_big", 0)) / launches,
        float(tendency.get("central_rotating_small", 0)) / launches,
        float(tendency.get("captures", 0)) / 80.0,
        float(tendency.get("losses", 0)) / 80.0,
    ]


def _one_hot(name, names):
    return [1.0 if name == item else 0.0 for item in names]


def _safe_ratio(value, denom):
    if denom <= 0:
        return 0.0
    return float(value) / float(denom)


def _target_rank_sort_key(source, target):
    return (
        int(target.ships),
        int(target.owner) != -1,
        distance(source, target),
        -int(target.production),
        corner_distance(target),
        int(target.id),
    )


def _target_rank_metrics(source, target, candidates):
    if target is None or not candidates:
        return {
            "rank": 1.0,
            "best": 0.0,
            "ship_gap": 0.0,
        }

    ordered = sorted(candidates, key=lambda candidate: _target_rank_sort_key(source, candidate))
    rank_index = next((idx for idx, candidate in enumerate(ordered) if int(candidate.id) == int(target.id)), None)
    if rank_index is None:
        return {
            "rank": 1.0,
            "best": 0.0,
            "ship_gap": 0.0,
        }

    best_ships = int(ordered[0].ships)
    denom = max(1, len(ordered) - 1)
    return {
        "rank": float(rank_index) / float(denom),
        "best": 1.0 if rank_index == 0 else 0.0,
        "ship_gap": min(float(int(target.ships) - best_ships), ACTION_FEATURE_SCALES["ship_gap"])
        / ACTION_FEATURE_SCALES["ship_gap"],
    }


def target_rank_profile(planets, player, source, target, control_half=None):
    if source is None or target is None:
        return {
            "quadrant_any_rank": 1.0,
            "quadrant_any_best": 0.0,
            "quadrant_any_gap": 0.0,
            "quadrant_class_rank": 1.0,
            "quadrant_class_best": 0.0,
            "quadrant_class_gap": 0.0,
            "half_any_rank": 1.0,
            "half_any_best": 0.0,
            "half_any_gap": 0.0,
            "half_class_rank": 1.0,
            "half_class_best": 0.0,
            "half_class_gap": 0.0,
        }

    player = int(player)
    target_quadrant = quadrant_index(target)
    target_is_static = is_static(target)
    same_side_targets = [
        planet
        for planet in planets
        if int(planet.owner) != player and same_equator_side(source, planet)
    ]
    quadrant_targets = [planet for planet in same_side_targets if quadrant_index(planet) == target_quadrant]
    half_targets = [
        planet
        for planet in same_side_targets
        if control_half and quadrant_index(planet) in control_half
    ]
    if not half_targets:
        half_targets = list(same_side_targets)

    quadrant_class_targets = [planet for planet in quadrant_targets if is_static(planet) == target_is_static]
    half_class_targets = [planet for planet in half_targets if is_static(planet) == target_is_static]

    quadrant_any = _target_rank_metrics(source, target, quadrant_targets)
    quadrant_class = _target_rank_metrics(source, target, quadrant_class_targets or quadrant_targets)
    half_any = _target_rank_metrics(source, target, half_targets)
    half_class = _target_rank_metrics(source, target, half_class_targets or half_targets)
    return {
        "quadrant_any_rank": quadrant_any["rank"],
        "quadrant_any_best": quadrant_any["best"],
        "quadrant_any_gap": quadrant_any["ship_gap"],
        "quadrant_class_rank": quadrant_class["rank"],
        "quadrant_class_best": quadrant_class["best"],
        "quadrant_class_gap": quadrant_class["ship_gap"],
        "half_any_rank": half_any["rank"],
        "half_any_best": half_any["best"],
        "half_any_gap": half_any["ship_gap"],
        "half_class_rank": half_class["rank"],
        "half_class_best": half_class["best"],
        "half_class_gap": half_class["ship_gap"],
    }


def equator_side(planet):
    return 1 if float(planet.y) >= CENTER_Y else -1


def same_equator_side(first, second):
    return equator_side(first) == equator_side(second)


def fleet_speed(ships):
    ships = max(1.0, float(ships))
    if ships <= 1.0:
        return 1.0
    scale = max(0.0, min(1.0, math.log(ships) / math.log(1000.0)))
    return max(1.0, min(6.0, 1.0 + (6.0 - 1.0) * (scale ** 1.5)))


def launch_point(source, angle):
    offset = float(source.radius) + LAUNCH_CLEARANCE
    return (
        float(source.x) + math.cos(float(angle)) * offset,
        float(source.y) + math.sin(float(angle)) * offset,
    )


def ray_board_exit_point(sx, sy, angle):
    sx, sy = float(sx), float(sy)
    dx, dy = math.cos(float(angle)), math.sin(float(angle))
    distances = []
    if dx > 1e-9:
        distances.append((100.0 - sx) / dx)
    elif dx < -1e-9:
        distances.append((0.0 - sx) / dx)
    if dy > 1e-9:
        distances.append((100.0 - sy) / dy)
    elif dy < -1e-9:
        distances.append((0.0 - sy) / dy)
    positive = [value for value in distances if value >= 0.0]
    if not positive:
        return sx, sy
    travel = min(positive)
    return sx + dx * travel, sy + dy * travel


def segment_entry_distance_to_circle(sx, sy, tx, ty, cx, cy, radius):
    sx, sy = float(sx), float(sy)
    tx, ty = float(tx), float(ty)
    cx, cy = float(cx), float(cy)
    vx, vy = tx - sx, ty - sy
    length_sq = vx * vx + vy * vy
    if length_sq <= 0.0:
        return None

    length = math.sqrt(length_sq)
    projection_distance = ((cx - sx) * vx + (cy - sy) * vy) / length
    if projection_distance < 0.0 or projection_distance > length:
        return None

    closest_x = sx + (projection_distance / length) * vx
    closest_y = sy + (projection_distance / length) * vy
    miss = distance_xy(closest_x, closest_y, cx, cy)
    radius = float(radius)
    if miss > radius:
        return None

    entry = projection_distance - math.sqrt(max(0.0, radius * radius - miss * miss))
    if entry < 0.0:
        entry = 0.0
    if entry > length:
        return None
    return entry


def segment_min_distance_to_point(sx, sy, tx, ty, px, py):
    sx, sy = float(sx), float(sy)
    tx, ty = float(tx), float(ty)
    px, py = float(px), float(py)
    vx, vy = tx - sx, ty - sy
    length_sq = vx * vx + vy * vy
    if length_sq <= 0.0:
        return distance_xy(sx, sy, px, py)

    projection = ((px - sx) * vx + (py - sy) * vy) / length_sq
    projection = max(0.0, min(1.0, projection))
    closest_x = sx + projection * vx
    closest_y = sy + projection * vy
    return distance_xy(closest_x, closest_y, px, py)


def action_hits_sun(source, angle):
    if source is None:
        return False
    sx, sy = launch_point(source, angle)
    ex, ey = ray_board_exit_point(sx, sy, angle)
    return segment_entry_distance_to_circle(sx, sy, ex, ey, CENTER_X, CENTER_Y, SUN_DANGER_RADIUS) is not None


def action_sun_clearance(source, target, angle):
    if source is None or target is None or angle is None:
        return 100.0
    sx, sy = launch_point(source, angle)
    tx, ty = float(target.x), float(target.y)
    return segment_min_distance_to_point(sx, sy, tx, ty, CENTER_X, CENTER_Y)


def estimate_travel_eta(source, target, ships):
    if source is None or target is None or int(ships) <= 0:
        return 12.0
    speed = fleet_speed(ships)
    angle = math.atan2(float(target.y) - float(source.y), float(target.x) - float(source.x))
    lx, ly = launch_point(source, angle)
    travel_distance = max(0.0, distance_xy(lx, ly, float(target.x), float(target.y)) - float(target.radius))
    return travel_distance / max(1.0, speed)


def _candidate_capture_need(candidate):
    return max(1, int(candidate.ships) + 1)


def _candidate_opportunity_value(source, candidate, ships, player, phase_name, control_half, planets):
    eta = estimate_travel_eta(source, candidate, max(int(ships), int(candidate.ships) + 1))
    value = float(candidate.production)
    if size_label(candidate) == "big":
        value *= 2.0
    if is_static(candidate):
        value *= 1.35
    else:
        value *= 0.85
    if int(candidate.owner) == -1:
        value *= 1.1
    if phase_name in ("initiation", "expansion") and is_static(candidate):
        value *= 1.15
    if phase_name == "attack" and int(candidate.owner) not in (-1, int(player)):
        value *= 1.1
    if control_half and quadrant_index(candidate) in control_half:
        value *= 1.1
    if quadrant_index(candidate) == quadrant_index(source):
        value *= 1.05
    capture_need = _candidate_capture_need(candidate)
    capture_efficiency = 1.0 / (1.0 + float(capture_need) / 14.0)
    rank_profile = target_rank_profile(planets, player, source, candidate, control_half=control_half)
    value *= 0.45 + 1.55 * capture_efficiency
    value *= 1.0 + 0.20 * (1.0 - rank_profile["quadrant_any_rank"])
    value *= 1.0 + 0.18 * (1.0 - rank_profile["quadrant_class_rank"])
    value *= 1.0 + 0.10 * (1.0 - rank_profile["half_class_rank"])
    if rank_profile["quadrant_class_best"]:
        value *= 1.08
    if rank_profile["half_class_best"]:
        value *= 1.05
    airtime_decay = 1.0 / (1.0 + 0.12 * max(0.0, eta - 2.5))
    if is_static(source):
        airtime_decay *= 1.0 / (1.0 + 0.22 * max(0.0, eta - STATIC_LONG_FLIGHT_SECONDS))
        if quadrant_index(candidate) != quadrant_index(source):
            airtime_decay *= 0.94
        if int(candidate.owner) != int(player):
            sun_clearance = action_sun_clearance(
                source,
                candidate,
                math.atan2(float(candidate.y) - float(source.y), float(candidate.x) - float(source.x)),
            )
            if sun_clearance < SUN_CORRIDOR_RADIUS and eta > 3.0:
                airtime_decay *= 0.86
    value *= airtime_decay
    return value / (1.0 + eta)


def action_penalty_profile_for_state(
    planets,
    fleets,
    player,
    source,
    target,
    ships,
    source_role=None,
    phase_name=None,
    action_angle=None,
):
    if source is None or int(ships) <= 0:
        return {
            "sun_penalty": 0.0,
            "miss_penalty": MISS_PENALTY,
            "long_flight_penalty": 0.0,
            "opportunity_penalty": 0.0,
            "opportunity_cost": 0.0,
            "chosen_value": 0.0,
            "best_available_value": 0.0,
            "eta": 12.0,
            "sun_hit": False,
            "target_missing": True,
            "quality_score": 0.0,
            "total_penalty": MISS_PENALTY,
        }

    player = int(player)
    source_role = source_role or "unknown"
    phase_name = phase_name or infer_phase_from_state(planets, fleets, player, target=target)
    control_half = current_control_half(planets, fleets, player)
    if action_angle is None and target is not None:
        action_angle = math.atan2(float(target.y) - float(source.y), float(target.x) - float(source.x))

    sun_hit = action_hits_sun(source, action_angle) if action_angle is not None else False
    target_missing = target is None
    eta = estimate_travel_eta(source, target, ships) if target is not None else 12.0
    sun_clearance = action_sun_clearance(source, target, action_angle) if target is not None else 100.0

    same_side_candidates = [
        planet
        for planet in planets
        if int(planet.owner) != player
        and same_equator_side(source, planet)
    ]
    if control_half:
        if source_role == "attacker":
            filtered_candidates = [planet for planet in same_side_candidates if quadrant_index(planet) in control_half]
            same_side_candidates = filtered_candidates or same_side_candidates
        else:
            filtered_candidates = [
                planet
                for planet in same_side_candidates
                if quadrant_index(planet) in control_half and (int(planet.owner) == -1 or quadrant_index(planet) == quadrant_index(source))
            ]
            same_side_candidates = filtered_candidates or same_side_candidates

    best_available_value = 0.0
    best_candidate = None
    for candidate in same_side_candidates:
        value = _candidate_opportunity_value(source, candidate, ships, player, phase_name, control_half, planets)
        if value > best_available_value:
            best_available_value = value
            best_candidate = candidate

    chosen_value = (
        0.0
        if target is None
        else _candidate_opportunity_value(source, target, ships, player, phase_name, control_half, planets)
    )
    opportunity_cost = max(0.0, best_available_value - chosen_value)
    opportunity_penalty = min(OPPORTUNITY_PENALTY_CAP, opportunity_cost / 6.0)
    if (
        best_candidate is not None
        and size_label(best_candidate) == "big"
        and is_static(best_candidate)
        and (target is None or not (size_label(target) == "big" and is_static(target)))
    ):
        opportunity_penalty = min(OPPORTUNITY_PENALTY_CAP, opportunity_penalty + 0.10)

    sun_penalty = SUN_PENALTY if sun_hit else 0.0
    miss_penalty = MISS_PENALTY if target_missing else 0.0
    long_flight_penalty = 0.0
    if eta > LONG_FLIGHT_SECONDS:
        long_flight_penalty = min(LONG_FLIGHT_PENALTY_CAP, 0.03 * (eta - LONG_FLIGHT_SECONDS))

    static_long_flight_penalty = 0.0
    static_remote_penalty = 0.0
    if target is not None and is_static(source):
        if eta > STATIC_LONG_FLIGHT_SECONDS:
            static_long_flight_penalty = min(
                STATIC_LONG_FLIGHT_PENALTY_CAP,
                0.055 * ((eta - STATIC_LONG_FLIGHT_SECONDS) ** 1.1),
            )

        if int(target.owner) != player:
            quadrant_gap = min(
                (quadrant_index(source) - quadrant_index(target)) % 4,
                (quadrant_index(target) - quadrant_index(source)) % 4,
            )
            if quadrant_gap >= 2 and eta > 3.0:
                static_remote_penalty += 0.12
            elif quadrant_gap == 1 and eta > LONG_FLIGHT_SECONDS:
                static_remote_penalty += 0.06

            if control_half and quadrant_index(target) not in control_half and eta > STATIC_LONG_FLIGHT_SECONDS:
                static_remote_penalty += 0.08

            if sun_clearance < SUN_CORRIDOR_RADIUS and eta > 3.0:
                static_remote_penalty += 0.10 + 0.015 * max(0.0, eta - 3.0)

        static_remote_penalty = min(STATIC_REMOTE_PENALTY_CAP, static_remote_penalty)

    total_penalty = clamp(
        sun_penalty
        + miss_penalty
        + long_flight_penalty
        + static_long_flight_penalty
        + static_remote_penalty
        + opportunity_penalty,
        0.0,
        1.0,
    )
    quality_score = clamp(1.0 - total_penalty, 0.0, 1.0)
    return {
        "sun_penalty": sun_penalty,
        "miss_penalty": miss_penalty,
        "long_flight_penalty": long_flight_penalty,
        "static_long_flight_penalty": static_long_flight_penalty,
        "static_remote_penalty": static_remote_penalty,
        "opportunity_penalty": opportunity_penalty,
        "opportunity_cost": clamp(opportunity_cost / ACTION_FEATURE_SCALES["opportunity_value"], 0.0, 1.0),
        "chosen_value": min(chosen_value, ACTION_FEATURE_SCALES["opportunity_value"]),
        "best_available_value": min(best_available_value, ACTION_FEATURE_SCALES["opportunity_value"]),
        "eta": eta,
        "sun_hit": sun_hit,
        "sun_clearance": sun_clearance,
        "target_missing": target_missing,
        "quality_score": quality_score,
        "total_penalty": total_penalty,
    }


def action_feature_vector_for_state(
    planets,
    fleets,
    player,
    source,
    target,
    ships,
    step=0,
    angular_velocity=0.0,
    tendency=None,
    source_role=None,
    phase_name=None,
    player_count=None,
    anchor_planet_id=None,
    attacker_planet_id=None,
    feeder_planet_id=None,
    action_angle=None,
    role_profiles=None,
    roles=None,
    role_summary=None,
    owner_scores=None,
    overtake_profile=None,
    trend_identity=None,
):
    if source is None or int(ships) <= 0:
        return []

    player = int(player)
    player_count = player_count_from_state(planets, fleets, player) if player_count is None else int(player_count)
    obs = obs_from_state(planets, fleets, player=player, step=step, angular_velocity=angular_velocity)
    quadrant_rows = numeric_quadrant_array(obs, player=player, player_count=player_count)
    flat_quadrants = []
    scales = [
        1200.0,
        100.0,
        4.0,
        8.0,
        1200.0,
        1200.0,
        1.0,
        1800.0,
        120.0,
        1800.0,
        4.0,
        1800.0,
        160.0,
        8.0,
        16.0,
        1800.0,
    ]
    for row in quadrant_rows:
        for idx, value in enumerate(row):
            flat_quadrants.append(float(value) / scales[idx])

    anchor_quadrant = best_anchor_quadrant(planets, fleets, player)
    phase_name = phase_name or infer_phase_from_state(planets, fleets, player, target=target, anchor_quadrant=anchor_quadrant)
    if role_profiles is None:
        role_profiles = role_confidence_map_from_state(
            planets,
            fleets,
            player,
            phase_name=phase_name,
            anchor_planet_id=anchor_planet_id,
            attacker_planet_id=attacker_planet_id,
            feeder_planet_id=feeder_planet_id,
        )
    if role_summary is None:
        role_summary = _role_summary_from_profiles(
            [planet for planet in planets if int(planet.owner) == player],
            role_profiles,
            established_quadrants(planets, player),
            top_n=1,
        )

    def first_role_score(key, score_name):
        items = role_summary.get(key, [])
        if not items:
            return 0.0
        return float(items[0].get(score_name, 0.0)) / 12.0

    role_features = [
        len(role_summary.get("established_quadrants", [])) / 4.0,
        first_role_score("anchor_candidates", "anchor_score"),
        first_role_score("feeder_candidates", "feeder_score"),
        first_role_score("sweeper_candidates", "sweeper_score"),
        first_role_score("strike_stage_candidates", "strike_stage_score"),
    ]

    if roles is None:
        roles = infer_role_assignments_from_state(
            planets,
            fleets,
            player,
            anchor_planet_id=anchor_planet_id,
            attacker_planet_id=attacker_planet_id,
            feeder_planet_id=feeder_planet_id,
            phase_name=phase_name,
            role_profiles=role_profiles,
        )
    source_role = source_role or roles.get(int(source.id), "unknown")
    source_profile = role_profiles.get(int(source.id), {})
    control_half = current_control_half(planets, fleets, player, anchor_quadrant=anchor_quadrant)
    penalty_profile = action_penalty_profile_for_state(
        planets,
        fleets,
        player,
        source,
        target,
        ships,
        source_role=source_role,
        phase_name=phase_name,
        action_angle=action_angle,
    )

    source_quadrant = quadrant_index(source)
    target_quadrant = quadrant_index(target) if target is not None else source_quadrant
    source_row = quadrant_rows[source_quadrant]
    target_row = quadrant_rows[target_quadrant]
    established = established_quadrants(planets, player)

    target_owner_group = "unknown"
    if target is not None:
        if int(target.owner) == -1:
            target_owner_group = "neutral"
        elif int(target.owner) == player:
            target_owner_group = "friendly"
        else:
            target_owner_group = "enemy"

    source_pressure = pressure_profile(source, planets, player)
    target_pressure = pressure_profile(target, planets, player) if target is not None else {"safety": 0.0}
    enemy_ships_total = sum(row[7] for row in quadrant_rows)
    enemy_prod_total = sum(row[8] for row in quadrant_rows)
    enemy_established = sum(row[10] for row in quadrant_rows) / 4.0
    distance_value = distance(source, target) if target is not None else 0.0
    need_gap = int(ships) - (int(target.ships) + 1 if target is not None else 0)
    target_owner_established = (
        1.0
        if target is not None and int(target.owner) >= 0 and target_quadrant in established_quadrants(planets, int(target.owner))
        else 0.0
    )
    target_open_quadrant = 1.0 if target_quadrant not in established else 0.0
    source_in_control_half = 1.0 if control_half and source_quadrant in control_half else 0.0
    target_in_control_half = 1.0 if control_half and target_quadrant in control_half else 0.0
    same_control_half = 1.0 if control_half and source_quadrant in control_half and target_quadrant in control_half else 0.0
    rank_profile = target_rank_profile(planets, player, source, target, control_half=control_half)
    if owner_scores is None:
        owner_scores = owner_control_scores(planets, fleets, player_count=player_count)
    if overtake_profile is None:
        overtake_profile = overtake_profile_for_target(
            planets,
            fleets,
            player,
            target,
            owner_scores=owner_scores,
            player_count=player_count,
        )
    if trend_identity is None:
        trend_identity = trend_identity_for_target(
            planets,
            fleets,
            player,
            target,
            owner_scores=owner_scores,
            player_count=player_count,
            tendency=tendency,
        )
    concentrated_pressure = concentrated_pressure_profile(planets, fleets, player)
    pressure_action = pressure_conditioned_action_profile(
        planets,
        fleets,
        player,
        source,
        target,
        ships,
        action_eta=penalty_profile["eta"],
        pressure_profile=concentrated_pressure,
    )

    features = [
        float(step) / 500.0,
        float(player_count) / 4.0,
        1.0,
        1.0 if player_count == 4 else 0.0,
    ]
    features.extend(flat_quadrants)
    features.extend(role_features)
    features.extend(flatten_tendency_features(tendency))
    features.extend(_one_hot(phase_name, PHASE_NAMES))
    features.extend(_one_hot(source_role, ROLE_NAMES))
    features.extend([float(source_profile.get(name, 0.0)) for name in ROLE_SCORE_NAMES])
    features.extend(
        [
            float(source_profile.get("primary_score", 0.0)),
            float(source_profile.get("role_certainty", 0.0)),
            float(source_profile.get(source_role, 0.0)),
        ]
    )
    features.extend(_one_hot(quadrant_name(source_quadrant), QUADRANT_NAMES))
    features.extend(_one_hot(quadrant_name(target_quadrant), QUADRANT_NAMES))
    features.extend(_one_hot(target_owner_group, ("neutral", "enemy", "friendly")))
    features.extend(
        [
            float(source.ships) / ACTION_FEATURE_SCALES["source_ships"],
            float(source.production) / ACTION_FEATURE_SCALES["source_prod"],
            1.0 if is_static(source) else 0.0,
            1.0 if is_rotating(source) else 0.0,
            1.0 if size_label(source) == "big" else 0.0,
            1.0 if source_quadrant in established else 0.0,
            float(quadrant_control_margin(planets, fleets, player, source_quadrant)) / ACTION_FEATURE_SCALES["source_margin"],
            float(source_pressure["safety"]) / ACTION_FEATURE_SCALES["source_safety"],
            float(corner_distance(source, source_quadrant)) / ACTION_FEATURE_SCALES["source_corner"],
        ]
    )
    features.extend(
        [
            0.0 if target is None else float(target.ships) / ACTION_FEATURE_SCALES["target_ships"],
            0.0 if target is None else float(target.production) / ACTION_FEATURE_SCALES["target_prod"],
            1.0 if target is not None and is_static(target) else 0.0,
            1.0 if target is not None and is_rotating(target) else 0.0,
            1.0 if target is not None and size_label(target) == "big" else 0.0,
            1.0 if target is not None and is_corner_node(target) else 0.0,
            target_owner_established,
            0.0
            if target is None
            else float(quadrant_control_margin(planets, fleets, player, target_quadrant)) / ACTION_FEATURE_SCALES["target_margin"],
            0.0 if target is None else float(target_pressure["safety"]) / ACTION_FEATURE_SCALES["target_safety"],
            0.0 if target is None else float(corner_distance(target, target_quadrant)) / ACTION_FEATURE_SCALES["target_corner"],
        ]
    )
    features.extend(
        [
            1.0 if target_quadrant == source_quadrant else 0.0,
            same_control_half,
            source_in_control_half,
            target_in_control_half,
            abs(int(source_quadrant) - int(target_quadrant)) / 3.0,
            float(distance_value) / ACTION_FEATURE_SCALES["distance"],
            float(ships) / ACTION_FEATURE_SCALES["ships_sent"],
            clamp(_safe_ratio(ships, max(1, int(source.ships)))),
            float(need_gap) / ACTION_FEATURE_SCALES["need_gap"],
            target_owner_established,
            target_open_quadrant,
            1.0 if target is not None and quadrant_index(target) == anchor_quadrant else 0.0,
        ]
    )
    features.extend(
        [
            float(rank_profile["quadrant_any_best"]),
            float(rank_profile["quadrant_class_best"]),
            float(rank_profile["half_any_best"]),
            float(rank_profile["half_class_best"]),
            float(rank_profile["quadrant_any_rank"]),
            float(rank_profile["quadrant_class_rank"]),
            float(rank_profile["half_any_rank"]),
            float(rank_profile["half_class_rank"]),
            float(rank_profile["quadrant_any_gap"]),
            float(rank_profile["quadrant_class_gap"]),
            float(rank_profile["half_any_gap"]),
            float(rank_profile["half_class_gap"]),
            float(overtake_profile["our_score_norm"]),
            float(overtake_profile["leader_gap_norm"]),
            float(overtake_profile["nearest_ahead_gap_norm"]),
            float(overtake_profile["projected_gain_norm"]),
            float(overtake_profile["leader_gap_improvement"]),
            float(overtake_profile["nearest_ahead_improvement"]),
            float(overtake_profile["projected_overtake_count"]),
            float(overtake_profile["leader_target"]),
            float(overtake_profile["ahead_owner_target"]),
            float(overtake_profile["quadrant_flip_pressure"]),
            float(overtake_profile["board_ownership_bonus"]),
            float(overtake_profile["overtake_bonus"]),
        ]
    )
    features.extend(_one_hot(trend_identity, TREND_NAMES))
    features.extend(
        [
            float(enemy_ships_total) / ACTION_FEATURE_SCALES["enemy_ships"],
            float(enemy_prod_total) / ACTION_FEATURE_SCALES["enemy_prod"],
            enemy_established,
            float(target_row[7]) / ACTION_FEATURE_SCALES["enemy_target_ships"],
            float(target_row[0]) / ACTION_FEATURE_SCALES["our_target_ships"],
            float(target_row[11]) / ACTION_FEATURE_SCALES["neutral_target_ships"],
            float(target_row[9]) / ACTION_FEATURE_SCALES["enemy_fleet_target"],
            float(target_row[5]) / ACTION_FEATURE_SCALES["our_fleet_target"],
        ]
    )
    features.extend(
        [
            1.0 if penalty_profile["sun_hit"] else 0.0,
            1.0 if penalty_profile["target_missing"] else 0.0,
            min(float(penalty_profile["eta"]), ACTION_FEATURE_SCALES["eta"]) / ACTION_FEATURE_SCALES["eta"],
            clamp(max(0.0, float(penalty_profile["eta"]) - LONG_FLIGHT_SECONDS) / ACTION_FEATURE_SCALES["eta"]),
            float(penalty_profile["chosen_value"]) / ACTION_FEATURE_SCALES["opportunity_value"],
            float(penalty_profile["best_available_value"]) / ACTION_FEATURE_SCALES["opportunity_value"],
            float(penalty_profile["opportunity_cost"]) / ACTION_FEATURE_SCALES["opportunity_cost"],
            float(penalty_profile["sun_penalty"]),
            float(penalty_profile["long_flight_penalty"]),
            float(penalty_profile["opportunity_penalty"]),
        ]
    )
    features.extend(
        [float(pressure_action[name]) for name in PRESSURE_ACTION_FEATURE_NAMES]
    )
    return features


def action_feature_vector(
    action,
    obs,
    player=None,
    tendency=None,
    source_role=None,
    phase_name=None,
    anchor_planet_id=None,
    attacker_planet_id=None,
    feeder_planet_id=None,
):
    planets = planets_from_obs(obs)
    fleets = fleets_from_obs(obs)
    player = player_from_obs(obs) if player is None else int(player)
    source = next((planet for planet in planets if planet.id == int(action[0])), None) if action else None
    target = infer_action_target(action, planets)
    ships = int(action[2]) if action and len(action) >= 3 else 0
    step = int(obs_get(obs, "step", 0) or 0)
    angular_velocity = float(obs_get(obs, "angular_velocity", 0.0) or 0.0)
    trend_identity = trend_identity_for_target(
        planets,
        fleets,
        player,
        target,
        tendency=tendency,
    )
    return action_feature_vector_for_state(
        planets,
        fleets,
        player,
        source,
        target,
        ships,
        step=step,
        angular_velocity=angular_velocity,
        tendency=tendency,
        source_role=source_role,
        phase_name=phase_name,
        player_count=player_count_from_state(planets, fleets, player),
        anchor_planet_id=anchor_planet_id,
        attacker_planet_id=attacker_planet_id,
        feeder_planet_id=feeder_planet_id,
        action_angle=float(action[1]) if action and len(action) >= 2 else None,
        trend_identity=trend_identity,
    )
