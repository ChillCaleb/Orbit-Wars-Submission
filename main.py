import math
from collections import namedtuple

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
except Exception:
    Planet = namedtuple("Planet", "id owner x y radius ships production")
    Fleet = namedtuple("Fleet", "id owner x y angle from_planet_id ships")

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, ROTATION_RADIUS_LIMIT
except Exception:
    CENTER = (50.0, 50.0)
    ROTATION_RADIUS_LIMIT = 30.0


NEUTRAL = -1
BOARD_SIZE = 100.0
CORNERS = ((0.0, 0.0), (0.0, BOARD_SIZE), (BOARD_SIZE, 0.0), (BOARD_SIZE, BOARD_SIZE))
QUADRANT_CORNERS = {
    0: (BOARD_SIZE, BOARD_SIZE),
    1: (0.0, BOARD_SIZE),
    2: (0.0, 0.0),
    3: (BOARD_SIZE, 0.0),
}
CORNER_CLUSTER_RADIUS = 45.0
SUN_DANGER_RADIUS = 5.0
_STATE = {}


def _obs_get(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def _center_xy():
    if hasattr(CENTER, "x") and hasattr(CENTER, "y"):
        return float(CENTER.x), float(CENTER.y)
    if isinstance(CENTER, (int, float)):
        return float(CENTER), float(CENTER)
    return float(CENTER[0]), float(CENTER[1])


def _coerce(row, cls):
    if hasattr(row, "id") and hasattr(row, "owner"):
        return row
    if isinstance(row, dict):
        return cls(*(row[field] for field in cls._fields))
    return cls(*row)


def _parse(obs):
    player = int(_obs_get(obs, "player", 0))
    comet_ids = set(_obs_get(obs, "comet_planet_ids", []) or [])
    raw_planets = _obs_get(obs, "planets", []) or []
    raw_fleets = _obs_get(obs, "fleets", []) or []
    angular_velocity = float(_obs_get(obs, "angular_velocity", 0.0) or 0.0)
    planets = [_coerce(row, Planet) for row in raw_planets]
    fleets = [_coerce(row, Fleet) for row in raw_fleets]
    planets = [p for p in planets if p.id not in comet_ids]
    return player, planets, fleets, angular_velocity, comet_ids


def _distance_xy(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _distance(a, b):
    return _distance_xy(float(a.x), float(a.y), float(b.x), float(b.y))


def _distance_to_center(p):
    cx, cy = _center_xy()
    return _distance_xy(float(p.x), float(p.y), cx, cy)


def _is_static(p):
    return _distance_to_center(p) + float(p.radius) >= float(ROTATION_RADIUS_LIMIT)


def _is_big(p):
    production = int(p.production)
    if production >= 5:
        return True
    if _is_static(p):
        return production >= 2
    return production >= 3


def _planet_angle(p):
    cx, cy = _center_xy()
    return math.atan2(float(p.y) - cy, float(p.x) - cx)


def _angle_quadrant(angle):
    normalized = float(angle) % (2.0 * math.pi)
    return int(normalized // (math.pi / 2.0)) % 4


def _quadrant(p):
    return _angle_quadrant(_planet_angle(p))


def _ahead_quadrant(quadrant, angular_velocity):
    if angular_velocity >= 0.0:
        return (int(quadrant) + 1) % 4
    return (int(quadrant) - 1) % 4


def _opposite_quadrant(quadrant):
    return (int(quadrant) + 2) % 4


def _quadrant_corner_xy(quadrant):
    return QUADRANT_CORNERS[int(quadrant) % 4]


def _distance_to_quadrant_corner(p, quadrant=None):
    if quadrant is None:
        quadrant = _quadrant(p)
    cx, cy = _quadrant_corner_xy(quadrant)
    return _distance_xy(float(p.x), float(p.y), cx, cy)


def _is_quadrant_corner_node(p, quadrant):
    return _is_static(p) and _quadrant(p) == int(quadrant) and _distance_to_quadrant_corner(p, quadrant) <= CORNER_CLUSTER_RADIUS


def _is_corner_node(p):
    return _is_quadrant_corner_node(p, _quadrant(p))


def _corner_nodes_for_quadrant(planets, quadrant):
    return [p for p in planets if _is_quadrant_corner_node(p, quadrant)]


def _quadrant_big_production(planets, quadrant):
    corner_nodes = _corner_nodes_for_quadrant(planets, quadrant)
    if not corner_nodes:
        return None
    return max(int(p.production) for p in corner_nodes)


def _is_quadrant_big(p, planets, quadrant):
    big_production = _quadrant_big_production(planets, quadrant)
    return big_production is not None and _is_quadrant_corner_node(p, quadrant) and int(p.production) == big_production


def _quadrant_corner_role_groups(planets, owner, quadrant):
    corner_nodes = _corner_nodes_for_quadrant(planets, quadrant)
    owned_corner = [p for p in corner_nodes if int(p.owner) == owner]
    big_production = _quadrant_big_production(planets, quadrant)
    if big_production is None:
        return [], []

    bigs = [p for p in owned_corner if int(p.production) == big_production]
    smalls = [p for p in owned_corner if int(p.production) < big_production]
    lower_production_nodes = [p for p in corner_nodes if int(p.production) < big_production]
    if len(lower_production_nodes) < 2 and bigs:
        anchor = max(bigs, key=lambda p: (int(p.production), int(p.ships), -_distance_to_quadrant_corner(p, quadrant)))
        bigs = [anchor]
        smalls = [p for p in owned_corner if p.id != anchor.id]
    return bigs, smalls


def _angle_to_xy(source, tx, ty):
    return math.atan2(float(ty) - float(source.y), float(tx) - float(source.x))


def _norm_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _fleet_speed(ships):
    return max(1.0, min(6.0, math.sqrt(max(1.0, float(ships)))))


def _available(source, reserved):
    return max(0, int(source.ships) - reserved.get(source.id, 0))


def _add_move(moves, reserved, source, angle, ships):
    amount = min(max(0, int(ships)), _available(source, reserved))
    if amount <= 0:
        return 0
    moves.append([source.id, _norm_angle(angle), amount])
    reserved[source.id] = reserved.get(source.id, 0) + amount
    return amount


def _point_after_orbit(p, turns, angular_velocity):
    if _is_static(p) or angular_velocity == 0.0:
        return float(p.x), float(p.y)
    cx, cy = _center_xy()
    radius = _distance_to_center(p)
    theta = math.atan2(float(p.y) - cy, float(p.x) - cx) + angular_velocity * float(turns)
    return cx + math.cos(theta) * radius, cy + math.sin(theta) * radius


def _aim_at(source, target, ships, angular_velocity):
    speed = _fleet_speed(ships)
    tx, ty = float(target.x), float(target.y)
    for _ in range(3):
        travel = _distance_xy(float(source.x), float(source.y), tx, ty) / speed
        tx, ty = _point_after_orbit(target, travel, angular_velocity)
    return _angle_to_xy(source, tx, ty), tx, ty


def _segment_distance_to_center(source, tx, ty):
    cx, cy = _center_xy()
    sx, sy = float(source.x), float(source.y)
    vx, vy = float(tx) - sx, float(ty) - sy
    length_sq = vx * vx + vy * vy
    if length_sq == 0.0:
        return _distance_xy(sx, sy, cx, cy)
    t = max(0.0, min(1.0, ((cx - sx) * vx + (cy - sy) * vy) / length_sq))
    px, py = sx + t * vx, sy + t * vy
    return _distance_xy(px, py, cx, cy)


def _clear_of_sun(source, tx, ty):
    return _segment_distance_to_center(source, tx, ty) > SUN_DANGER_RADIUS


def _capture_need(source, target, angular_velocity, base=None):
    if base is None:
        base = int(target.ships) + 1
    speed = _fleet_speed(base)
    eta = _distance(source, target) / speed
    production_buffer = 0 if int(target.owner) == NEUTRAL else int(math.ceil(float(target.production) * eta))
    return max(1, int(target.ships) + production_buffer + 1)


def _state_for(player, obs):
    state = _STATE.setdefault(
        player,
        {
            "bursts": [],
            "prime_quadrant": None,
            "opened": False,
            "opening_target_ids": [],
            "opening_launched_ids": [],
            "primary_anchor_id": None,
        },
    )
    turn = _obs_get(obs, "step", _obs_get(obs, "turn", None))
    last_turn = state.get("last_turn")
    if turn is not None and last_turn is not None and turn < last_turn:
        state.clear()
        state.update(
            {
                "bursts": [],
                "prime_quadrant": None,
                "opened": False,
                "opening_target_ids": [],
                "opening_launched_ids": [],
                "primary_anchor_id": None,
            }
        )
    state["last_turn"] = turn
    return state


def _choose_prime_quadrant(state, my_planets):
    if state.get("prime_quadrant") is not None:
        return state["prime_quadrant"]
    if not my_planets:
        state["prime_quadrant"] = 0
        return state["prime_quadrant"]
    score = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for p in my_planets:
        score[_quadrant(p)] += float(p.ships) + 8.0 * float(p.production)
    state["prime_quadrant"] = max(score, key=score.get)
    return state["prime_quadrant"]


def _drain_bursts(state, player, planets, moves, reserved, angular_velocity):
    by_id = {p.id: p for p in planets}
    next_bursts = []
    for burst in state.get("bursts", []):
        source = by_id.get(burst["source_id"])
        target = by_id.get(burst["target_id"])
        if source is None or target is None:
            continue
        if int(source.owner) != player or int(target.owner) == player:
            continue
        if _available(source, reserved) < min(2, burst["remaining"]):
            next_bursts.append(burst)
            continue
        amount = min(2, int(burst["remaining"]))
        angle, tx, ty = _aim_at(source, target, amount, angular_velocity)
        if _clear_of_sun(source, tx, ty):
            sent = _add_move(moves, reserved, source, angle, amount)
            burst["remaining"] -= sent
        if burst["remaining"] > 0:
            next_bursts.append(burst)
    state["bursts"] = next_bursts


def _opening_target_key(p):
    return (int(p.production), float(p.radius), int(p.ships), _distance_to_center(p))


def _cleanup_opening_targets(state, player, planets):
    by_id = {p.id: p for p in planets}
    active_targets = []
    for target_id in state.get("opening_target_ids", []):
        target = by_id.get(target_id)
        if target is not None and int(target.owner) == NEUTRAL:
            active_targets.append(target_id)
    state["opening_target_ids"] = active_targets
    if not active_targets and len(state.get("opening_launched_ids", [])) >= 2:
        state["opened"] = True


def _send_opening_payload(state, source, target, moves, reserved, angular_velocity):
    need_now = int(target.ships) + 1
    available = _available(source, reserved)
    if available <= 0:
        return False

    payload = need_now if available >= need_now else available
    angle, tx, ty = _aim_at(source, target, payload, angular_velocity)
    if not _clear_of_sun(source, tx, ty):
        return False

    sent = _add_move(moves, reserved, source, angle, payload)
    if sent <= 0:
        return False

    if sent < need_now:
        remaining = max(0, int(target.ships) + 2 - sent)
        if remaining:
            state.setdefault("bursts", []).append(
                {"source_id": source.id, "target_id": target.id, "remaining": remaining}
            )
    return True


def _send_expansion_payload(state, source, target, moves, reserved, angular_velocity, need=None):
    if need is None:
        need = int(target.ships) + 2
    available = _available(source, reserved)
    if available <= 0:
        return False

    payload = need if available >= need else available
    angle, tx, ty = _aim_at(source, target, payload, angular_velocity)
    if not _clear_of_sun(source, tx, ty):
        return False

    sent = _add_move(moves, reserved, source, angle, payload)
    if sent <= 0:
        return False

    if sent < need:
        remaining = max(0, int(need) - sent)
        if remaining:
            state.setdefault("bursts", []).append(
                {"source_id": source.id, "target_id": target.id, "remaining": remaining}
            )
    return True


def _initiation_phase(state, player, planets, moves, reserved, angular_velocity):
    my_planets = [p for p in planets if int(p.owner) == player]
    neutral_planets = [p for p in planets if int(p.owner) == NEUTRAL]
    if not my_planets or not neutral_planets:
        state["opened"] = True
        return

    prime_quadrant = _choose_prime_quadrant(state, my_planets)
    _cleanup_opening_targets(state, player, planets)
    if state.get("opened"):
        return

    if len(my_planets) >= 3 and not state.get("opening_target_ids"):
        state["opened"] = True
        return

    targets = sorted([p for p in neutral_planets if _quadrant(p) == prime_quadrant], key=_opening_target_key)
    if not targets:
        state["opened"] = True
        return

    source_pool = [p for p in my_planets if _quadrant(p) == prime_quadrant]
    if not source_pool:
        return

    active_target_ids = set(state.get("opening_target_ids", []))
    launched_target_ids = set(state.get("opening_launched_ids", []))
    if len(launched_target_ids) >= 2:
        return

    burst_target_ids = {burst["target_id"] for burst in state.get("bursts", [])}
    desired_targets = [target for target in targets if target.id not in launched_target_ids]

    for target in desired_targets:
        if len(launched_target_ids) >= 2:
            break
        if target.id in active_target_ids or target.id in burst_target_ids:
            continue
        source = max(source_pool, key=lambda p: _available(p, reserved))
        if _send_opening_payload(state, source, target, moves, reserved, angular_velocity):
            state.setdefault("opening_target_ids", []).append(target.id)
            state.setdefault("opening_launched_ids", []).append(target.id)
            active_target_ids.add(target.id)
            launched_target_ids.add(target.id)


def _established_for(planets, owner, quadrant):
    big_static, small_static = _quadrant_corner_role_groups(planets, owner, quadrant)
    return len(big_static) >= 1 and len(small_static) >= 2


def _our_roles(planets, player):
    roles = {}
    for quadrant in range(4):
        if not _established_for(planets, player, quadrant):
            continue
        bigs, smalls = _quadrant_corner_role_groups(planets, player, quadrant)
        bigs = sorted(bigs, key=lambda p: -int(p.ships))
        smalls = sorted(smalls, key=lambda p: _distance_to_quadrant_corner(p, quadrant))
        if smalls:
            roles[smalls[0].id] = "sweeper"
            roles[smalls[-1].id] = "shield"
        if bigs:
            roles[bigs[0].id] = "battery"
    return roles


def _incoming_threats(player, planets, fleets):
    my_planets = [p for p in planets if int(p.owner) == player]
    threats = []
    for fleet in fleets:
        if int(fleet.owner) == player:
            continue
        fx, fy = float(fleet.x), float(fleet.y)
        dx, dy = math.cos(float(fleet.angle)), math.sin(float(fleet.angle))
        speed = _fleet_speed(fleet.ships)
        for target in my_planets:
            tx, ty = float(target.x), float(target.y)
            ahead = (tx - fx) * dx + (ty - fy) * dy
            if ahead <= 0.0:
                continue
            closest_x, closest_y = fx + ahead * dx, fy + ahead * dy
            miss = _distance_xy(closest_x, closest_y, tx, ty)
            if miss <= float(target.radius) + 0.75:
                threats.append((ahead / speed, fleet, target))
    threats.sort(key=lambda item: item[0])
    return threats


def _reactive_trap(player, planets, fleets, moves, reserved, blocked_source_ids=None):
    blocked_source_ids = blocked_source_ids or set()
    responded = set()
    for _, fleet, target in _incoming_threats(player, planets, fleets):
        if target.id in responded or target.id in blocked_source_ids:
            continue
        needed = int(fleet.ships) + 1
        if _available(target, reserved) >= needed:
            _add_move(moves, reserved, target, float(fleet.angle) + math.pi, needed)
            responded.add(target.id)


def _best_static_target(source, candidates):
    if not candidates:
        return None

    def score(target):
        nearest_corner = min(_distance_xy(float(target.x), float(target.y), cx, cy) for cx, cy in CORNERS)
        corner_bonus = 0.0 if nearest_corner <= 18.0 else 35.0
        return (corner_bonus + nearest_corner * 0.2 + _distance(source, target), int(target.ships))

    return min(candidates, key=score)


def _opposite_tangent(source, target):
    cx, cy = _center_xy()
    source_theta = math.atan2(float(source.y) - cy, float(source.x) - cx)
    target_theta = math.atan2(float(target.y) - cy, float(target.x) - cx)
    return math.cos(source_theta - target_theta) < 0.0


def _try_capture(source, target, moves, reserved, angular_velocity, max_payload=None):
    need = _capture_need(source, target, angular_velocity)
    if max_payload is not None:
        need = min(need, int(max_payload))
    if _available(source, reserved) < need:
        return False
    angle, tx, ty = _aim_at(source, target, need, angular_velocity)
    if not _clear_of_sun(source, tx, ty):
        return False
    return _add_move(moves, reserved, source, angle, need) > 0


def _feeder_logic(player, planets, moves, reserved, angular_velocity):
    for quadrant in range(4):
        if not _established_for(planets, player, quadrant):
            continue
        open_quadrants = [q for q in range(4) if q != quadrant and not _established_for(planets, player, q)]
        if not open_quadrants:
            continue
        feeder_pool = [
            p
            for p in planets
            if int(p.owner) == player and _is_static(p) and _quadrant(p) == quadrant and _available(p, reserved) > 8
        ]
        if not feeder_pool:
            continue
        feeder = min(feeder_pool, key=lambda p: _distance_to_center(p))
        target_quadrant = min(open_quadrants, key=lambda q: abs(q - quadrant) % 4)
        targets = [
            p
            for p in planets
            if int(p.owner) != player and _is_static(p) and _quadrant(p) == target_quadrant
        ]
        target = min(targets, key=lambda p: (int(p.owner) != NEUTRAL, int(p.ships), _distance(feeder, p)), default=None)
        if target is not None:
            _try_capture(feeder, target, moves, reserved, angular_velocity)


def _role_actions(player, planets, moves, reserved, roles, angular_velocity):
    by_id = {p.id: p for p in planets}
    _, cy = _center_xy()
    for planet_id, role in roles.items():
        source = by_id.get(planet_id)
        if source is None or int(source.owner) != player:
            continue
        targets = [p for p in planets if int(p.owner) != player]
        if role == "sweeper" and _available(source, reserved) >= 12:
            equator_targets = [p for p in targets if abs(float(p.y) - cy) <= 16.0]
            target = min(equator_targets, key=lambda p: (_distance(source, p), int(p.ships)), default=None)
            if target is not None:
                _try_capture(source, target, moves, reserved, angular_velocity)
        elif role == "shield" and _available(source, reserved) >= 34:
            target = min(targets, key=lambda p: (_distance(source, p), int(p.owner) != NEUTRAL), default=None)
            if target is not None:
                amount = min(30, max(20, _available(source, reserved) - 12))
                angle, tx, ty = _aim_at(source, target, amount, angular_velocity)
                if _clear_of_sun(source, tx, ty):
                    _add_move(moves, reserved, source, angle, amount)
        elif role == "battery":
            # Batteries only act in striker mode; otherwise they stockpile 60-100 ships.
            continue


def _enemy_established(planets, enemy_owner, quadrant):
    if enemy_owner == NEUTRAL:
        return False
    return _established_for(planets, enemy_owner, quadrant)


def _striker_mode(player, planets, moves, reserved, roles, angular_velocity, primary_anchor_id=None):
    batteries = [
        p
        for p in planets
        if roles.get(p.id) == "battery" and int(p.owner) == player and p.id != primary_anchor_id
    ]
    probes = [p for p in planets if int(p.owner) == player and not _is_big(p)]
    enemy_owners = {int(p.owner) for p in planets if int(p.owner) not in (player, NEUTRAL)}
    if not batteries or not probes or not enemy_owners:
        return

    for battery in sorted(batteries, key=lambda p: -int(p.ships)):
        if _available(battery, reserved) < 60:
            continue
        source_quadrant = _quadrant(battery)
        target_quadrant = _opposite_quadrant(source_quadrant)
        enemy_targets = [
            p
            for p in planets
            if int(p.owner) in enemy_owners
            and _quadrant(p) == target_quadrant
            and not _enemy_established(planets, int(p.owner), target_quadrant)
        ]
        if not enemy_targets:
            continue
        target = min(enemy_targets, key=lambda p: (int(p.ships), _distance(battery, p)))
        probe = min(probes, key=lambda p: (_distance(p, target), _distance(battery, p)))
        combined = _available(battery, reserved) + _available(probe, reserved)
        if combined < 2 * int(target.ships) + 1:
            continue
        if probe.id != battery.id and _available(battery, reserved) >= 60:
            payload = min(_available(battery, reserved) - 20, max(40, int(target.ships) + 15))
            if payload > 0:
                angle, tx, ty = _aim_at(battery, probe, payload, angular_velocity)
                if _clear_of_sun(battery, tx, ty):
                    _add_move(moves, reserved, battery, angle, payload)
        if _available(probe, reserved) >= int(target.ships) + 1:
            _try_capture(probe, target, moves, reserved, angular_velocity)


def _update_primary_anchor(state, player, planets):
    my_planets = [p for p in planets if int(p.owner) == player]
    prime_quadrant = _choose_prime_quadrant(state, my_planets)
    anchor_id = state.get("primary_anchor_id")
    current_anchor = next((p for p in my_planets if p.id == anchor_id), None)
    if (
        current_anchor is not None
        and _is_quadrant_big(current_anchor, planets, prime_quadrant)
    ):
        return current_anchor

    candidates = [
        p
        for p in my_planets
        if _is_quadrant_big(p, planets, prime_quadrant)
    ]
    if not candidates:
        state["primary_anchor_id"] = None
        return None

    anchor = min(candidates, key=lambda p: (int(p.id), -int(p.ships)))
    state["primary_anchor_id"] = anchor.id
    return anchor


def _primary_anchor_in_battery_mode(state, player, planets):
    anchor = _update_primary_anchor(state, player, planets)
    if anchor is None:
        return None
    if _established_for(planets, player, _quadrant(anchor)):
        return anchor
    return None


def _primary_anchor_action(state, player, planets, moves, reserved, angular_velocity):
    anchor = _primary_anchor_in_battery_mode(state, player, planets)
    if anchor is None:
        return

    anchor_quadrant = _quadrant(anchor)
    _, small_nodes = _quadrant_corner_role_groups(planets, player, anchor_quadrant)
    small_nodes = [p for p in small_nodes if p.id != anchor.id]
    if not small_nodes or _available(anchor, reserved) <= 60:
        return

    target = min(small_nodes, key=lambda p: (int(p.ships), _distance(anchor, p)))
    amount = min(_available(anchor, reserved) - 60, max(0, 35 - int(target.ships)))
    if amount <= 0:
        return

    angle, tx, ty = _aim_at(anchor, target, amount, angular_velocity)
    if _clear_of_sun(anchor, tx, ty):
        _add_move(moves, reserved, anchor, angle, amount)


def _needed_establishment_targets(planets, player, quadrant, claimed_target_ids):
    if _established_for(planets, player, quadrant):
        return []

    owned_bigs, owned_smalls = _quadrant_corner_role_groups(planets, player, quadrant)
    big_production = _quadrant_big_production(planets, quadrant)
    corner_targets = [
        p
        for p in planets
        if int(p.owner) != player
        and p.id not in claimed_target_ids
        and _is_quadrant_corner_node(p, quadrant)
    ]

    needed = []
    if big_production is not None and not owned_bigs:
        needed.extend([p for p in corner_targets if int(p.production) == big_production])
    if len(owned_smalls) < 2:
        small_targets = [p for p in corner_targets if big_production is None or int(p.production) < big_production]
        needed.extend(small_targets or [p for p in corner_targets if p not in needed])
    return needed or corner_targets


def _nearest_unconquered_targets(source, player, planets, claimed_target_ids):
    candidates = [p for p in planets if int(p.owner) != player and p.id not in claimed_target_ids]
    if not candidates:
        return []

    source_quadrant = _quadrant(source)
    establishment_targets = _needed_establishment_targets(planets, player, source_quadrant, claimed_target_ids)
    if establishment_targets:
        candidates = establishment_targets
    elif not _established_for(planets, player, source_quadrant):
        same_quadrant_targets = [p for p in candidates if _quadrant(p) == source_quadrant]
        if same_quadrant_targets:
            candidates = same_quadrant_targets

    return sorted(
        candidates,
        key=lambda p: (
            int(p.owner) != NEUTRAL,
            _distance(source, p),
            int(p.ships),
            _distance_to_quadrant_corner(p, _quadrant(p)),
        ),
    )


def _expansion(state, player, planets, moves, reserved, roles, angular_velocity):
    role_ids = set(roles)
    my_planets = [p for p in planets if int(p.owner) == player]
    if len(my_planets) < 2:
        return

    battery_anchor = _primary_anchor_in_battery_mode(state, player, planets)
    battery_anchor_id = battery_anchor.id if battery_anchor is not None else None
    claimed_target_ids = {burst["target_id"] for burst in state.get("bursts", [])}

    for source in sorted(
        my_planets,
        key=lambda p: (
            _established_for(planets, player, _quadrant(p)),
            _distance_to_quadrant_corner(p, _quadrant(p)),
            -int(p.ships),
        ),
    ):
        source_established = _established_for(planets, player, _quadrant(source))
        if source.id == battery_anchor_id:
            continue
        if source.id in role_ids and source_established:
            continue
        available = _available(source, reserved)
        if available <= 0:
            continue

        candidates = _nearest_unconquered_targets(source, player, planets, claimed_target_ids)
        for target in candidates[:6]:
            need = max(int(target.ships) + 2, _capture_need(source, target, angular_velocity, base=int(target.ships) + 2))
            if _send_expansion_payload(state, source, target, moves, reserved, angular_velocity, need=need):
                claimed_target_ids.add(target.id)
                break


def anchor_feeder_agent(obs, config=None):
    player, planets, fleets, angular_velocity, _ = _parse(obs)
    state = _state_for(player, obs)
    moves = []
    reserved = {}

    if not planets:
        return moves

    battery_anchor = _primary_anchor_in_battery_mode(state, player, planets)
    blocked_trap_ids = {battery_anchor.id} if battery_anchor is not None else set()
    roles = _our_roles(planets, player)

    _reactive_trap(player, planets, fleets, moves, reserved, blocked_source_ids=blocked_trap_ids)
    _drain_bursts(state, player, planets, moves, reserved, angular_velocity)
    _initiation_phase(state, player, planets, moves, reserved, angular_velocity)
    owned_count = sum(1 for p in planets if int(p.owner) == player)
    if owned_count < 2 and state.get("opening_target_ids"):
        return moves
    if owned_count < 2 and not state.get("opened") and len(state.get("opening_launched_ids", [])) < 2:
        return moves

    roles = _our_roles(planets, player)
    _primary_anchor_action(state, player, planets, moves, reserved, angular_velocity)
    primary_anchor_id = state.get("primary_anchor_id")
    _striker_mode(player, planets, moves, reserved, roles, angular_velocity, primary_anchor_id=primary_anchor_id)
    _feeder_logic(player, planets, moves, reserved, angular_velocity)
    _role_actions(player, planets, moves, reserved, roles, angular_velocity)
    _expansion(state, player, planets, moves, reserved, roles, angular_velocity)

    return moves


def agent(obs, config=None):
    return anchor_feeder_agent(obs, config)


nearest_planet_sniper = anchor_feeder_agent
