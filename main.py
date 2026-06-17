import math
import os
import time
from collections import namedtuple
from pathlib import Path

try:
    from agents import agent_smith as _smith_controller
except Exception:
    _smith_controller = None

try:
    from agents import agent_1039_launch_safety as _agent_1039_controller
except Exception:
    _agent_1039_controller = None

try:
    from agents import agent_1200_ppo_strategy as _agent_1200_controller
except Exception:
    _agent_1200_controller = None

try:
    from agents.best_orbit import agent_best_orbit as _agent_best_controller
except Exception:
    _agent_best_controller = None

try:
    from agents.light_intruder import agent_light_intruder as _agent_intruder_controller
except Exception:
    _agent_intruder_controller = None

try:
    import numpy as np
except Exception:
    np = None

try:
    from tactical_features import (
        action_feature_vector_for_state as _tf_action_feature_vector_for_state,
        action_penalty_profile_for_state as _tf_action_penalty_profile_for_state,
        infer_action_target as _tf_infer_action_target,
        infer_role_assignments_from_state as _tf_infer_roles,
        numeric_quadrant_array as _tf_numeric_quadrant_array,
        role_scores as _tf_role_scores,
        trend_identity_for_target as _tf_trend_identity_for_target,
    )
except Exception:
    _tf_action_feature_vector_for_state = None
    _tf_action_penalty_profile_for_state = None
    _tf_infer_action_target = None
    _tf_infer_roles = None
    _tf_numeric_quadrant_array = None
    _tf_role_scores = None
    _tf_trend_identity_for_target = None

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet
except Exception:
    Planet = namedtuple("Planet", "id owner x y radius ships production")
    Fleet = namedtuple("Fleet", "id owner x y angle from_planet_id ships")

try:
    from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, ROTATION_RADIUS_LIMIT
except Exception:
    CENTER = (50.0, 50.0)
    ROTATION_RADIUS_LIMIT = 50.0


NEUTRAL = -1
BOARD_SIZE = 100.0
SHIP_SPEED_MAX = 6.0
CORNERS = ((0.0, 0.0), (0.0, BOARD_SIZE), (BOARD_SIZE, 0.0), (BOARD_SIZE, BOARD_SIZE))
QUADRANT_CORNERS = {
    0: (BOARD_SIZE, BOARD_SIZE),
    1: (0.0, BOARD_SIZE),
    2: (0.0, 0.0),
    3: (BOARD_SIZE, 0.0),
}
CORNER_CLUSTER_RADIUS = 45.0
SUN_RADIUS = 10.0
SUN_DANGER_RADIUS = SUN_RADIUS + 1.0
FLEET_SPAWN_CLEARANCE = 0.2
OPENING_STALL_SHIPS = 20
SMALL_ATTACK_READY_SHIPS = 20
BIG_ATTACK_READY_SHIPS = 40
EASY_TARGET_SHIPS = 10
TOTAL_STEPS = 500.0
TACTICAL_FEATURE_SCALES = (
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
)
TACTICAL_MODEL_TOP_K = 24
ATTACK_SUPPORT_RADIUS = 18.0
ATTACK_LONG_RANGE_RADIUS = 26.0
ATTACK_MIN_FRONTLINE_SHIPS = 24
ESTABLISHED_STATIC_ASSAULT_RADIUS = 24.0
ANCHOR_SMALL_HOLD_SHIPS = 18
ANCHOR_SMALL_PRESSURED_HOLD_SHIPS = 24
PRE_ESTABLISHMENT_BURST_MAX_NEED = 12
PRE_ESTABLISHMENT_TRAP_MAX_ETA = 7.0
PRE_ESTABLISHMENT_TRAP_SPARE_SHIPS = 4
ATTACKER_STAGE_TARGET_SHIPS = 72
ATTACKER_STAGE_IN_QUADRANT_SHIPS = 96
ATTACKER_STAGE_MIN_FEED = 28
ATTACKER_STAGE_BIG_FEED = 44
USE_SMITH_DELEGATION = False
MODEL_PRESSURE_ARRIVAL_WINDOW = 9.0
MODEL_PRESSURE_MIN_SOURCE_COUNT = 2
MODEL_PRESSURE_MIN_DEFENSE_RATIO = 0.45
MODEL_PRESSURE_MIN_HOSTILE_SHIPS = 10
MODEL_DECISIVE_ATTACK_MIN_MARGIN = 0.0
MODEL_DECISIVE_OVERPOWER_RATIO = 1.35
MODEL_DECISIVE_OVERPOWER_MIN_EXTRA = 14
PROPOSAL_AGENT_TIME_BUDGET = 0.75
PROPOSAL_MAX_MOVES = 12
PROPOSAL_SOURCE_PRIOR = {
    "best": 0.55,
    "intruder": 0.40,
    "ppo1200": 0.25,
    "smith": 0.20,
}
CONTROLLER_SOURCE_NAMES = ("best", "intruder", "ppo1200", "smith", "hold")
CONTROLLER_TREND_NAMES = (
    "neutral",
    "pressured",
    "cash_in",
    "overtake_window",
    "chasing_leader",
    "pressure-cover",
    "foundation",
    "maintenance",
    "thin-enemy",
    "thin-neutral",
)
CONTROLLER_MODEL_WEIGHT = 6.0
OPPORTUNITY_TREND_BONUS = {
    "overtake_window": 2.60,
    "cash_in": 2.25,
    "chasing_leader": 1.85,
    "pressured": 1.45,
    "neutral": 0.0,
}
PlanetProfile = namedtuple("PlanetProfile", "planet labels quadrant angle distance_to_center corner_distance")
AttackMeasurement = namedtuple(
    "AttackMeasurement",
    "angle target_x target_y eta speed launch_x launch_y sun_distance blocker_id clear",
)
_STATE = {}
_TACTICAL_MODEL = None
_TACTICAL_MODEL_ATTEMPTED = False
_CONTROLLER_MODEL = None
_CONTROLLER_MODEL_ATTEMPTED = False
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")


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


def _is_orbiting(p):
    return not _is_static(p)


def _is_big(p):
    production = int(p.production)
    if production >= 5:
        return True
    if _is_static(p):
        return production >= 2
    return production >= 3


def _is_large_production(p):
    return int(p.production) >= 5


def _planet_angle(p):
    cx, cy = _center_xy()
    return math.atan2(float(p.y) - cy, float(p.x) - cx)


def _angle_quadrant(angle):
    normalized = float(angle) % (2.0 * math.pi)
    return int(normalized // (math.pi / 2.0)) % 4


def _quadrant(p):
    return _angle_quadrant(_planet_angle(p))


def _equator_side(p):
    _, cy = _center_xy()
    return 1 if float(p.y) >= cy else -1


def _same_equator_side(a, b):
    return _equator_side(a) == _equator_side(b)


def _same_equator_targets(source, candidates):
    return [p for p in candidates if _same_equator_side(source, p)]


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


def _angle_from_xy(ax, ay, bx, by):
    return math.atan2(float(by) - float(ay), float(bx) - float(ax))


def _norm_angle(angle):
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _fleet_speed(ships):
    ships = max(1.0, float(ships))
    if ships <= 1.0:
        return 1.0
    scale = max(0.0, min(1.0, math.log(ships) / math.log(1000.0)))
    return max(1.0, min(SHIP_SPEED_MAX, 1.0 + (SHIP_SPEED_MAX - 1.0) * (scale ** 1.5)))


def _available(source, reserved):
    return max(0, int(source.ships) - reserved.get(source.id, 0))


def _attack_ready_ships(source):
    return BIG_ATTACK_READY_SHIPS if _is_big(source) else SMALL_ATTACK_READY_SHIPS


def _post_init_attack_ready(source, reserved):
    return _available(source, reserved) >= _attack_ready_ships(source)


def _source_can_attempt_capture(source, reserved):
    if _is_large_production(source):
        return _available(source, reserved) > 0
    return _post_init_attack_ready(source, reserved)


def _capture_ready_for_need(source, reserved, need):
    if _is_large_production(source):
        return _available(source, reserved) >= int(need)
    return _post_init_attack_ready(source, reserved)


def _expansion_ready_for_need(source, reserved, need):
    return _available(source, reserved) >= int(need)


def _add_move(moves, reserved, source, angle, ships):
    amount = min(max(0, int(ships)), _available(source, reserved))
    if amount <= 0:
        return 0
    safe_angle = _norm_angle(angle)
    if not _angle_clear_of_sun(source, safe_angle):
        return 0
    moves.append([source.id, safe_angle, amount])
    reserved[source.id] = reserved.get(source.id, 0) + amount
    return amount


def _point_after_orbit(p, turns, angular_velocity):
    if _is_static(p) or angular_velocity == 0.0:
        return float(p.x), float(p.y)
    cx, cy = _center_xy()
    radius = _distance_to_center(p)
    theta = math.atan2(float(p.y) - cy, float(p.x) - cx) + angular_velocity * float(turns)
    return cx + math.cos(theta) * radius, cy + math.sin(theta) * radius


def _launch_point(source, angle):
    offset = float(source.radius) + FLEET_SPAWN_CLEARANCE
    return (
        float(source.x) + math.cos(float(angle)) * offset,
        float(source.y) + math.sin(float(angle)) * offset,
    )


def _ray_board_exit_point(sx, sy, angle):
    sx, sy = float(sx), float(sy)
    dx, dy = math.cos(float(angle)), math.sin(float(angle))
    distances = []
    if dx > 1e-9:
        distances.append((BOARD_SIZE - sx) / dx)
    elif dx < -1e-9:
        distances.append((0.0 - sx) / dx)
    if dy > 1e-9:
        distances.append((BOARD_SIZE - sy) / dy)
    elif dy < -1e-9:
        distances.append((0.0 - sy) / dy)
    positive = [distance for distance in distances if distance >= 0.0]
    if not positive:
        return sx, sy
    travel = min(positive)
    return sx + dx * travel, sy + dy * travel


def _aim_solution(source, target, ships, angular_velocity):
    speed = _fleet_speed(ships)
    tx, ty = float(target.x), float(target.y)
    angle = _angle_to_xy(source, tx, ty)
    travel = 0.0
    for _ in range(6):
        lx, ly = _launch_point(source, angle)
        travel_distance = max(0.0, _distance_xy(lx, ly, tx, ty) - float(target.radius))
        travel = travel_distance / speed
        tx, ty = _point_after_orbit(target, travel, angular_velocity)
        angle = _angle_to_xy(source, tx, ty)
    lx, ly = _launch_point(source, angle)
    travel_distance = max(0.0, _distance_xy(lx, ly, tx, ty) - float(target.radius))
    travel = travel_distance / speed
    return angle, tx, ty, travel, lx, ly


def _segment_entry_distance_to_circle(sx, sy, tx, ty, cx, cy, radius):
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
    miss = _distance_xy(closest_x, closest_y, cx, cy)
    radius = float(radius)
    if miss > radius:
        return None

    entry = projection_distance - math.sqrt(max(0.0, radius * radius - miss * miss))
    if entry < 0.0:
        entry = 0.0
    if entry > length:
        return None
    return entry


def _sun_entry_distance_on_segment(sx, sy, tx, ty):
    cx, cy = _center_xy()
    return _segment_entry_distance_to_circle(sx, sy, tx, ty, cx, cy, SUN_DANGER_RADIUS)


def _angle_clear_of_sun(source, angle):
    sx, sy = _launch_point(source, angle)
    ex, ey = _ray_board_exit_point(sx, sy, angle)
    return _sun_entry_distance_on_segment(sx, sy, ex, ey) is None


def _ray_distance_to_center(source, angle):
    sx, sy = _launch_point(source, angle)
    ex, ey = _ray_board_exit_point(sx, sy, angle)
    return _segment_distance_to_center_xy(sx, sy, ex, ey)


def _first_planet_on_ray(source, angle, planets):
    if not planets:
        return None
    sx, sy = _launch_point(source, angle)
    return _first_planet_on_ray_from_point(sx, sy, angle, planets, excluded_ids={source.id})


def _first_planet_on_ray_from_point(sx, sy, angle, planets, excluded_ids=None):
    if not planets:
        return None
    excluded_ids = excluded_ids or set()
    ex, ey = _ray_board_exit_point(sx, sy, angle)
    closest = None
    closest_planet = None
    for planet in planets:
        if planet.id in excluded_ids:
            continue
        entry = _segment_entry_distance_to_circle(
            sx,
            sy,
            ex,
            ey,
            float(planet.x),
            float(planet.y),
            float(planet.radius) + 0.1,
        )
        if entry is None:
            continue
        if closest is None or entry < closest:
            closest = entry
            closest_planet = planet
    return closest_planet


def _projected_fleet_target_id(fleet, planets):
    target = _first_planet_on_ray_from_point(
        float(fleet.x),
        float(fleet.y),
        float(fleet.angle),
        planets,
        excluded_ids={int(fleet.from_planet_id)},
    )
    return target.id if target is not None else None


def _incoming_friendly_ships_by_target(player, planets, fleets):
    committed = {}
    for fleet in fleets or []:
        if int(fleet.owner) != int(player):
            continue
        target_id = _projected_fleet_target_id(fleet, planets)
        if target_id is None:
            continue
        committed[target_id] = committed.get(target_id, 0) + int(fleet.ships)
    return committed


def _claimed_capture_threshold(target):
    threshold = int(target.ships) + 1
    if int(target.owner) != NEUTRAL:
        threshold += int(target.production) + (2 if _is_static(target) else 1)
    elif _is_big(target):
        threshold += 1
    return threshold


def _claimed_target_ids(state, player, planets, fleets=None):
    claimed = {int(burst["target_id"]) for burst in state.get("bursts", [])}
    claimed.update(int(target_id) for target_id in state.get("turn_claimed_target_ids", set()))
    incoming = _incoming_friendly_ships_by_target(player, planets, fleets or [])
    for target in planets:
        if int(target.owner) == int(player):
            continue
        if incoming.get(int(target.id), 0) >= _claimed_capture_threshold(target):
            claimed.add(int(target.id))
    return claimed


def _first_planet_blocker(source, target, lx, ly, tx, ty, eta, planets, angular_velocity):
    if not planets:
        return None

    target_entry = max(0.0, _distance_xy(lx, ly, tx, ty) - float(target.radius))
    closest_entry = None
    closest_id = None
    for planet in planets:
        if planet.id in (source.id, target.id):
            continue
        px, py = _point_after_orbit(planet, eta, angular_velocity)
        entry = _segment_entry_distance_to_circle(
            lx,
            ly,
            tx,
            ty,
            px,
            py,
            float(planet.radius) + 0.1,
        )
        if entry is None or entry >= target_entry - 0.05:
            continue
        if closest_entry is None or entry < closest_entry:
            closest_entry = entry
            closest_id = planet.id
    return closest_id


def _attack_measurement(source, target, ships, angular_velocity, planets=None):
    angle, tx, ty, eta, lx, ly = _aim_solution(source, target, ships, angular_velocity)
    sun_distance = _ray_distance_to_center(source, angle)
    blocker_id = _first_planet_blocker(source, target, lx, ly, tx, ty, eta, planets, angular_velocity)
    return AttackMeasurement(
        angle=angle,
        target_x=tx,
        target_y=ty,
        eta=eta,
        speed=_fleet_speed(ships),
        launch_x=lx,
        launch_y=ly,
        sun_distance=sun_distance,
        blocker_id=blocker_id,
        clear=_angle_clear_of_sun(source, angle) and blocker_id is None,
    )


def _aim_at(source, target, ships, angular_velocity):
    measurement = _attack_measurement(source, target, ships, angular_velocity)
    return measurement.angle, measurement.target_x, measurement.target_y


def _segment_distance_to_center_xy(sx, sy, tx, ty):
    cx, cy = _center_xy()
    sx, sy = float(sx), float(sy)
    vx, vy = float(tx) - sx, float(ty) - sy
    length_sq = vx * vx + vy * vy
    if length_sq == 0.0:
        return _distance_xy(sx, sy, cx, cy)
    t = max(0.0, min(1.0, ((cx - sx) * vx + (cy - sy) * vy) / length_sq))
    px, py = sx + t * vx, sy + t * vy
    return _distance_xy(px, py, cx, cy)


def _segment_distance_to_center(source, tx, ty, angle=None):
    if angle is None:
        sx, sy = float(source.x), float(source.y)
    else:
        sx, sy = _launch_point(source, angle)
    return _segment_distance_to_center_xy(sx, sy, tx, ty)


def _clear_of_sun(source, tx, ty, angle=None):
    if angle is not None:
        return _angle_clear_of_sun(source, angle)
    return _segment_distance_to_center(source, tx, ty, angle=angle) > SUN_DANGER_RADIUS


def _capture_need(source, target, angular_velocity, base=None):
    if _is_large_production(source):
        return max(1, int(target.ships) + 1)
    if base is None:
        base = int(target.ships) + 1
    eta = _attack_measurement(source, target, base, angular_velocity).eta
    production_buffer = 0 if int(target.owner) == NEUTRAL else int(math.ceil(float(target.production) * eta))
    return max(1, int(target.ships) + production_buffer + 1)


def _planned_capture_need(source, target, angular_velocity, safety=2):
    return max(1, int(target.ships) + 1)


def _local_support_totals(player, target, planets, fleets=None, radius=ATTACK_SUPPORT_RADIUS):
    friendly = 0
    enemy = 0
    for planet in planets:
        if planet.id == target.id:
            continue
        if _distance(planet, target) > radius:
            continue
        weight = int(planet.ships) + 2 * int(planet.production)
        if int(planet.owner) == player:
            friendly += weight
        elif int(planet.owner) not in (player, NEUTRAL):
            enemy += weight

    for fleet in fleets or []:
        if _distance_xy(float(fleet.x), float(fleet.y), float(target.x), float(target.y)) > radius:
            continue
        if int(fleet.owner) == player:
            friendly += int(fleet.ships)
        elif int(fleet.owner) >= 0 and int(fleet.owner) != player:
            enemy += int(fleet.ships)

    return friendly, enemy


def _quadrant_totals(player, quadrant, planets, fleets=None):
    ours = 0
    enemy = 0
    for planet in planets:
        if _quadrant(planet) != quadrant:
            continue
        weight = int(planet.ships) + 3 * int(planet.production)
        if int(planet.owner) == player:
            ours += weight
        elif int(planet.owner) not in (player, NEUTRAL):
            enemy += weight

    for fleet in fleets or []:
        if _angle_quadrant(math.atan2(float(fleet.y) - _center_xy()[1], float(fleet.x) - _center_xy()[0])) != quadrant:
            continue
        if int(fleet.owner) == player:
            ours += int(fleet.ships)
        elif int(fleet.owner) >= 0 and int(fleet.owner) != player:
            enemy += int(fleet.ships)

    return ours, enemy


def _quadrant_control_margin(player, quadrant, planets, fleets=None):
    ours, enemy = _quadrant_totals(player, quadrant, planets, fleets=fleets)
    return ours - enemy


def _adjacent_quadrants(quadrant):
    quadrant = int(quadrant)
    return ((quadrant - 1) % 4, (quadrant + 1) % 4)


def _half_control_margin(player, quadrants, planets, fleets=None):
    return sum(_quadrant_control_margin(player, quadrant, planets, fleets=fleets) for quadrant in quadrants)


def _offensive_capture_need(source, target, player, planets, fleets, angular_velocity):
    base = _capture_need(source, target, angular_velocity)
    friendly_support, enemy_support = _local_support_totals(player, target, planets, fleets=fleets)
    support_pressure = max(0, enemy_support - friendly_support)
    support_buffer = min(18, int(math.ceil(support_pressure / 6.0)))
    static_buffer = 2 if _is_static(target) and int(target.owner) != NEUTRAL else 0
    big_buffer = 2 if _is_big(target) and int(target.owner) != NEUTRAL else 0
    return base + support_buffer + static_buffer + big_buffer


def _decisive_attack_margin(source, target, player, planets, fleets, angular_velocity, ships=None):
    if source is None or target is None or int(target.owner) in (int(player), NEUTRAL):
        return 0.0
    payload = int(source.ships) if ships is None else int(ships)
    need = _offensive_capture_need(source, target, player, planets, fleets, angular_velocity)
    return max(-1.0, min(1.0, (float(payload) - float(need)) / max(12.0, float(need))))


def _decisive_attack_payload(available, need, margin):
    available = int(available)
    need = int(need)
    if available <= need:
        return max(1, min(available, need))
    ratio = MODEL_DECISIVE_OVERPOWER_RATIO + min(0.45, max(0.0, float(margin)) * 0.30)
    desired = max(
        need,
        int(math.ceil(float(need) * ratio)),
        need + MODEL_DECISIVE_OVERPOWER_MIN_EXTRA,
    )
    return max(1, min(available, desired))


def _small_node_hold_level(player, target, planets):
    friendly_support, enemy_support = _local_support_totals(player, target, planets, radius=ATTACK_SUPPORT_RADIUS)
    support_gap = enemy_support - friendly_support
    if support_gap > 0:
        return ANCHOR_SMALL_PRESSURED_HOLD_SHIPS + min(8, int(math.ceil(support_gap / 10.0)))

    local_enemy_static = any(
        int(planet.owner) not in (player, NEUTRAL)
        and _is_static(planet)
        and _quadrant(planet) == _quadrant(target)
        and _distance(planet, target) <= ESTABLISHED_STATIC_ASSAULT_RADIUS
        for planet in planets
    )
    if local_enemy_static:
        return ANCHOR_SMALL_PRESSURED_HOLD_SHIPS
    return ANCHOR_SMALL_HOLD_SHIPS


def _is_easy_target(p):
    return int(p.ships) <= EASY_TARGET_SHIPS


def _planet_labels(p, player=None, planets=None):
    quadrant = _quadrant(p)
    labels = []
    if player is not None:
        if int(p.owner) == player:
            labels.append("ours")
        elif int(p.owner) == NEUTRAL:
            labels.append("neutral")
        else:
            labels.append("enemy")
    elif int(p.owner) == NEUTRAL:
        labels.append("neutral")
    else:
        labels.append("owned")

    labels.append("orbiting" if _is_orbiting(p) else "static")
    labels.append("big" if _is_big(p) else "small")
    labels.append("easy" if _is_easy_target(p) else "held")
    labels.append("corner" if _is_corner_node(p) else "non-corner")
    labels.append("q%d" % quadrant)

    if planets is not None and _is_corner_node(p):
        labels.append("corner-big" if _is_quadrant_big(p, planets, quadrant) else "corner-small")

    return tuple(labels)


def _planet_profile(p, player=None, planets=None):
    quadrant = _quadrant(p)
    return PlanetProfile(
        planet=p,
        labels=_planet_labels(p, player=player, planets=planets),
        quadrant=quadrant,
        angle=_planet_angle(p),
        distance_to_center=_distance_to_center(p),
        corner_distance=_distance_to_quadrant_corner(p, quadrant),
    )


def _smallest_target_key(source, target):
    target_profile = _planet_profile(target)
    return (
        "easy" not in target_profile.labels,
        int(target.owner) != NEUTRAL,
        int(target.ships),
        _distance(source, target),
        -int(target.production),
        -float(target.radius),
        target_profile.corner_distance,
    )


def _easy_targets(source, player, planets, claimed_target_ids):
    return sorted(
        [
            p
            for p in planets
            if int(p.owner) != player and p.id not in claimed_target_ids and _is_easy_target(p)
        ],
        key=lambda p: _smallest_target_key(source, p),
    )


def _prepend_unique_targets(preferred, candidates):
    seen = set()
    ordered = []
    for target in list(preferred) + list(candidates):
        if target is None or target.id in seen:
            continue
        seen.add(target.id)
        ordered.append(target)
    return ordered


def _learning_focus_targets(state, player, planets, fleets, source, candidates):
    if len(candidates) < 2:
        return candidates

    source_quadrant = _quadrant(source)
    control_half = _current_control_half(state, player, planets, fleets=fleets)
    attacker_target_quadrant = state.get("attacker_target_quadrant")
    highlighted = []

    def _add_group(group):
        if not group:
            return
        ordered = sorted(group, key=lambda target: _smallest_target_key(source, target))
        highlighted.extend(ordered[:2])
        highlighted.extend([target for target in ordered if _is_static(target)][:2])
        highlighted.extend([target for target in ordered if _is_orbiting(target)][:2])

    same_quadrant = [target for target in candidates if _quadrant(target) == source_quadrant]
    target_quadrant_group = (
        [target for target in candidates if _quadrant(target) == int(attacker_target_quadrant)]
        if attacker_target_quadrant is not None
        else []
    )
    half_group = [target for target in candidates if control_half and _quadrant(target) in control_half]

    for group in (target_quadrant_group, same_quadrant, half_group, candidates):
        _add_group(group)

    return _prepend_unique_targets(highlighted, candidates)


def _empty_tactical_tendency():
    return {
        "launches": 0,
        "ships_launched": 0,
        "neutral_targets": 0,
        "enemy_targets": 0,
        "friendly_targets": 0,
        "static_targets": 0,
        "rotating_targets": 0,
        "central_rotating_big": 0,
        "central_rotating_small": 0,
        "captures": 0,
        "losses": 0,
    }


def _fresh_state():
    return {
        "bursts": [],
        "turn_claimed_target_ids": set(),
        "prime_quadrant": None,
        "opened": False,
        "opening_target_ids": [],
        "opening_launched_ids": [],
        "primary_anchor_id": None,
        "attacker_planet_id": None,
        "attacker_target_quadrant": None,
        "static_collector_id": None,
        "previous_owned_ids": None,
        "previous_owner_by_planet": None,
        "recent_static_capture_id": None,
        "recent_static_capture_quadrant": None,
        "tactical_tendency": _empty_tactical_tendency(),
        "controller_intent": None,
        "controller_intent_until": -1,
        "controller_intent_reason": "smith-native",
        "model_controller_action": None,
        "model_controller_reason": "idle",
    }


def _sigmoid(value):
    value = max(-60.0, min(60.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _configured_weight_path(env_name):
    value = os.environ.get(env_name)
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _load_tactical_model():
    global _TACTICAL_MODEL, _TACTICAL_MODEL_ATTEMPTED
    if _TACTICAL_MODEL_ATTEMPTED:
        return _TACTICAL_MODEL

    _TACTICAL_MODEL_ATTEMPTED = True
    if np is None:
        return None

    candidates = []
    configured_model = _configured_weight_path("ORBIT_MODEL_WEIGHTS")
    if configured_model is not None:
        candidates.append(configured_model)
    candidates.append(ROOT / "model_weights.npz")
    candidates.extend(sorted(ROOT.glob("data/training_runs/*/model_weights.npz"), reverse=True))
    candidates.extend(sorted(ROOT.glob("TRAINING_RUNS/*/model_weights.npz"), reverse=True))
    for path in candidates:
        if not path.exists():
            continue
        try:
            with np.load(path) as model:
                weights = np.asarray(model["weights"], dtype=np.float32)
                bias = float(np.asarray(model["bias"], dtype=np.float32).reshape(-1)[0])
                mean = np.asarray(model["mean"], dtype=np.float32)
                std = np.asarray(model["std"], dtype=np.float32)
        except Exception:
            continue
        if weights.ndim != 1 or weights.shape[0] <= 0 or mean.shape != weights.shape or std.shape != weights.shape:
            continue
        std = std.copy()
        std[std < 1e-6] = 1.0
        _TACTICAL_MODEL = {
            "path": str(path),
            "weights": weights,
            "bias": bias,
            "mean": mean,
            "std": std,
        }
        return _TACTICAL_MODEL
    return None


def _load_controller_model():
    global _CONTROLLER_MODEL, _CONTROLLER_MODEL_ATTEMPTED
    if _CONTROLLER_MODEL_ATTEMPTED:
        return _CONTROLLER_MODEL

    _CONTROLLER_MODEL_ATTEMPTED = True
    if os.environ.get("ORBIT_DISABLE_CONTROLLER", "").lower() in ("1", "true", "yes"):
        return None
    if np is None:
        return None

    candidates = []
    configured_model = _configured_weight_path("ORBIT_CONTROLLER_WEIGHTS")
    if configured_model is not None:
        candidates.append(configured_model)
    candidates.append(ROOT / "controller_weights.npz")
    if os.environ.get("ORBIT_AUTO_CONTROLLER_WEIGHTS", "").lower() in ("1", "true", "yes"):
        candidates.extend(sorted(ROOT.glob("data/training_runs/*/controller_weights.npz"), reverse=True))
        candidates.extend(sorted(ROOT.glob("TRAINING_RUNS/*/controller_weights.npz"), reverse=True))
    for path in candidates:
        if not path.exists():
            continue
        try:
            with np.load(path) as model:
                weights = np.asarray(model["weights"], dtype=np.float32)
                bias = float(np.asarray(model["bias"], dtype=np.float32).reshape(-1)[0])
                mean = np.asarray(model["mean"], dtype=np.float32)
                std = np.asarray(model["std"], dtype=np.float32)
        except Exception:
            continue
        if weights.ndim != 1 or weights.shape[0] <= 0 or mean.shape != weights.shape or std.shape != weights.shape:
            continue
        std = std.copy()
        std[std < 1e-6] = 1.0
        _CONTROLLER_MODEL = {
            "path": str(path),
            "weights": weights,
            "bias": bias,
            "mean": mean,
            "std": std,
        }
        return _CONTROLLER_MODEL
    return None


def _player_count(planets, fleets, player):
    owners = [int(player)]
    owners.extend(int(p.owner) for p in planets if int(p.owner) >= 0)
    owners.extend(int(f.owner) for f in fleets if int(f.owner) >= 0)
    return max(2, max(owners, default=0) + 1)


def _planet_row(p):
    return [int(p.id), int(p.owner), float(p.x), float(p.y), float(p.radius), int(p.ships), int(p.production)]


def _fleet_row(f):
    return [int(f.id), int(f.owner), float(f.x), float(f.y), float(f.angle), int(f.from_planet_id), int(f.ships)]


def _tactical_obs(player, step, planets, fleets, angular_velocity):
    return {
        "player": int(player),
        "step": int(step),
        "angular_velocity": float(angular_velocity),
        "planets": [_planet_row(p) for p in planets],
        "fleets": [_fleet_row(f) for f in fleets],
        "comet_planet_ids": [],
    }


def _fallback_numeric_quadrant_array(planets, fleets, player, player_count):
    rows = []
    established_by_owner = {owner: _operationally_established_quadrants(planets, owner) for owner in range(player_count)}
    for quadrant in range(4):
        our_ships = 0
        our_production = 0
        our_big_static = 0
        our_small_static = 0
        our_rotating_ships = 0
        neutral_ships = 0
        neutral_production = 0
        neutral_big_static = 0
        neutral_small_static = 0
        neutral_rotating_ships = 0
        enemy_ships = 0
        enemy_production = 0
        enemy_fleet_ships = 0
        enemy_established = 0
        our_fleet_ships = 0
        our_established = 1 if quadrant in established_by_owner.get(int(player), set()) else 0

        for owner in range(player_count):
            if owner != int(player) and quadrant in established_by_owner.get(owner, set()):
                enemy_established += 1

        for planet in planets:
            if _quadrant(planet) != quadrant:
                continue
            static = _is_static(planet)
            big_static = static and int(planet.production) >= 5
            small_static = static and not big_static
            if int(planet.owner) == int(player):
                our_ships += int(planet.ships)
                our_production += int(planet.production)
                if big_static:
                    our_big_static += 1
                if small_static:
                    our_small_static += 1
                if not static:
                    our_rotating_ships += int(planet.ships)
            elif int(planet.owner) == NEUTRAL:
                neutral_ships += int(planet.ships)
                neutral_production += int(planet.production)
                if big_static:
                    neutral_big_static += 1
                if small_static:
                    neutral_small_static += 1
                if not static:
                    neutral_rotating_ships += int(planet.ships)
            else:
                enemy_ships += int(planet.ships)
                enemy_production += int(planet.production)

        for fleet in fleets:
            fleet_quadrant = _angle_quadrant(math.atan2(float(fleet.y) - _center_xy()[1], float(fleet.x) - _center_xy()[0]))
            if fleet_quadrant != quadrant:
                continue
            if int(fleet.owner) == int(player):
                our_fleet_ships += int(fleet.ships)
            elif int(fleet.owner) >= 0:
                enemy_fleet_ships += int(fleet.ships)

        rows.append(
            [
                our_ships,
                our_production,
                our_big_static,
                our_small_static,
                our_rotating_ships,
                our_fleet_ships,
                our_established,
                enemy_ships,
                enemy_production,
                enemy_fleet_ships,
                enemy_established,
                neutral_ships,
                neutral_production,
                neutral_big_static,
                neutral_small_static,
                neutral_rotating_ships,
            ]
        )
    return rows


def _flatten_quadrant_features(obs, player, planets, fleets, player_count):
    if _tf_numeric_quadrant_array is not None:
        rows = _tf_numeric_quadrant_array(obs, player=player, player_count=player_count)
    else:
        rows = _fallback_numeric_quadrant_array(planets, fleets, player, player_count)
    values = []
    for row in rows:
        for idx, value in enumerate(row):
            values.append(float(value) / TACTICAL_FEATURE_SCALES[idx])
    return values


def _flatten_role_features(obs, player, planets):
    if _tf_role_scores is None:
        return [len(_operationally_established_quadrants(planets, player)) / 4.0, 0.0, 0.0, 0.0, 0.0]

    labels = _tf_role_scores(obs, player=player, top_n=1)

    def _first_score(key, score_name):
        items = labels.get(key, [])
        if not items:
            return 0.0
        return float(items[0].get(score_name, 0.0)) / 12.0

    return [
        len(labels.get("established_quadrants", [])) / 4.0,
        _first_score("anchor_candidates", "anchor_score"),
        _first_score("feeder_candidates", "feeder_score"),
        _first_score("sweeper_candidates", "sweeper_score"),
        _first_score("strike_stage_candidates", "strike_stage_score"),
    ]


def _flatten_tactical_tendency(tendency):
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


def _tactical_feature_vector(step, player, planets, fleets, angular_velocity, tendency):
    player_count = _player_count(planets, fleets, player)
    obs = _tactical_obs(player, step, planets, fleets, angular_velocity)
    features = [
        min(float(step), TOTAL_STEPS) / TOTAL_STEPS,
        float(player_count) / 4.0,
        1.0,
        1.0 if player_count == 4 else 0.0,
    ]
    features.extend(_flatten_quadrant_features(obs, player, planets, fleets, player_count))
    features.extend(_flatten_role_features(obs, player, planets))
    features.extend(_flatten_tactical_tendency(tendency))
    return features


def _tactical_tendency(state):
    tendency = state.get("tactical_tendency")
    if tendency is None:
        tendency = _empty_tactical_tendency()
        state["tactical_tendency"] = tendency
    return tendency


def _smith_controller_intent(state, player, planets, fleets, step):
    tendency = _tactical_tendency(state)
    captures = int(tendency.get("captures", 0))
    losses = int(tendency.get("losses", 0))

    production_by_owner = {}
    ships_by_owner = {}
    fleet_ships_by_owner = {}
    for planet in planets:
        owner = int(planet.owner)
        if owner < 0:
            continue
        production_by_owner[owner] = production_by_owner.get(owner, 0) + int(planet.production)
        ships_by_owner[owner] = ships_by_owner.get(owner, 0) + int(planet.ships)
    for fleet in fleets:
        owner = int(fleet.owner)
        if owner < 0:
            continue
        fleet_ships_by_owner[owner] = fleet_ships_by_owner.get(owner, 0) + int(fleet.ships)
        ships_by_owner[owner] = ships_by_owner.get(owner, 0) + int(fleet.ships)

    enemy_owners = [owner for owner in production_by_owner if owner != int(player)]
    my_production = float(production_by_owner.get(int(player), 0))
    enemy_production = float(max((production_by_owner.get(owner, 0) for owner in enemy_owners), default=0))
    my_ships = float(ships_by_owner.get(int(player), 0))
    enemy_ships = float(max((ships_by_owner.get(owner, 0) for owner in enemy_owners), default=0))
    enemy_fleet_ships = float(sum(fleet_ships_by_owner.get(owner, 0) for owner in enemy_owners))
    enemy_commitment = enemy_fleet_ships / max(1.0, sum(ships_by_owner.get(owner, 0) for owner in enemy_owners))

    evidence = {"patient": 0.0, "opportunistic": 0.0, "pressure": 0.0}
    reasons = []

    if step < 14:
        evidence["patient"] += 1.0
        reasons.append("opening")
    if enemy_production > my_production * 1.15:
        evidence["pressure"] += 1.25
        reasons.append("production-deficit")
    elif my_production > enemy_production * 1.15 and my_production > 0:
        evidence["opportunistic"] += 0.9
        reasons.append("production-window")
    if losses >= captures + 2:
        evidence["pressure"] += 1.1
        reasons.append("loss-trend")
    if captures >= losses + 2:
        evidence["opportunistic"] += 0.7
        reasons.append("capture-trend")
    if enemy_commitment >= 0.22:
        evidence["pressure"] += 0.9
        evidence["opportunistic"] += 0.35
        reasons.append("enemy-committed")
    if my_ships >= max(40.0, enemy_ships * 1.12):
        evidence["opportunistic"] += 0.8
        reasons.append("ship-window")

    trend_counts = {}
    if _tf_trend_identity_for_target is not None:
        for target in planets:
            if int(target.owner) == int(player):
                continue
            trend = _tf_trend_identity_for_target(
                planets,
                fleets,
                player,
                target,
                tendency=tendency,
            )
            trend_counts[trend] = trend_counts.get(trend, 0) + 1
    if trend_counts.get("overtake_window", 0):
        evidence["opportunistic"] += 1.2
        reasons.append("overtake-window")
    if trend_counts.get("cash_in", 0):
        evidence["opportunistic"] += 0.8
        reasons.append("cash-in")
    if trend_counts.get("pressured", 0):
        evidence["pressure"] += 0.8
    if trend_counts.get("chasing_leader", 0):
        evidence["pressure"] += 0.45
        evidence["opportunistic"] += 0.35
        reasons.append("leader-target")

    proposed, score = max(evidence.items(), key=lambda item: item[1])
    if score < 1.15:
        proposed = None

    previous = state.get("controller_intent")
    intent_until = int(state.get("controller_intent_until", -1))
    if proposed is None and previous is not None and step <= intent_until:
        proposed = previous
    elif proposed is not None:
        state["controller_intent"] = proposed
        state["controller_intent_until"] = step + 4
    else:
        state["controller_intent"] = None

    state["controller_intent_reason"] = ",".join(reasons) if reasons else "smith-native"
    return proposed


def _update_tactical_events(state, player, planets):
    tendency = _tactical_tendency(state)
    current_owner_by_planet = {int(p.id): int(p.owner) for p in planets}
    previous = state.get("previous_owner_by_planet")
    if previous is not None:
        for planet_id, new_owner in current_owner_by_planet.items():
            old_owner = previous.get(planet_id)
            if old_owner is None or old_owner == new_owner:
                continue
            if new_owner == int(player) and old_owner != int(player):
                tendency["captures"] += 1
            elif old_owner == int(player) and new_owner != int(player):
                tendency["losses"] += 1
    state["previous_owner_by_planet"] = current_owner_by_planet


def _project_tactical_tendency(state, player, target, ships):
    tendency = dict(_tactical_tendency(state))
    tendency["launches"] += 1
    tendency["ships_launched"] += int(ships)
    if target is not None:
        if int(target.owner) == NEUTRAL:
            tendency["neutral_targets"] += 1
        elif int(target.owner) == int(player):
            tendency["friendly_targets"] += 1
        else:
            tendency["enemy_targets"] += 1
        if _is_static(target):
            tendency["static_targets"] += 1
        else:
            tendency["rotating_targets"] += 1
            if int(target.production) >= 5:
                tendency["central_rotating_big"] += 1
            else:
                tendency["central_rotating_small"] += 1
    return tendency


def _record_tactical_move(state, player, target, ships):
    if state is None or target is None or int(ships) <= 0:
        return
    tendency = _tactical_tendency(state)
    updated = _project_tactical_tendency(state, player, target, ships)
    tendency.clear()
    tendency.update(updated)


def _commit_targeted_move(state, player, source, target, moves, reserved, angle, ships):
    sent = _add_move(moves, reserved, source, angle, ships)
    if sent > 0:
        state.setdefault("turn_claimed_target_ids", set()).add(int(target.id))
        _record_tactical_move(state, player, target, sent)
    return sent


def _project_planets_after_capture(planets, player, source, target, ships):
    projected = []
    for planet in planets:
        if planet.id == source.id:
            projected.append(
                Planet(
                    int(planet.id),
                    int(planet.owner),
                    float(planet.x),
                    float(planet.y),
                    float(planet.radius),
                    max(0, int(planet.ships) - int(ships)),
                    int(planet.production),
                )
            )
            continue
        if planet.id == target.id:
            projected.append(
                Planet(
                    int(planet.id),
                    int(player),
                    float(planet.x),
                    float(planet.y),
                    float(planet.radius),
                    max(1, int(ships) - int(target.ships)),
                    int(planet.production),
                )
            )
            continue
        projected.append(planet)
    return projected


def _source_role_for_model(state, player, source, planets, fleets):
    if source is None:
        return "unknown"
    if int(source.id) == int(state.get("primary_anchor_id") or -1):
        return "anchor"
    if int(source.id) == int(state.get("attacker_planet_id") or -1):
        return "attacker"
    if int(source.id) == int(state.get("static_collector_id") or -1):
        return "feeder"

    role = _our_roles(planets, player, fleets=fleets, state=state).get(int(source.id))
    if role:
        return role
    return "expander"


def _phase_name_for_model(state, player, planets, fleets, target=None, primary_anchor_id=None):
    owned = [planet for planet in planets if int(planet.owner) == int(player)]
    if len(owned) < 2 and not state.get("opened"):
        return "initiation"

    established = _operationally_established_quadrants(planets, player)
    if not established:
        return "expansion"

    if target is not None and int(target.owner) not in (player, NEUTRAL):
        return "attack"

    control_half = _current_control_half(
        state,
        player,
        planets,
        fleets=fleets,
        primary_anchor_id=primary_anchor_id,
    )
    enemy_in_half = any(
        int(planet.owner) not in (player, NEUTRAL)
        and (control_half is None or _quadrant(planet) in control_half)
        for planet in planets
    )
    if enemy_in_half:
        return "attack"
    return "established"


def _predict_tactical_value(state, player, planets, fleets, angular_velocity, step, source, target, ships):
    model = _load_tactical_model()
    if (
        model is None
        or target is None
        or int(ships) <= 0
        or np is None
        or _tf_action_feature_vector_for_state is None
        or _tf_action_penalty_profile_for_state is None
    ):
        return None

    measurement = _attack_measurement(source, target, ships, angular_velocity, planets=planets)
    if measurement is None:
        return None
    if not measurement.clear:
        return 0.0

    source_role = _source_role_for_model(state, player, source, planets, fleets)
    phase_name = _phase_name_for_model(
        state,
        player,
        planets,
        fleets,
        target=target,
        primary_anchor_id=state.get("primary_anchor_id"),
    )
    penalty_profile = _tf_action_penalty_profile_for_state(
        planets,
        fleets,
        player,
        source,
        target,
        ships,
        source_role=source_role,
        phase_name=phase_name,
        action_angle=measurement.angle,
    )
    features = _tf_action_feature_vector_for_state(
        planets,
        fleets,
        player,
        source,
        target,
        ships,
        step=step,
        angular_velocity=angular_velocity,
        tendency=_tactical_tendency(state),
        source_role=source_role,
        phase_name=phase_name,
        player_count=_player_count(planets, fleets, player),
        anchor_planet_id=state.get("primary_anchor_id"),
        attacker_planet_id=state.get("attacker_planet_id"),
        feeder_planet_id=state.get("static_collector_id"),
        action_angle=measurement.angle,
    )
    model_feature_count = int(model["weights"].shape[0])
    if len(features) < model_feature_count:
        return None
    # New action features are append-only so older checkpoints can safely
    # ignore them until a retrain supplies matching weights.
    if len(features) > model_feature_count:
        features = features[:model_feature_count]

    vector = np.asarray(features, dtype=np.float32)
    logits = ((vector - model["mean"]) / model["std"]) @ model["weights"] + model["bias"]
    score = _sigmoid(float(logits))
    # Keep the model's preference signal strong; the heuristic layers already
    # guard against clearly unsafe launches.
    return max(0.0, min(1.0, 0.75 * score + 0.25 * float(penalty_profile["quality_score"])))


def _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, source, candidates):
    if len(candidates) < 2 or _load_tactical_model() is None:
        return candidates

    candidates = _learning_focus_targets(state, player, planets, fleets, source, candidates)
    scored = []
    limit = len(candidates) if len(candidates) <= 32 else TACTICAL_MODEL_TOP_K
    for idx, target in enumerate(candidates[:limit]):
        need = _planned_capture_need(source, target, angular_velocity)
        score = _predict_tactical_value(state, player, planets, fleets, angular_velocity, step, source, target, need)
        if score is None:
            return candidates
        scored.append((score, idx, target))

    selected_ids = {target.id for _, _, target in scored}
    ranked = [target for _, _, target in sorted(scored, key=lambda item: (-item[0], item[1]))]
    ranked.extend([target for target in candidates if target.id not in selected_ids])
    return ranked


def _state_for(player, obs):
    state = _STATE.setdefault(
        player,
        _fresh_state(),
    )
    turn = _obs_get(obs, "step", _obs_get(obs, "turn", None))
    last_turn = state.get("last_turn")
    if turn is not None and last_turn is not None and turn < last_turn:
        state.clear()
        state.update(_fresh_state())
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


def _reset_opening_state(state, source=None):
    state["bursts"] = []
    state["turn_claimed_target_ids"] = set()
    state["opened"] = False
    state["opening_target_ids"] = []
    state["opening_launched_ids"] = []
    state["primary_anchor_id"] = None
    state["attacker_planet_id"] = None
    state["attacker_target_quadrant"] = None
    state["static_collector_id"] = None
    state["recent_static_capture_id"] = None
    state["recent_static_capture_quadrant"] = None
    if source is not None:
        state["prime_quadrant"] = _quadrant(source)


def _update_recent_static_capture_focus(state, player, planets):
    by_id = {p.id: p for p in planets}
    owned_ids = {p.id for p in planets if int(p.owner) == player}
    previous_owned_ids = state.get("previous_owned_ids")
    if previous_owned_ids is None:
        state["previous_owned_ids"] = owned_ids
        return None

    gained_static = [
        by_id[planet_id]
        for planet_id in owned_ids - set(previous_owned_ids)
        if planet_id in by_id and _is_static(by_id[planet_id])
    ]
    if gained_static:
        focus = max(
            gained_static,
            key=lambda p: (int(p.production), float(p.radius), int(p.ships), -_distance_to_center(p)),
        )
        state["recent_static_capture_id"] = focus.id
        state["recent_static_capture_quadrant"] = _quadrant(focus)

    focus = by_id.get(state.get("recent_static_capture_id"))
    if focus is None or int(focus.owner) != player or not _is_static(focus):
        state["recent_static_capture_id"] = None
        state["recent_static_capture_quadrant"] = None
        focus = None
    elif _operationally_established_for(planets, player, _quadrant(focus)):
        state["recent_static_capture_id"] = None
        state["recent_static_capture_quadrant"] = None
        focus = None

    state["previous_owned_ids"] = owned_ids
    return focus


def _recent_static_focus(state, player, planets):
    focus_id = state.get("recent_static_capture_id")
    focus_quadrant = state.get("recent_static_capture_quadrant")
    if focus_id is None or focus_quadrant is None:
        return None, None
    focus = next((p for p in planets if p.id == focus_id and int(p.owner) == player and _is_static(p)), None)
    if focus is None:
        return None, None
    return focus, int(focus_quadrant)


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
        if int(target.owner) not in (player, NEUTRAL):
            continue
        if _available(source, reserved) < min(2, burst["remaining"]):
            next_bursts.append(burst)
            continue
        amount = min(2, int(burst["remaining"]))
        if burst.get("require_attack_ready") and not _expansion_ready_for_need(source, reserved, amount):
            next_bursts.append(burst)
            continue
        measurement = _attack_measurement(source, target, amount, angular_velocity, planets=planets)
        if measurement.clear:
            sent = _commit_targeted_move(state, player, source, target, moves, reserved, measurement.angle, amount)
            burst["remaining"] -= sent
        if burst["remaining"] > 0:
            next_bursts.append(burst)
    state["bursts"] = next_bursts


def _opening_target_key(p):
    return (int(p.ships), int(p.production), float(p.radius), _distance_to_center(p))


def _opening_targets_for_quadrant(neutral_planets, quadrant):
    same_quadrant = [p for p in neutral_planets if _quadrant(p) == quadrant]
    stationary = [p for p in same_quadrant if _is_static(p)]
    return sorted(stationary or same_quadrant, key=_opening_target_key)


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


def _send_opening_payload(state, source, target, moves, reserved, angular_velocity, planets=None):
    need_now = int(target.ships) + 1
    available = _available(source, reserved)
    if available < need_now:
        return False

    payload = need_now
    measurement = _attack_measurement(source, target, payload, angular_velocity, planets=planets)
    if not measurement.clear:
        return False

    sent = _commit_targeted_move(state, int(source.owner), source, target, moves, reserved, measurement.angle, payload)
    if sent <= 0:
        return False
    return True


def _send_expansion_payload(
    state,
    source,
    target,
    moves,
    reserved,
    angular_velocity,
    need=None,
    require_attack_ready=False,
    planets=None,
    fleets=None,
    allow_thin_enemy=False,
):
    if not _same_equator_side(source, target):
        return False

    if need is None:
        need = _planned_capture_need(source, target, angular_velocity)
    player = int(source.owner)
    if int(target.owner) not in (player, NEUTRAL):
        enemy_need = _offensive_capture_need(source, target, player, planets or [], fleets or [], angular_velocity)
        need = max(int(need), int(enemy_need))
        margin = _decisive_attack_margin(source, target, player, planets or [], fleets or [], angular_velocity, ships=need)
        if not allow_thin_enemy and margin < MODEL_DECISIVE_ATTACK_MIN_MARGIN:
            return False
    if require_attack_ready and not _expansion_ready_for_need(source, reserved, need):
        return False

    available = _available(source, reserved)
    if available < need:
        return False

    payload = need
    measurement = _attack_measurement(source, target, payload, angular_velocity, planets=planets)
    if not measurement.clear:
        return False

    sent = _commit_targeted_move(state, int(source.owner), source, target, moves, reserved, measurement.angle, payload)
    if sent <= 0:
        return False

    if sent < need:
        remaining = max(0, int(need) - sent)
        if remaining:
            state.setdefault("bursts", []).append(
                {
                    "source_id": source.id,
                    "target_id": target.id,
                    "remaining": remaining,
                    "require_attack_ready": bool(require_attack_ready),
                }
            )
    return True


def _send_staggered_payload(
    state,
    source,
    target,
    moves,
    reserved,
    angular_velocity,
    need,
    planets=None,
):
    if not _same_equator_side(source, target):
        return False

    available = _available(source, reserved)
    if available <= 0:
        return False

    payload = min(int(need), available)
    measurement = _attack_measurement(source, target, payload, angular_velocity, planets=planets)
    if not measurement.clear:
        return False

    sent = _commit_targeted_move(state, int(source.owner), source, target, moves, reserved, measurement.angle, payload)
    if sent <= 0:
        return False

    if sent < int(need):
        remaining = max(0, int(need) - sent)
        if remaining:
            state.setdefault("bursts", []).append(
                {
                    "source_id": source.id,
                    "target_id": target.id,
                    "remaining": remaining,
                    "require_attack_ready": False,
                }
            )
    return True


def _should_stagger_establishment_capture(source, target, player, planets, need):
    source_quadrant = _quadrant(source)
    if _operationally_established_for(planets, player, source_quadrant):
        return False
    if int(target.owner) != NEUTRAL or _quadrant(target) != source_quadrant:
        return False
    if not _is_static(target) or _is_big(target):
        return False
    if int(need) > PRE_ESTABLISHMENT_BURST_MAX_NEED:
        return False
    return _is_quadrant_corner_node(target, source_quadrant) or _distance(source, target) <= ATTACK_SUPPORT_RADIUS


def _force_nearest_unconquered_move(
    state,
    player,
    planets,
    fleets,
    moves,
    reserved,
    angular_velocity,
    step,
    min_ships=1,
    require_attack_ready=False,
):
    targets = [p for p in planets if int(p.owner) != player]
    if not targets:
        return False

    claimed_target_ids = _claimed_target_ids(state, player, planets, fleets)
    sources = [
        p
        for p in planets
        if int(p.owner) == player and _available(p, reserved) >= int(min_ships)
    ]
    sources.sort(key=lambda p: (-_available(p, reserved), _distance_to_center(p)))

    for source in sources:
        attack_candidates = _serious_attack_targets(
            state,
            source,
            player,
            planets,
            fleets,
            reserved,
            claimed_target_ids,
            angular_velocity,
            step,
        )
        for target, need in attack_candidates[:8]:
            if _send_expansion_payload(
                state,
                source,
                target,
                moves,
                reserved,
                angular_velocity,
                need=need,
                require_attack_ready=require_attack_ready,
                planets=planets,
                fleets=fleets,
            ):
                return True

        candidates = _nearest_unconquered_targets(
            state,
            source,
            player,
            planets,
            fleets,
            claimed_target_ids,
            angular_velocity,
            step,
        )
        if not candidates:
            fallback_targets = [target for target in targets if target.id not in claimed_target_ids]
            candidates = sorted(_same_equator_targets(source, fallback_targets), key=lambda p: _smallest_target_key(source, p))
            candidates = _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, source, candidates)
        for target in candidates[:16]:
            need = _planned_capture_need(source, target, angular_velocity)
            if _should_stagger_establishment_capture(source, target, player, planets, need):
                if _send_staggered_payload(
                    state,
                    source,
                    target,
                    moves,
                    reserved,
                    angular_velocity,
                    need=need,
                    planets=planets,
                ):
                    return True
            if _send_expansion_payload(
                state,
                source,
                target,
                moves,
                reserved,
                angular_velocity,
                need=need,
                require_attack_ready=require_attack_ready,
                planets=planets,
                fleets=fleets,
            ):
                return True
    return False


def _unstick_opening_if_stalled(state, player, planets, fleets, moves, reserved, angular_velocity, step):
    if moves:
        return False

    my_planets = [p for p in planets if int(p.owner) == player]
    if len(my_planets) != 1:
        return False

    source = max(my_planets, key=lambda p: _available(p, reserved))
    if _available(source, reserved) < OPENING_STALL_SHIPS:
        return False

    _reset_opening_state(state, source)
    _initiation_phase(state, player, planets, moves, reserved, angular_velocity)
    if moves:
        return True
    return _force_nearest_unconquered_move(
        state,
        player,
        planets,
        fleets,
        moves,
        reserved,
        angular_velocity,
        step,
        min_ships=OPENING_STALL_SHIPS,
    )


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

    targets = _opening_targets_for_quadrant(neutral_planets, prime_quadrant)
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

    burst_target_ids = set(state.get("turn_claimed_target_ids", set()))
    burst_target_ids.update(int(burst["target_id"]) for burst in state.get("bursts", []))
    desired_targets = [target for target in targets if target.id not in launched_target_ids]

    for target in desired_targets:
        if len(launched_target_ids) >= 2:
            break
        if target.id in active_target_ids or target.id in burst_target_ids:
            continue
        source = max(source_pool, key=lambda p: _available(p, reserved))
        if _send_opening_payload(state, source, target, moves, reserved, angular_velocity, planets=planets):
            state.setdefault("opening_target_ids", []).append(target.id)
            state.setdefault("opening_launched_ids", []).append(target.id)
            active_target_ids.add(target.id)
            launched_target_ids.add(target.id)


def _role_ready_for(planets, owner, quadrant):
    big_static, small_static = _quadrant_corner_role_groups(planets, owner, quadrant)
    return len(big_static) >= 1 and len(small_static) >= 2


def _established_for(planets, owner, quadrant):
    anchor = _quadrant_anchor_planet(planets, quadrant)
    return anchor is not None and int(anchor.owner) == int(owner)


def _quadrant_anchor_planet(planets, quadrant):
    quadrant = int(quadrant)
    corner_nodes = _corner_nodes_for_quadrant(planets, quadrant)
    candidates = corner_nodes or [p for p in planets if _is_static(p) and _quadrant(p) == quadrant]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda p: (
            _distance_to_quadrant_corner(p, quadrant),
            -int(p.production),
            int(p.ships),
            int(p.id),
        ),
    )


def _operationally_established_for(planets, owner, quadrant):
    return _established_for(planets, owner, quadrant)


def _any_established(planets, owner):
    return any(_established_for(planets, owner, quadrant) for quadrant in range(4))


def _established_quadrants(planets, owner):
    return {quadrant for quadrant in range(4) if _established_for(planets, owner, quadrant)}


def _open_quadrants(planets, owner):
    established = _established_quadrants(planets, owner)
    return [quadrant for quadrant in range(4) if quadrant not in established]


def _operationally_any_established(planets, owner):
    return any(_operationally_established_for(planets, owner, quadrant) for quadrant in range(4))


def _operationally_established_quadrants(planets, owner):
    return {quadrant for quadrant in range(4) if _operationally_established_for(planets, owner, quadrant)}


def _operationally_open_quadrants(planets, owner):
    established = _operationally_established_quadrants(planets, owner)
    return [quadrant for quadrant in range(4) if quadrant not in established]


def _best_anchor_quadrant(player, planets, fleets=None):
    best = None
    best_score = None
    for quadrant in range(4):
        anchor = _quadrant_anchor_planet(planets, quadrant)
        if anchor is None or int(anchor.owner) != player:
            continue
        local_margin = _quadrant_control_margin(player, quadrant, planets, fleets=fleets)
        half_margin = max(
            _half_control_margin(player, (quadrant, adjacent), planets, fleets=fleets)
            for adjacent in _adjacent_quadrants(quadrant)
        )
        score = (
            half_margin,
            local_margin,
            int(anchor.production),
            int(anchor.ships),
            -_distance_to_quadrant_corner(anchor, quadrant),
        )
        if best_score is None or score > best_score:
            best_score = score
            best = quadrant
    return best


def _current_control_half(state, player, planets, fleets=None, primary_anchor_id=None, claimed_target_ids=None):
    by_id = {p.id: p for p in planets}
    anchor = by_id.get(primary_anchor_id or state.get("primary_anchor_id"))
    if anchor is None or int(anchor.owner) != player:
        best_anchor_quadrant = _best_anchor_quadrant(player, planets, fleets=fleets)
        if best_anchor_quadrant is None:
            return None
        anchor = _quadrant_anchor_planet(planets, best_anchor_quadrant)
        if anchor is None or int(anchor.owner) != player:
            return None

    anchor_quadrant = _quadrant(anchor)
    attacker_target_quadrant = state.get("attacker_target_quadrant")
    if attacker_target_quadrant in _adjacent_quadrants(anchor_quadrant):
        return (anchor_quadrant, int(attacker_target_quadrant))

    claimed_target_ids = claimed_target_ids or set()
    best_half = None
    best_score = None
    for adjacent in _adjacent_quadrants(anchor_quadrant):
        targets = _attack_quadrant_targets(planets, player, adjacent, claimed_target_ids)
        focus = targets[0] if targets else None
        score = (
            _half_control_margin(player, (anchor_quadrant, adjacent), planets, fleets=fleets),
            1 if focus is not None else 0,
            -(int(focus.ships) if focus is not None else 999),
            int(focus.production) if focus is not None else 0,
            -(_distance(anchor, focus) if focus is not None else 999.0),
        )
        if best_score is None or score > best_score:
            best_score = score
            best_half = (anchor_quadrant, adjacent)
    return best_half


def _attack_quadrant_targets(planets, player, quadrant, claimed_target_ids):
    needed = _needed_establishment_targets(planets, player, quadrant, claimed_target_ids)
    if needed:
        return needed

    candidates = [
        p
        for p in planets
        if int(p.owner) != player and _quadrant(p) == int(quadrant) and p.id not in claimed_target_ids
    ]
    return sorted(
        candidates,
        key=lambda p: (
            int(p.owner) != NEUTRAL,
            not _is_static(p),
            int(p.ships),
            -int(p.production),
            _distance_to_quadrant_corner(p, quadrant),
        ),
    )


def _attacker_stage_goal(attacker, target_quadrant):
    if attacker is None or target_quadrant is None:
        return ATTACKER_STAGE_TARGET_SHIPS
    if _quadrant(attacker) == int(target_quadrant):
        return ATTACKER_STAGE_IN_QUADRANT_SHIPS
    return ATTACKER_STAGE_TARGET_SHIPS


def _sync_attacker_stage(state, player, planets, reserved, claimed_target_ids, primary_anchor_id=None):
    if not _operationally_any_established(planets, player):
        state["attacker_planet_id"] = None
        state["attacker_target_quadrant"] = None
        return None, None

    by_id = {p.id: p for p in planets}
    anchor = by_id.get(primary_anchor_id or state.get("primary_anchor_id"))
    if anchor is None or int(anchor.owner) != player:
        prime_quadrant = _choose_prime_quadrant(state, [p for p in planets if int(p.owner) == player])
        anchor = _quadrant_anchor_planet(planets, prime_quadrant)
        if anchor is None or int(anchor.owner) != player:
            state["attacker_planet_id"] = None
            state["attacker_target_quadrant"] = None
            return None, None

    anchor_quadrant = _quadrant(anchor)
    control_half = _current_control_half(
        state,
        player,
        planets,
        primary_anchor_id=primary_anchor_id,
        claimed_target_ids=claimed_target_ids,
    )
    if not control_half:
        state["attacker_planet_id"] = None
        state["attacker_target_quadrant"] = None
        return None, None

    open_quadrants = [quadrant for quadrant in control_half if quadrant != anchor_quadrant and not _operationally_established_for(planets, player, quadrant)]
    if not open_quadrants:
        state["attacker_planet_id"] = None
        state["attacker_target_quadrant"] = None
        return None, None

    target_quadrant = state.get("attacker_target_quadrant")
    current_targets = (
        _attack_quadrant_targets(planets, player, target_quadrant, claimed_target_ids)
        if target_quadrant in open_quadrants
        else []
    )
    if target_quadrant not in open_quadrants or not current_targets:
        best_quadrant = None
        best_score = None
        for quadrant in open_quadrants:
            targets = _attack_quadrant_targets(planets, player, quadrant, claimed_target_ids)
            if not targets:
                continue
            focus = targets[0]
            score = (
                _half_control_margin(player, control_half, planets),
                int(focus.owner) != NEUTRAL,
                -int(focus.ships),
                int(focus.production),
                -(0 if _same_equator_side(anchor, focus) else 1),
                -_distance(anchor, focus),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_quadrant = quadrant
        target_quadrant = best_quadrant if best_quadrant is not None else open_quadrants[0]
        state["attacker_target_quadrant"] = target_quadrant
        current_targets = _attack_quadrant_targets(planets, player, target_quadrant, claimed_target_ids)

    if target_quadrant is None or not current_targets:
        state["attacker_planet_id"] = None
        return None, None

    focus = current_targets[0]
    current_attacker = by_id.get(state.get("attacker_planet_id"))
    if (
        current_attacker is not None
        and int(current_attacker.owner) == player
        and current_attacker.id != primary_anchor_id
    ):
        return target_quadrant, current_attacker

    candidates = [
        p
        for p in planets
        if int(p.owner) == player and p.id != primary_anchor_id
    ]
    if not candidates:
        state["attacker_planet_id"] = None
        return target_quadrant, None

    attacker = min(
        candidates,
        key=lambda p: (
            _quadrant(p) != target_quadrant,
            0 if _same_equator_side(p, focus) else 1,
            _quadrant_distance(_quadrant(p), target_quadrant),
            _is_big(p),
            _distance(p, focus),
            _available(p, reserved) <= 0,
            -_available(p, reserved),
        ),
    )
    state["attacker_planet_id"] = attacker.id
    return target_quadrant, attacker


def _feed_attacker(
    state,
    player,
    source,
    planets,
    moves,
    reserved,
    angular_velocity,
    claimed_target_ids,
    keep_ships,
    min_batch,
    primary_anchor_id=None,
):
    target_quadrant, attacker = _sync_attacker_stage(
        state,
        player,
        planets,
        reserved,
        claimed_target_ids,
        primary_anchor_id=primary_anchor_id,
    )
    if attacker is None or target_quadrant is None or attacker.id == source.id:
        return False

    if _operationally_established_for(planets, player, target_quadrant):
        return False

    transferable = _available(source, reserved) - int(keep_ships)
    if transferable <= 0:
        return False

    desired_total = _attacker_stage_goal(attacker, target_quadrant)
    deficit = max(0, int(desired_total) - int(attacker.ships))
    if deficit <= 0:
        return False

    amount = min(transferable, deficit)
    if amount < int(min_batch):
        return False

    measurement = _attack_measurement(source, attacker, amount, angular_velocity, planets=planets)
    if not measurement.clear:
        return False

    return _commit_targeted_move(state, player, source, attacker, moves, reserved, measurement.angle, amount) > 0


def _quadrant_distance(a, b):
    clockwise = (int(a) - int(b)) % 4
    counter_clockwise = (int(b) - int(a)) % 4
    return min(clockwise, counter_clockwise)


def _target_quadrant_rank(source_quadrant, target_quadrant, established_quadrants):
    source_quadrant = int(source_quadrant)
    target_quadrant = int(target_quadrant)
    source_established = source_quadrant in established_quadrants
    target_established = target_quadrant in established_quadrants

    if not source_established:
        if target_quadrant == source_quadrant:
            return 0
        if not target_established:
            return 2 + _quadrant_distance(source_quadrant, target_quadrant)
        return 8 + _quadrant_distance(source_quadrant, target_quadrant)

    if not target_established:
        return _quadrant_distance(source_quadrant, target_quadrant)
    if target_quadrant == source_quadrant:
        return 6
    return 7 + _quadrant_distance(source_quadrant, target_quadrant)


def _our_roles(planets, player, fleets=None, state=None):
    if _tf_infer_roles is not None and fleets is not None:
        try:
            return _tf_infer_roles(
                planets,
                fleets,
                player,
                anchor_planet_id=(state or {}).get("primary_anchor_id"),
                attacker_planet_id=(state or {}).get("attacker_planet_id"),
                feeder_planet_id=(state or {}).get("static_collector_id"),
            )
        except Exception:
            pass

    roles = {}
    for quadrant in range(4):
        if not _role_ready_for(planets, player, quadrant):
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


def _reactive_trap(player, planets, fleets, moves, reserved, blocked_source_ids=None, angular_velocity=0.0):
    blocked_source_ids = blocked_source_ids or set()
    responded = set()
    established = _operationally_any_established(planets, player)
    for eta, fleet, target in _incoming_threats(player, planets, fleets):
        if target.id in responded or target.id in blocked_source_ids:
            continue
        needed = int(fleet.ships) + 1
        available = _available(target, reserved)

        trap_angle = float(fleet.angle) + math.pi
        first_hit = _first_planet_on_ray(target, trap_angle, planets)
        if first_hit is None or int(first_hit.owner) == player:
            continue
        if int(first_hit.owner) == NEUTRAL:
            needed = max(needed, _planned_capture_need(target, first_hit, angular_velocity))
        else:
            needed = max(
                needed,
                _offensive_capture_need(target, first_hit, player, planets, fleets, angular_velocity),
            )
        if available < needed:
            continue

        if not established:
            if eta > PRE_ESTABLISHMENT_TRAP_MAX_ETA:
                continue
            if available - needed < PRE_ESTABLISHMENT_TRAP_SPARE_SHIPS:
                continue

        sent = _add_move(moves, reserved, target, trap_angle, needed)
        if sent > 0:
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


def _try_capture(
    state,
    player,
    source,
    target,
    moves,
    reserved,
    angular_velocity,
    max_payload=None,
    require_attack_ready=False,
    planets=None,
    fleets=None,
):
    need = _capture_need(source, target, angular_velocity)
    if int(target.owner) not in (int(player), NEUTRAL):
        need = max(need, _offensive_capture_need(source, target, player, planets or [], fleets or [], angular_velocity))
    if max_payload is not None:
        need = min(need, int(max_payload))
    if require_attack_ready and not _capture_ready_for_need(source, reserved, need):
        return False

    if _available(source, reserved) < need:
        return False
    measurement = _attack_measurement(source, target, need, angular_velocity, planets=planets)
    if not measurement.clear:
        return False
    return _commit_targeted_move(state, player, source, target, moves, reserved, measurement.angle, need) > 0


def _static_collector_source(state, player, planets, reserved, primary_anchor_id=None):
    established_quadrants = _operationally_established_quadrants(planets, player)
    if not established_quadrants:
        state["static_collector_id"] = None
        return None

    by_id = {p.id: p for p in planets}
    current = by_id.get(state.get("static_collector_id"))
    if (
        current is not None
        and int(current.owner) == player
        and _is_static(current)
        and _quadrant(current) in established_quadrants
        and current.id != primary_anchor_id
    ):
        return current

    cx, _ = _center_xy()
    candidates = [
        p
        for p in planets
        if int(p.owner) == player
        and _is_static(p)
        and _quadrant(p) in established_quadrants
        and p.id != primary_anchor_id
    ]
    if not candidates:
        candidates = [
            p
            for p in planets
            if int(p.owner) == player
            and _is_static(p)
            and _quadrant(p) in established_quadrants
        ]
    if not candidates:
        state["static_collector_id"] = None
        return None

    collector = min(
        candidates,
        key=lambda p: (
            _available(p, reserved) <= 0,
            _is_big(p),
            abs(float(p.x) - cx),
            _distance_to_center(p),
            -int(p.ships),
        ),
    )
    state["static_collector_id"] = collector.id
    return collector


def _static_collector_targets(source, player, planets, claimed_target_ids):
    return [
        p
        for p in planets
        if int(p.owner) != player
        and _is_static(p)
        and p.id not in claimed_target_ids
        and _same_equator_side(source, p)
    ]


def _feeder_logic(state, player, planets, fleets, moves, reserved, angular_velocity, step, primary_anchor_id=None):
    collector = _static_collector_source(
        state,
        player,
        planets,
        reserved,
        primary_anchor_id=primary_anchor_id,
    )
    if collector is None or _available(collector, reserved) <= 0:
        return

    claimed_target_ids = _claimed_target_ids(state, player, planets, fleets)
    if _feed_attacker(
        state,
        player,
        collector,
        planets,
        moves,
        reserved,
        angular_velocity,
        claimed_target_ids,
        keep_ships=16 if _is_big(collector) else 10,
        min_batch=ATTACKER_STAGE_MIN_FEED,
        primary_anchor_id=primary_anchor_id,
    ):
        return

    targets = _static_collector_targets(collector, player, planets, claimed_target_ids)
    if not targets:
        return

    assault_targets = _established_static_assault_targets(
        state,
        collector,
        player,
        planets,
        fleets,
        reserved,
        claimed_target_ids,
        angular_velocity,
        step,
    )
    for target, need in assault_targets[:8]:
        if _send_expansion_payload(
            state,
            collector,
            target,
            moves,
            reserved,
            angular_velocity,
            need=need,
            require_attack_ready=True,
            planets=planets,
            fleets=fleets,
        ):
            return

    control_half = _current_control_half(
        state,
        player,
        planets,
        fleets=fleets,
        primary_anchor_id=primary_anchor_id,
        claimed_target_ids=claimed_target_ids,
    )
    established_quadrants = _operationally_established_quadrants(planets, player)
    open_quadrants = (set(control_half) if control_half else set(range(4))) - established_quadrants
    needed_ids = set()
    for quadrant in open_quadrants:
        needed_ids.update(p.id for p in _needed_establishment_targets(planets, player, quadrant, claimed_target_ids))

    source_quadrant = _quadrant(collector)

    def score(target):
        profile = _planet_profile(target, player=player, planets=planets)
        return (
            profile.quadrant not in open_quadrants,
            target.id not in needed_ids if needed_ids else False,
            "neutral" not in profile.labels,
            "easy" not in profile.labels,
            int(target.ships),
            _quadrant_distance(source_quadrant, profile.quadrant),
            _distance(collector, target),
            -int(target.production),
        )

    ranked_targets = sorted(targets, key=score)
    ranked_targets = _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, collector, ranked_targets)
    for target in ranked_targets[:16]:
        need = _planned_capture_need(collector, target, angular_velocity)
        if _send_expansion_payload(
            state,
            collector,
            target,
            moves,
            reserved,
            angular_velocity,
            need=need,
            require_attack_ready=True,
            planets=planets,
            fleets=fleets,
        ):
            return


def _role_actions(state, player, planets, fleets, moves, reserved, roles, angular_velocity, step, blocked_source_ids=None):
    blocked_source_ids = blocked_source_ids or set()
    by_id = {p.id: p for p in planets}
    _, cy = _center_xy()
    control_half = _current_control_half(state, player, planets, fleets=fleets)
    open_quadrant_set = set(_operationally_open_quadrants(planets, player))
    if control_half:
        open_quadrant_set &= set(control_half)
    for planet_id, role in roles.items():
        source = by_id.get(planet_id)
        if source is None or int(source.owner) != player:
            continue
        if source.id in blocked_source_ids:
            continue
        targets = [p for p in planets if int(p.owner) != player]
        local_targets = _same_equator_targets(source, targets)
        expansion_targets = [p for p in local_targets if _quadrant(p) in open_quadrant_set] or local_targets
        if role == "sweeper" and _source_can_attempt_capture(source, reserved):
            equator_targets = [p for p in expansion_targets if abs(float(p.y) - cy) <= 16.0]
            ranked_targets = sorted(equator_targets, key=lambda p: _smallest_target_key(source, p))
            ranked_targets = _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, source, ranked_targets)
            for target in ranked_targets[:16]:
                if _try_capture(
                    state,
                    player,
                    source,
                    target,
                    moves,
                    reserved,
                    angular_velocity,
                    require_attack_ready=True,
                    planets=planets,
                    fleets=fleets,
                ):
                    break
        elif role == "shield" and _source_can_attempt_capture(source, reserved):
            ranked_targets = sorted(expansion_targets, key=lambda p: _smallest_target_key(source, p))
            ranked_targets = _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, source, ranked_targets)
            for target in ranked_targets[:16]:
                if int(target.owner) not in (player, NEUTRAL):
                    need = _offensive_capture_need(source, target, player, planets, fleets, angular_velocity)
                    if _send_expansion_payload(
                        state,
                        source,
                        target,
                        moves,
                        reserved,
                        angular_velocity,
                        need=need,
                        require_attack_ready=True,
                        planets=planets,
                        fleets=fleets,
                    ):
                        break
                    continue
                amount = min(30, max(20, _available(source, reserved) - 12))
                if amount <= 0:
                    continue
                measurement = _attack_measurement(source, target, amount, angular_velocity, planets=planets)
                if measurement.clear and _commit_targeted_move(state, player, source, target, moves, reserved, measurement.angle, amount) > 0:
                    break
        elif role == "battery":
            # Batteries only act in striker mode; otherwise they stockpile 60-100 ships.
            continue


def _enemy_established(planets, enemy_owner, quadrant):
    if enemy_owner == NEUTRAL:
        return False
    return _operationally_established_for(planets, enemy_owner, quadrant)


def _striker_mode(state, player, planets, fleets, moves, reserved, roles, angular_velocity, step, primary_anchor_id=None):
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
        if _available(battery, reserved) < 60 or not _source_can_attempt_capture(battery, reserved):
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
        ranked_targets = sorted(enemy_targets, key=lambda p: (int(p.ships), _distance(battery, p)))
        ranked_targets = _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, battery, ranked_targets)
        for target in ranked_targets[:16]:
            direct_need = _offensive_capture_need(battery, target, player, planets, fleets, angular_velocity)
            if direct_need <= _available(battery, reserved):
                direct_measurement = _attack_measurement(battery, target, direct_need, angular_velocity, planets=planets)
                if direct_measurement.clear:
                    if _commit_targeted_move(
                        state,
                        player,
                        battery,
                        target,
                        moves,
                        reserved,
                        direct_measurement.angle,
                        direct_need,
                    ) > 0:
                        break

            probe = min(probes, key=lambda p: (_distance(p, target), _distance(battery, p)))
            combined = _available(battery, reserved) + _available(probe, reserved)
            if combined < 2 * int(target.ships) + 1:
                continue
            if probe.id != battery.id and _available(battery, reserved) >= 60:
                payload = min(_available(battery, reserved) - 20, max(40, int(target.ships) + 15))
                if payload > 0:
                    measurement = _attack_measurement(battery, probe, payload, angular_velocity, planets=planets)
                    if measurement.clear:
                        _commit_targeted_move(state, player, battery, probe, moves, reserved, measurement.angle, payload)
            if _available(probe, reserved) >= int(target.ships) + 1 and _try_capture(
                state,
                player,
                probe,
                target,
                moves,
                reserved,
                angular_velocity,
                require_attack_ready=True,
                planets=planets,
                fleets=fleets,
            ):
                break


def _update_primary_anchor(state, player, planets):
    if not _operationally_any_established(planets, player):
        state["primary_anchor_id"] = None
        return None

    best_anchor_quadrant = _best_anchor_quadrant(player, planets)
    anchor = _quadrant_anchor_planet(planets, best_anchor_quadrant) if best_anchor_quadrant is not None else None
    if anchor is None or int(anchor.owner) != player:
        state["primary_anchor_id"] = None
        return None

    state["primary_anchor_id"] = anchor.id
    return anchor


def _primary_anchor_in_battery_mode(state, player, planets):
    anchor = _update_primary_anchor(state, player, planets)
    if anchor is None:
        return None
    if _operationally_established_for(planets, player, _quadrant(anchor)):
        return anchor
    return None


def _primary_anchor_action(state, player, planets, moves, reserved, angular_velocity):
    anchor = _primary_anchor_in_battery_mode(state, player, planets)
    if anchor is None:
        return

    claimed_target_ids = _claimed_target_ids(state, player, planets, [])
    if _feed_attacker(
        state,
        player,
        anchor,
        planets,
        moves,
        reserved,
        angular_velocity,
        claimed_target_ids,
        keep_ships=32 if _is_big(anchor) else 18,
        min_batch=ATTACKER_STAGE_BIG_FEED if _is_big(anchor) else ATTACKER_STAGE_MIN_FEED,
        primary_anchor_id=anchor.id,
    ):
        return

    anchor_quadrant = _quadrant(anchor)
    _, small_nodes = _quadrant_corner_role_groups(planets, player, anchor_quadrant)
    small_nodes = [p for p in small_nodes if p.id != anchor.id]
    if not small_nodes or _available(anchor, reserved) <= 60:
        return

    reinforcement_needs = [
        (planet, _small_node_hold_level(player, planet, planets))
        for planet in small_nodes
    ]
    candidates = [
        (planet, hold_level)
        for planet, hold_level in reinforcement_needs
        if int(planet.ships) < hold_level
    ]
    if not candidates:
        return

    target, hold_level = min(
        candidates,
        key=lambda item: (
            int(item[0].ships) - int(item[1]),
            int(item[0].ships),
            _distance(anchor, item[0]),
        ),
    )
    amount = min(_available(anchor, reserved) - 60, max(0, int(hold_level) - int(target.ships)))
    if amount <= 0:
        return

    measurement = _attack_measurement(anchor, target, amount, angular_velocity, planets=planets)
    if measurement.clear:
        _commit_targeted_move(state, player, anchor, target, moves, reserved, measurement.angle, amount)


def _needed_establishment_targets(planets, player, quadrant, claimed_target_ids):
    if _operationally_established_for(planets, player, quadrant):
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

    small_targets = sorted(
        [p for p in corner_targets if big_production is None or int(p.production) < big_production],
        key=lambda p: (int(p.ships), _distance_to_quadrant_corner(p, quadrant), _distance_to_center(p)),
    )
    big_targets = sorted(
        [p for p in corner_targets if big_production is not None and int(p.production) == big_production],
        key=lambda p: (int(p.ships), _distance_to_quadrant_corner(p, quadrant), _distance_to_center(p)),
    )

    if len(owned_smalls) < 2:
        return small_targets or corner_targets
    if big_targets and not owned_bigs:
        return big_targets
    return small_targets + big_targets if (small_targets or big_targets) else corner_targets


def _pre_establishment_expansion_targets(source, player, planets, claimed_target_ids):
    candidates = [
        p
        for p in planets
        if int(p.owner) == NEUTRAL
        and p.id not in claimed_target_ids
        and p.id != int(source.id)
    ]
    if not candidates:
        return []

    source_quadrant = _quadrant(source)

    def score(target):
        target_quadrant = _quadrant(target)
        return (
            target_quadrant != source_quadrant,
            _quadrant_distance(source_quadrant, target_quadrant),
            0 if _same_equator_side(source, target) else 1,
            int(target.ships),
            _distance(source, target),
            _distance_to_quadrant_corner(target, target_quadrant),
            -int(target.production),
            -float(target.radius),
        )

    return sorted(candidates, key=score)


def _pre_establishment_expansion(state, player, planets, fleets, moves, reserved, angular_velocity, step):
    claimed_target_ids = _claimed_target_ids(state, player, planets, fleets)
    sources = sorted(
        [p for p in planets if int(p.owner) == player and _available(p, reserved) > 0],
        key=lambda p: (-_available(p, reserved), _quadrant(p), _distance_to_center(p)),
    )
    for source in sources:
        candidates = _pre_establishment_expansion_targets(source, player, planets, claimed_target_ids)
        candidates = _rerank_targets_with_model(
            state,
            player,
            planets,
            fleets,
            angular_velocity,
            step,
            source,
            candidates,
        )
        for target in candidates[:16]:
            need = _planned_capture_need(source, target, angular_velocity)
            if _available(source, reserved) < need:
                continue
            if _send_expansion_payload(
                state,
                source,
                target,
                moves,
                reserved,
                angular_velocity,
                need=need,
                require_attack_ready=False,
                planets=planets,
                fleets=fleets,
            ):
                claimed_target_ids.add(target.id)
                break


def _attacker_stage_action(
    state,
    player,
    planets,
    fleets,
    moves,
    reserved,
    angular_velocity,
    step,
    primary_anchor_id=None,
):
    claimed_target_ids = _claimed_target_ids(state, player, planets, fleets)
    target_quadrant, attacker = _sync_attacker_stage(
        state,
        player,
        planets,
        reserved,
        claimed_target_ids,
        primary_anchor_id=primary_anchor_id,
    )
    if attacker is None or target_quadrant is None:
        return

    available = _available(attacker, reserved)
    if available <= 0:
        return

    targets = _attack_quadrant_targets(planets, player, target_quadrant, claimed_target_ids)
    if not targets:
        return

    if _quadrant(attacker) != target_quadrant and available < _attacker_stage_goal(attacker, target_quadrant):
        return

    ranked_targets = _rerank_targets_with_model(
        state,
        player,
        planets,
        fleets,
        angular_velocity,
        step,
        attacker,
        targets,
    )
    for target in ranked_targets[:12]:
        need = (
            _offensive_capture_need(attacker, target, player, planets, fleets, angular_velocity)
            if int(target.owner) not in (player, NEUTRAL)
            else _planned_capture_need(attacker, target, angular_velocity)
        )
        if available < need:
            continue
        if int(target.owner) == NEUTRAL and _should_stagger_establishment_capture(attacker, target, player, planets, need):
            if _send_staggered_payload(
                state,
                attacker,
                target,
                moves,
                reserved,
                angular_velocity,
                need=need,
                planets=planets,
            ):
                return
        elif _send_expansion_payload(
            state,
            attacker,
            target,
            moves,
            reserved,
            angular_velocity,
            need=need,
            require_attack_ready=False,
            planets=planets,
            fleets=fleets,
        ):
            return


def _recent_static_focus_targets(state, player, planets, fleets, claimed_target_ids, angular_velocity, step):
    focus, focus_quadrant = _recent_static_focus(state, player, planets)
    if focus is None:
        return []

    quadrant_targets = [
        p
        for p in planets
        if int(p.owner) != player and _quadrant(p) == focus_quadrant
    ]
    if not quadrant_targets:
        state["recent_static_capture_id"] = None
        state["recent_static_capture_quadrant"] = None
        return []

    candidates = [p for p in quadrant_targets if p.id not in claimed_target_ids]
    def score(p):
        profile = _planet_profile(p, player=player, planets=planets)
        return (
            "easy" not in profile.labels,
            "neutral" not in profile.labels,
            int(p.ships),
            -int(p.production),
            -float(p.radius),
            _distance(focus, p),
            profile.corner_distance,
        )

    ranked = sorted(candidates, key=score)
    return _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, focus, ranked)


def _serious_attack_targets(state, source, player, planets, fleets, reserved, claimed_target_ids, angular_velocity, step):
    available = _available(source, reserved)
    if available < ATTACK_MIN_FRONTLINE_SHIPS and not _is_large_production(source):
        return []

    source_quadrant = _quadrant(source)
    attacker_id = state.get("attacker_planet_id")
    control_half = _current_control_half(state, player, planets, fleets=fleets, claimed_target_ids=claimed_target_ids)
    established_quadrants = _operationally_established_quadrants(planets, player)
    source_established = source_quadrant in established_quadrants
    candidates = [
        p
        for p in planets
        if int(p.owner) not in (player, NEUTRAL)
        and p.id not in claimed_target_ids
        and _same_equator_side(source, p)
        and (_quadrant_distance(source_quadrant, _quadrant(p)) <= 1 or available >= 44)
    ]
    if not candidates:
        return []

    if control_half:
        if source.id == attacker_id:
            candidates = [p for p in candidates if _quadrant(p) in control_half]
        else:
            candidates = [p for p in candidates if _quadrant(p) == source_quadrant]
        if not candidates:
            return []

    if not source_established:
        same_quadrant = [p for p in candidates if _quadrant(p) == source_quadrant]
        if same_quadrant:
            candidates = same_quadrant

    ranked = []
    for target in candidates:
        need = _offensive_capture_need(source, target, player, planets, fleets, angular_velocity)
        if need > available:
            continue
        friendly_support, enemy_support = _local_support_totals(player, target, planets, fleets=fleets)
        support_gap = friendly_support - enemy_support
        target_quadrant = _quadrant(target)
        our_quadrant_total, enemy_quadrant_total = _quadrant_totals(player, target_quadrant, planets, fleets=fleets)
        quadrant_gap = our_quadrant_total - enemy_quadrant_total
        distance = _distance(source, target)
        need_ratio = need / max(1.0, float(available))
        enemy_established = _enemy_established(planets, int(target.owner), target_quadrant)

        score = 0.0
        score += 52.0
        score += 15.0 * float(target.production)
        score += 14.0 if _is_big(target) else 4.0
        score += 8.0 if _is_static(target) else 2.0
        score += 10.0 if target_quadrant == source_quadrant else 4.0
        score += 12.0 if enemy_established else 0.0
        score += 0.22 * support_gap
        score += 0.08 * quadrant_gap
        score += 0.35 * max(0, available - need)
        score -= 1.15 * distance
        score -= 34.0 * need_ratio
        score -= 0.55 * int(target.ships)

        if not source_established and support_gap < -6 and not _is_big(target):
            continue
        if available - need < 0:
            continue
        ranked.append((score, target, need))

    if not ranked:
        return []

    ranked.sort(key=lambda item: (-item[0], item[2], _distance(source, item[1])))
    ranked_targets = [target for _, target, _ in ranked]
    ranked_targets = _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, source, ranked_targets)
    need_by_id = {target.id: need for _, target, need in ranked}
    return [(target, need_by_id[target.id]) for target in ranked_targets if target.id in need_by_id]


def _established_static_assault_targets(
    state,
    source,
    player,
    planets,
    fleets,
    reserved,
    claimed_target_ids,
    angular_velocity,
    step,
):
    source_quadrant = _quadrant(source)
    if not _operationally_established_for(planets, player, source_quadrant):
        return []

    available = _available(source, reserved)
    if available < _attack_ready_ships(source):
        return []

    attacker_id = state.get("attacker_planet_id")
    control_half = _current_control_half(state, player, planets, fleets=fleets, claimed_target_ids=claimed_target_ids)
    ranked = []
    for target in planets:
        if int(target.owner) in (player, NEUTRAL):
            continue
        if not _is_static(target) or target.id in claimed_target_ids:
            continue
        if not _same_equator_side(source, target):
            continue

        target_quadrant = _quadrant(target)
        if control_half and target_quadrant not in control_half:
            continue
        if source.id != attacker_id and target_quadrant != source_quadrant:
            continue
        quadrant_distance = _quadrant_distance(source_quadrant, target_quadrant)
        if quadrant_distance > 1:
            continue

        distance = _distance(source, target)
        distance_limit = ESTABLISHED_STATIC_ASSAULT_RADIUS if target_quadrant == source_quadrant else ATTACK_LONG_RANGE_RADIUS
        if distance > distance_limit:
            continue

        need = _offensive_capture_need(source, target, player, planets, fleets, angular_velocity)
        if need > available:
            continue

        friendly_support, enemy_support = _local_support_totals(player, target, planets, fleets=fleets)
        support_gap = friendly_support - enemy_support
        our_quadrant_total, enemy_quadrant_total = _quadrant_totals(player, target_quadrant, planets, fleets=fleets)
        quadrant_gap = our_quadrant_total - enemy_quadrant_total
        enemy_established = _enemy_established(planets, int(target.owner), target_quadrant)

        score = 80.0
        score += 22.0 if target_quadrant == source_quadrant else 11.0
        score += 18.0 if _is_corner_node(target) else 0.0
        score += 16.0 * float(target.production)
        score += 8.0 if enemy_established else 0.0
        score += 0.28 * support_gap
        score += 0.12 * quadrant_gap
        score += 0.40 * max(0, available - need)
        score -= 1.45 * distance
        score -= 0.80 * int(target.ships)

        ranked.append((score, target, need))

    if not ranked:
        return []

    ranked.sort(key=lambda item: (-item[0], item[2], _distance(source, item[1])))
    ranked_targets = [target for _, target, _ in ranked]
    ranked_targets = _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, source, ranked_targets)
    need_by_id = {target.id: need for _, target, need in ranked}
    return [(target, need_by_id[target.id]) for target in ranked_targets if target.id in need_by_id]


def _nearest_unconquered_targets(state, source, player, planets, fleets, claimed_target_ids, angular_velocity, step):
    candidates = [
        p
        for p in planets
        if int(p.owner) != player
        and p.id not in claimed_target_ids
        and _same_equator_side(source, p)
    ]
    if not candidates:
        return []

    source_quadrant = _quadrant(source)
    attacker_id = state.get("attacker_planet_id")
    control_half = _current_control_half(state, player, planets, fleets=fleets, claimed_target_ids=claimed_target_ids)
    established_quadrants = _operationally_established_quadrants(planets, player)
    source_established = source_quadrant in established_quadrants
    focus_targets = _recent_static_focus_targets(state, player, planets, fleets, claimed_target_ids, angular_velocity, step)
    focus_targets = _same_equator_targets(source, focus_targets)
    if focus_targets and (source_established or _quadrant(focus_targets[0]) == source_quadrant):
        return _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, source, focus_targets)

    if not source_established:
        establishment_targets = _needed_establishment_targets(planets, player, source_quadrant, claimed_target_ids)
        establishment_targets = _same_equator_targets(source, establishment_targets)
        if establishment_targets:
            candidates = establishment_targets
        else:
            same_quadrant_targets = [p for p in candidates if _quadrant(p) == source_quadrant]
            if same_quadrant_targets:
                candidates = same_quadrant_targets
    else:
        open_quadrant_set = set(_operationally_open_quadrants(planets, player))
        if control_half:
            open_quadrant_set &= set(control_half)
        open_targets = [p for p in candidates if _quadrant(p) in open_quadrant_set]
        if open_targets:
            candidates = open_targets

    if not candidates:
        candidates = [
            p
            for p in planets
            if int(p.owner) != player
            and p.id not in claimed_target_ids
            and _same_equator_side(source, p)
        ]

    if source_established and control_half:
        if source.id == attacker_id:
            candidates = [p for p in candidates if _quadrant(p) in control_half]
        else:
            candidates = [
                p
                for p in candidates
                if _quadrant(p) in control_half and (int(p.owner) == NEUTRAL or _quadrant(p) == source_quadrant)
            ]

    if not source_established:
        same_quadrant_targets = [p for p in candidates if _quadrant(p) == source_quadrant]
        if same_quadrant_targets:
            candidates = same_quadrant_targets

    def score(p):
        profile = _planet_profile(p, player=player, planets=planets)
        return (
            _target_quadrant_rank(source_quadrant, profile.quadrant, established_quadrants),
            "easy" not in profile.labels,
            "neutral" not in profile.labels,
            int(p.ships),
            _distance(source, p),
            -int(p.production),
            profile.corner_distance,
        )

    ranked = sorted(candidates, key=score)
    return _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, source, ranked)


def _expansion(state, player, planets, fleets, moves, reserved, roles, angular_velocity, step):
    role_ids = set(roles)
    my_planets = [p for p in planets if int(p.owner) == player]
    if len(my_planets) < 2:
        return

    battery_anchor = _primary_anchor_in_battery_mode(state, player, planets)
    battery_anchor_id = battery_anchor.id if battery_anchor is not None else None
    attacker_id = state.get("attacker_planet_id")
    attacker_target_quadrant = state.get("attacker_target_quadrant")
    claimed_target_ids = _claimed_target_ids(state, player, planets, fleets)
    focus, focus_quadrant = _recent_static_focus(state, player, planets)
    established_quadrants = _operationally_established_quadrants(planets, player)

    def source_score(p):
        quadrant = _quadrant(p)
        return (
            quadrant in established_quadrants,
            focus is not None and quadrant != focus_quadrant,
            _distance(p, focus) if focus is not None else 0.0,
            _distance_to_quadrant_corner(p, quadrant),
            -int(p.ships),
        )

    for source in sorted(
        my_planets,
        key=source_score,
    ):
        source_established = _quadrant(source) in established_quadrants
        if source.id == battery_anchor_id:
            continue
        if source.id == attacker_id and attacker_target_quadrant in _operationally_open_quadrants(planets, player):
            continue
        if source.id in role_ids and source_established:
            continue
        available = _available(source, reserved)
        if available <= 0:
            continue

        if source_established:
            assault_targets = _established_static_assault_targets(
                state,
                source,
                player,
                planets,
                fleets,
                reserved,
                claimed_target_ids,
                angular_velocity,
                step,
            )
            assaulted = False
            for target, need in assault_targets[:8]:
                if _send_expansion_payload(
                    state,
                    source,
                    target,
                    moves,
                    reserved,
                    angular_velocity,
                    need=need,
                    require_attack_ready=True,
                    planets=planets,
                    fleets=fleets,
                ):
                    claimed_target_ids.add(target.id)
                    assaulted = True
                    break
            if assaulted:
                continue

        candidates = _nearest_unconquered_targets(
            state,
            source,
            player,
            planets,
            fleets,
            claimed_target_ids,
            angular_velocity,
            step,
        )
        prefer_attack = source_established or (candidates and int(candidates[0].owner) != NEUTRAL)
        if prefer_attack:
            attack_candidates = _serious_attack_targets(
                state,
                source,
                player,
                planets,
                fleets,
                reserved,
                claimed_target_ids,
                angular_velocity,
                step,
            )
            attacked = False
            for target, need in attack_candidates[:8]:
                if _send_expansion_payload(
                    state,
                    source,
                    target,
                    moves,
                    reserved,
                    angular_velocity,
                    need=need,
                    require_attack_ready=True,
                    planets=planets,
                    fleets=fleets,
                ):
                    claimed_target_ids.add(target.id)
                    attacked = True
                    break
            if attacked:
                continue

        for target in candidates[:16]:
            need = _planned_capture_need(source, target, angular_velocity)
            if not source_established and _should_stagger_establishment_capture(source, target, player, planets, need):
                if _send_staggered_payload(
                    state,
                    source,
                    target,
                    moves,
                    reserved,
                    angular_velocity,
                    need=need,
                    planets=planets,
                ):
                    claimed_target_ids.add(target.id)
                    break
            if _send_expansion_payload(
                state,
                source,
                target,
                moves,
                reserved,
                angular_velocity,
                need=need,
                require_attack_ready=True,
                planets=planets,
                fleets=fleets,
            ):
                claimed_target_ids.add(target.id)
                break


def _model_pressure_groups(player, planets, fleets):
    grouped = {}
    for eta, fleet, target in _incoming_threats(player, planets, fleets):
        grouped.setdefault(int(target.id), []).append((float(eta), fleet, target))

    pressure = []
    for items in grouped.values():
        target = items[0][2]
        incoming = sum(int(fleet.ships) for _, fleet, _ in items)
        source_ids = {int(fleet.from_planet_id) for _, fleet, _ in items if int(fleet.from_planet_id) >= 0}
        earliest = min(eta for eta, _, _ in items)
        if (
            len(source_ids) < MODEL_PRESSURE_MIN_SOURCE_COUNT
            and incoming < max(MODEL_PRESSURE_MIN_HOSTILE_SHIPS, int(target.ships) * MODEL_PRESSURE_MIN_DEFENSE_RATIO)
            and earliest > MODEL_PRESSURE_ARRIVAL_WINDOW
        ):
            continue
        pressure.append((earliest, incoming, len(source_ids), target, items))

    pressure.sort(key=lambda item: (item[0], -item[1], int(item[3].ships)))
    return pressure


def _model_source_order(state, player, planets, reserved, roles, primary_anchor_id=None):
    role_rank = {
        "attacker": 0,
        "battery": 1,
        "feeder": 2,
        "sweeper": 3,
        "shield": 4,
        "anchor": 5,
        "expander": 6,
    }

    def key(source):
        role = roles.get(int(source.id)) or _source_role_for_model(state, player, source, planets, [])
        return (
            role_rank.get(role, 7),
            int(source.id) == int(primary_anchor_id or -1),
            -_available(source, reserved),
            _distance_to_center(source),
        )

    return sorted([p for p in planets if int(p.owner) == int(player)], key=key)


def _model_pressure_counter_targets(player, planets, pressure_items, threatened_target, claimed_target_ids):
    by_id = {int(p.id): p for p in planets}
    preferred = []
    for _, fleet, _ in pressure_items:
        source = by_id.get(int(fleet.from_planet_id))
        if source is not None and int(source.owner) not in (player, NEUTRAL):
            preferred.append(source)

    fallback = [
        p
        for p in planets
        if int(p.owner) not in (player, NEUTRAL)
        and int(p.id) not in claimed_target_ids
        and _quadrant(p) == _quadrant(threatened_target)
    ]
    candidates = _prepend_unique_targets(preferred, fallback)

    return sorted(
        [p for p in candidates if int(p.id) not in claimed_target_ids],
        key=lambda p: (
            p not in preferred,
            0 if _same_equator_side(p, threatened_target) else 1,
            int(p.ships),
            -int(p.production),
            _distance(p, threatened_target),
        ),
    )


def _model_pressure_action(
    state,
    player,
    planets,
    fleets,
    moves,
    reserved,
    roles,
    angular_velocity,
    step,
    primary_anchor_id=None,
):
    claimed_target_ids = _claimed_target_ids(state, player, planets, fleets)
    sources = _model_source_order(state, player, planets, reserved, roles, primary_anchor_id=primary_anchor_id)
    for earliest, incoming, source_count, threatened, pressure_items in _model_pressure_groups(player, planets, fleets):
        counter_targets = _model_pressure_counter_targets(player, planets, pressure_items, threatened, claimed_target_ids)
        for source in sources:
            if _available(source, reserved) < ATTACK_MIN_FRONTLINE_SHIPS and not _is_large_production(source):
                continue
            for target in counter_targets:
                if not _same_equator_side(source, target):
                    continue
                need = _offensive_capture_need(source, target, player, planets, fleets, angular_velocity)
                if _available(source, reserved) < need:
                    continue
                margin = _decisive_attack_margin(source, target, player, planets, fleets, angular_velocity, ships=_available(source, reserved))
                payload = _decisive_attack_payload(_available(source, reserved), need, margin)
                if _send_expansion_payload(
                    state,
                    source,
                    target,
                    moves,
                    reserved,
                    angular_velocity,
                    need=payload,
                    require_attack_ready=True,
                    planets=planets,
                    fleets=fleets,
                ):
                    state["model_controller_action"] = "pressure_counter"
                    state["model_controller_reason"] = "pressure:%s incoming from %s sources" % (
                        int(incoming),
                        int(source_count),
                    )
                    return "pressure_counter"

        need = max(0, int(incoming) - int(threatened.ships) + 1)
        hold_need = _small_node_hold_level(player, threatened, planets) - int(threatened.ships)
        need = max(need, int(hold_need))
        if need <= 0:
            continue

        reinforcers = [p for p in sources if int(p.id) != int(threatened.id)]
        reinforcers.sort(key=lambda p: (_distance(p, threatened), -_available(p, reserved)))
        for source in reinforcers:
            keep = 18 if _is_big(source) else 8
            amount = min(max(0, _available(source, reserved) - keep), need)
            if amount < min(8, need):
                continue
            measurement = _attack_measurement(source, threatened, amount, angular_velocity, planets=planets)
            if not measurement.clear or measurement.eta > earliest + 2.5:
                continue
            if _commit_targeted_move(state, player, source, threatened, moves, reserved, measurement.angle, amount) > 0:
                state["model_controller_action"] = "pressure_reinforce"
                state["model_controller_reason"] = "pressure:%s incoming reinforced" % int(incoming)
                return "pressure_reinforce"

    return None


def _model_enemy_candidates(
    state,
    source,
    player,
    planets,
    fleets,
    reserved,
    claimed_target_ids,
    angular_velocity,
    step,
):
    candidates = []
    seen = set()

    def add_targets(items):
        for target, need in items:
            if int(target.id) in seen:
                continue
            seen.add(int(target.id))
            candidates.append((target, need))

    add_targets(
        _established_static_assault_targets(
            state,
            source,
            player,
            planets,
            fleets,
            reserved,
            claimed_target_ids,
            angular_velocity,
            step,
        )
    )
    add_targets(
        _serious_attack_targets(
            state,
            source,
            player,
            planets,
            fleets,
            reserved,
            claimed_target_ids,
            angular_velocity,
            step,
        )
    )

    source_quadrant = _quadrant(source)
    available = _available(source, reserved)
    extra_targets = [
        p
        for p in planets
        if int(p.owner) not in (player, NEUTRAL)
        and int(p.id) not in claimed_target_ids
        and int(p.id) not in seen
        and _same_equator_side(source, p)
        and (_quadrant_distance(source_quadrant, _quadrant(p)) <= 1 or available >= 70)
    ]
    extra_targets = _rerank_targets_with_model(state, player, planets, fleets, angular_velocity, step, source, extra_targets)
    for target in extra_targets[:12]:
        need = _offensive_capture_need(source, target, player, planets, fleets, angular_velocity)
        if need <= available:
            candidates.append((target, need))

    return candidates


def _trend_for_model_action(state, player, planets, fleets, target):
    if _tf_trend_identity_for_target is None:
        return "neutral"
    try:
        return _tf_trend_identity_for_target(
            planets,
            fleets,
            player,
            target,
            tendency=_tactical_tendency(state),
            player_count=_player_count(planets, fleets, player),
        )
    except Exception:
        return "neutral"


def _model_decisive_attack_action(
    state,
    player,
    planets,
    fleets,
    moves,
    reserved,
    roles,
    angular_velocity,
    step,
    primary_anchor_id=None,
):
    claimed_target_ids = _claimed_target_ids(state, player, planets, fleets)
    best = None
    for source in _model_source_order(state, player, planets, reserved, roles, primary_anchor_id=primary_anchor_id):
        available = _available(source, reserved)
        if available < ATTACK_MIN_FRONTLINE_SHIPS and not _is_large_production(source):
            continue
        for target, need in _model_enemy_candidates(
            state,
            source,
            player,
            planets,
            fleets,
            reserved,
            claimed_target_ids,
            angular_velocity,
            step,
        ):
            if available < need:
                continue
            measurement = _attack_measurement(source, target, need, angular_velocity, planets=planets)
            if not measurement.clear:
                continue
            margin = _decisive_attack_margin(source, target, player, planets, fleets, angular_velocity, ships=available)
            if margin < MODEL_DECISIVE_ATTACK_MIN_MARGIN:
                continue
            payload = _decisive_attack_payload(available, need, margin)
            model_score = _predict_tactical_value(
                state,
                player,
                planets,
                fleets,
                angular_velocity,
                step,
                source,
                target,
                payload,
            )
            if model_score is None:
                model_score = 0.5
            trend = _trend_for_model_action(state, player, planets, fleets, target)
            trend_bonus = {
                "pressured": 20.0,
                "overtake_window": 24.0,
                "cash_in": 18.0,
                "chasing_leader": 14.0,
                "neutral": 0.0,
            }.get(trend, 0.0)
            target_quadrant = _quadrant(target)
            our_total, enemy_total = _quadrant_totals(player, target_quadrant, planets, fleets=fleets)
            swing = enemy_total - our_total
            score = 100.0 * float(model_score)
            score += 42.0 * float(margin)
            score += 18.0 * float(target.production)
            score += 16.0 if _is_big(target) else 4.0
            score += 10.0 if _is_static(target) else 2.0
            score += 0.08 * float(swing)
            score += trend_bonus
            score -= 0.18 * float(need)
            score -= 0.65 * _distance(source, target)
            candidate = (score, source, target, payload, trend, model_score, margin)
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        return None

    _, source, target, payload, trend, model_score, margin = best
    if _send_expansion_payload(
        state,
        source,
        target,
        moves,
        reserved,
        angular_velocity,
        need=payload,
        require_attack_ready=True,
        planets=planets,
        fleets=fleets,
    ):
        claimed_target_ids.add(int(target.id))
        state["model_controller_action"] = "decisive_attack"
        state["model_controller_reason"] = "%s target:%s model:%.3f margin:%.3f" % (
            trend,
            int(target.id),
            float(model_score),
            float(margin),
        )
        return "decisive_attack"
    return None


def _model_controller_action(
    state,
    player,
    planets,
    fleets,
    moves,
    reserved,
    roles,
    angular_velocity,
    step,
    primary_anchor_id=None,
):
    state["model_controller_action"] = None
    state["model_controller_reason"] = "idle"
    pressure_action = _model_pressure_action(
        state,
        player,
        planets,
        fleets,
        moves,
        reserved,
        roles,
        angular_velocity,
        step,
        primary_anchor_id=primary_anchor_id,
    )
    if pressure_action is not None:
        return pressure_action
    return _model_decisive_attack_action(
        state,
        player,
        planets,
        fleets,
        moves,
        reserved,
        roles,
        angular_velocity,
        step,
        primary_anchor_id=primary_anchor_id,
    )


def _proposal_sources():
    sources = []
    if _agent_best_controller is not None:
        sources.append(("best", _agent_best_controller, "agent"))
    if _agent_intruder_controller is not None:
        sources.append(("intruder", _agent_intruder_controller, "agent"))
    if _agent_1200_controller is not None:
        sources.append(("ppo1200", _agent_1200_controller, "agent"))
    if _smith_controller is not None:
        sources.append(("smith", _smith_controller, "smith"))
    return sources


def _reset_proposal_source_memory(module):
    runtime = getattr(module, "_RUNTIME", None)
    if runtime is not None and hasattr(runtime, "reset"):
        try:
            runtime.reset()
        except Exception:
            pass


def _reset_proposal_sources_for_new_game(state, step):
    if int(step) != 0 or state.get("proposal_sources_reset_turn") == 0:
        return
    for _, module, _ in _proposal_sources():
        _reset_proposal_source_memory(module)
    state["proposal_sources_reset_turn"] = 0


def _call_proposal_source(name, module, kind, obs, config, state, player, planets, fleets, step):
    try:
        if kind == "smith":
            intent = _smith_controller_intent(state, player, planets, fleets, step)
            return module.controller_agent(obs, config=config, intent=intent)
        agent_fn = getattr(module, "agent", None)
        if agent_fn is None:
            return []
        try:
            return agent_fn(obs, config)
        except TypeError:
            return agent_fn(obs)
    except Exception:
        return []


def _proposal_target_for_move(move, planets):
    if _tf_infer_action_target is not None:
        try:
            target = _tf_infer_action_target(move, planets)
            if target is not None:
                return target
        except Exception:
            pass
    by_id = {int(p.id): p for p in planets}
    source = by_id.get(int(move[0])) if move and len(move) >= 1 else None
    if source is None:
        return None
    return _first_planet_on_ray(source, float(move[1]), planets)


def _sanitize_proposal_moves(moves, player, planets):
    by_id = {int(p.id): p for p in planets}
    reserved = {}
    cleaned = []
    invalid = 0.0
    for move in list(moves or [])[:PROPOSAL_MAX_MOVES]:
        if not move or len(move) < 3:
            invalid += 1.0
            continue
        try:
            source_id = int(move[0])
            angle = _norm_angle(float(move[1]))
            ships = int(move[2])
        except Exception:
            invalid += 1.0
            continue
        source = by_id.get(source_id)
        if source is None or int(source.owner) != int(player) or ships <= 0:
            invalid += 1.0
            continue
        amount = min(ships, _available(source, reserved))
        if amount <= 0:
            invalid += 1.0
            continue
        if amount < ships:
            invalid += 0.25 + (float(ships - amount) / max(1.0, float(ships)))
        if not _angle_clear_of_sun(source, angle):
            invalid += 1.5
            continue
        cleaned.append([source_id, angle, amount])
        reserved[source_id] = reserved.get(source_id, 0) + amount
    return cleaned, invalid


def _proposal_pressure_lookup(player, planets, fleets):
    pressure = {}
    for eta, fleet, target in _incoming_threats(player, planets, fleets):
        row = pressure.setdefault(int(target.id), {"ships": 0, "earliest": float(eta), "sources": set()})
        row["ships"] += int(fleet.ships)
        row["earliest"] = min(float(row["earliest"]), float(eta))
        if int(fleet.from_planet_id) >= 0:
            row["sources"].add(int(fleet.from_planet_id))
    return pressure


def _fallback_action_value(state, player, planets, fleets, angular_velocity, step, source, target, ships):
    if target is None or source is None or int(ships) <= 0:
        return 0.0
    if int(target.owner) == int(player):
        pressure = _proposal_pressure_lookup(player, planets, fleets).get(int(target.id))
        if pressure:
            need = max(1, int(pressure["ships"]) - int(target.ships) + 1)
            return 0.55 + min(1.0, float(ships) / float(max(1, need)))
        return 0.18
    if int(target.owner) == NEUTRAL:
        need = _planned_capture_need(source, target, angular_velocity)
        capture_bonus = 0.55 if int(ships) >= int(need) else -0.35
        return capture_bonus + 0.10 * float(target.production) + (0.08 if _is_static(target) else 0.0)
    need = _offensive_capture_need(source, target, player, planets, fleets, angular_velocity)
    margin = (float(ships) - float(need)) / max(12.0, float(need))
    return 0.55 + 0.65 * margin + 0.10 * float(target.production) + (0.12 if _is_big(target) else 0.0)


def _proposal_opportunity_value(state, player, planets, fleets, angular_velocity, step, source, target, ships):
    if source is None or target is None or int(ships) <= 0:
        return 0.0, "none"
    if int(target.owner) == int(player):
        pressure = _proposal_pressure_lookup(player, planets, fleets).get(int(target.id))
        return (0.45, "pressure-cover") if pressure else (-0.70, "maintenance")
    if int(target.owner) == NEUTRAL:
        owned_count = sum(1 for p in planets if int(p.owner) == int(player))
        need = _planned_capture_need(source, target, angular_velocity)
        if int(ships) < int(need):
            return -0.45, "thin-neutral"
        foundation_bonus = 0.75 if owned_count < 3 else 0.05
        prod_bonus = 0.08 * float(target.production)
        static_bonus = 0.10 if _is_static(target) and owned_count < 4 else 0.0
        return foundation_bonus + prod_bonus + static_bonus, "foundation" if owned_count < 3 else "neutral"

    need = _offensive_capture_need(source, target, player, planets, fleets, angular_velocity)
    margin = (float(ships) - float(need)) / max(12.0, float(need))
    if margin < -0.20:
        return -1.60, "thin-enemy"

    trend = _trend_for_model_action(state, player, planets, fleets, target)
    target_quadrant = _quadrant(target)
    our_total, enemy_total = _quadrant_totals(player, target_quadrant, planets, fleets=fleets)
    quadrant_swing = max(0.0, float(enemy_total - our_total)) / 110.0
    production_swing = float(target.production) / 5.0
    pressure_source_ships = sum(
        int(fleet.ships)
        for fleet in fleets or []
        if int(fleet.owner) not in (player, NEUTRAL) and int(fleet.from_planet_id) == int(target.id)
    )
    pressure_source_bonus = min(1.4, float(pressure_source_ships) / 65.0)
    decisive_bonus = max(0.0, min(2.0, margin * 2.4))
    structural_bonus = 0.35 if _is_big(target) else 0.0
    structural_bonus += 0.25 if _is_static(target) else 0.0
    trend_bonus = float(OPPORTUNITY_TREND_BONUS.get(trend, 0.0))
    opportunity = (
        1.05
        + trend_bonus
        + decisive_bonus
        + 0.85 * production_swing
        + 0.55 * quadrant_swing
        + pressure_source_bonus
        + structural_bonus
    )
    return opportunity, trend


def _controller_clip(value, scale=1.0, limit=4.0):
    if scale == 0:
        scale = 1.0
    scaled = float(value) / float(scale)
    return max(-float(limit), min(float(limit), scaled))


def _controller_ratio(part, whole):
    return 0.0 if float(whole) <= 0.0 else float(part) / float(whole)


def _controller_one_hot(value, names):
    return [1.0 if value == name else 0.0 for name in names]


def _controller_proposal_features(state, proposal, player, planets, fleets, angular_velocity, step):
    moves = proposal.get("moves", []) or []
    name = proposal.get("name", "unknown")
    by_id = {int(p.id): p for p in planets}
    pressure = _proposal_pressure_lookup(player, planets, fleets)
    player_count = _player_count(planets, fleets, player)
    owned = [p for p in planets if int(p.owner) == int(player)]
    enemy_owned = [p for p in planets if int(p.owner) not in (int(player), NEUTRAL)]
    neutral_owned = [p for p in planets if int(p.owner) == NEUTRAL]
    incoming_ships = sum(int(row["ships"]) for row in pressure.values())
    max_pressure = max((int(row["ships"]) for row in pressure.values()), default=0)
    stockpile = max((int(p.ships) for p in owned), default=0)

    move_count = len(moves)
    ships_total = 0
    max_ships = 0
    target_ids = set()
    source_quadrants = set()
    target_quadrants = set()
    owner_counts = {"enemy": 0, "neutral": 0, "friendly": 0, "unknown": 0}
    owner_ships = {"enemy": 0, "neutral": 0, "friendly": 0, "unknown": 0}
    enemy_max = 0
    friendly_pressure_count = 0
    friendly_pressure_coverage = 0.0
    maintenance_count = 0
    source_drain_pressure_count = 0
    static_count = 0
    rotating_count = 0
    big_count = 0
    same_quadrant_count = 0
    enemy_margin_sum = 0.0
    enemy_margin_min = 4.0
    enemy_margin_max = -4.0
    decisive_enemy_count = 0
    thin_enemy_count = 0
    neutral_capture_count = 0
    thin_neutral_count = 0
    opportunity_total = 0.0
    best_opportunity = -4.0
    positive_opportunity_count = 0
    trend_counts = {trend: 0 for trend in CONTROLLER_TREND_NAMES}

    for move in moves:
        if not move or len(move) < 3:
            continue
        source = by_id.get(int(move[0]))
        target = _proposal_target_for_move(move, planets)
        ships = int(move[2])
        ships_total += ships
        max_ships = max(max_ships, ships)
        if source is not None:
            source_quadrants.add(_quadrant(source))
        if target is None:
            owner_group = "unknown"
        else:
            target_ids.add(int(target.id))
            target_quadrants.add(_quadrant(target))
            if _is_static(target):
                static_count += 1
            else:
                rotating_count += 1
            if _is_big(target):
                big_count += 1
            if source is not None and _quadrant(source) == _quadrant(target):
                same_quadrant_count += 1
            if int(target.owner) == int(player):
                owner_group = "friendly"
                target_pressure = pressure.get(int(target.id))
                if target_pressure:
                    needed = max(1, int(target_pressure["ships"]) - int(target.ships) + 1)
                    friendly_pressure_count += 1
                    friendly_pressure_coverage += min(2.5, float(ships) / float(max(1, needed)))
                else:
                    maintenance_count += 1
            elif int(target.owner) == NEUTRAL:
                owner_group = "neutral"
                need = _planned_capture_need(source, target, angular_velocity) if source is not None else int(target.ships) + 1
                if ships >= need:
                    neutral_capture_count += 1
                else:
                    thin_neutral_count += 1
            else:
                owner_group = "enemy"
                enemy_max = max(enemy_max, ships)
                need = _offensive_capture_need(source, target, player, planets, fleets, angular_velocity) if source is not None else int(target.ships) + 1
                margin = (float(ships) - float(need)) / max(12.0, float(need))
                enemy_margin_sum += margin
                enemy_margin_min = min(enemy_margin_min, margin)
                enemy_margin_max = max(enemy_margin_max, margin)
                if margin >= 0.0:
                    decisive_enemy_count += 1
                else:
                    thin_enemy_count += 1

            if source is not None and int(source.id) in pressure:
                source_drain_pressure_count += 1

        owner_counts[owner_group] += 1
        owner_ships[owner_group] += ships
        opportunity_value, opportunity_reason = _proposal_opportunity_value(
            state,
            player,
            planets,
            fleets,
            angular_velocity,
            step,
            source,
            target,
            ships,
        )
        opportunity_total += float(opportunity_value)
        best_opportunity = max(best_opportunity, float(opportunity_value))
        if opportunity_value > 0.0:
            positive_opportunity_count += 1
        if opportunity_reason in trend_counts:
            trend_counts[opportunity_reason] += 1

    enemy_count = owner_counts["enemy"]
    if enemy_count == 0:
        enemy_margin_min = 0.0
        enemy_margin_max = 0.0
    enemy_margin_avg = _controller_ratio(enemy_margin_sum, enemy_count)
    avg_ships = _controller_ratio(ships_total, move_count)

    features = [
        _controller_clip(step, TOTAL_STEPS),
        _controller_clip(player_count, 4.0),
        _controller_clip(len(planets), 40.0),
        _controller_clip(len(fleets or []), 180.0),
        _controller_clip(len(owned), 32.0),
        _controller_clip(len(enemy_owned), 32.0),
        _controller_clip(len(neutral_owned), 32.0),
        _controller_clip(incoming_ships, 400.0),
        _controller_clip(len(pressure), 8.0),
        _controller_clip(max_pressure, 220.0),
        _controller_clip(stockpile, 300.0),
    ]
    features.extend(_controller_one_hot(name, CONTROLLER_SOURCE_NAMES))
    features.extend(
        [
            _controller_clip(float(proposal.get("score", 0.0)), 18.0),
            _controller_clip(move_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(ships_total, 520.0),
            _controller_clip(avg_ships, 120.0),
            _controller_clip(max_ships, 260.0),
            _controller_clip(len(target_ids), float(PROPOSAL_MAX_MOVES)),
            _controller_clip(len(source_quadrants), 4.0),
            _controller_clip(len(target_quadrants), 4.0),
            _controller_clip(owner_counts["enemy"], float(PROPOSAL_MAX_MOVES)),
            _controller_clip(owner_ships["enemy"], 520.0),
            _controller_clip(enemy_max, 260.0),
            _controller_clip(owner_counts["neutral"], float(PROPOSAL_MAX_MOVES)),
            _controller_clip(owner_ships["neutral"], 520.0),
            _controller_clip(owner_counts["friendly"], float(PROPOSAL_MAX_MOVES)),
            _controller_clip(owner_ships["friendly"], 520.0),
            _controller_clip(friendly_pressure_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(friendly_pressure_coverage, 5.0),
            _controller_clip(maintenance_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(source_drain_pressure_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(static_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(rotating_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(big_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(same_quadrant_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(enemy_margin_avg, 1.5),
            _controller_clip(enemy_margin_min, 1.5),
            _controller_clip(enemy_margin_max, 1.5),
            _controller_clip(decisive_enemy_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(thin_enemy_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(neutral_capture_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(thin_neutral_count, float(PROPOSAL_MAX_MOVES)),
            _controller_clip(opportunity_total, 18.0),
            _controller_clip(best_opportunity, 8.0),
            _controller_clip(positive_opportunity_count, float(PROPOSAL_MAX_MOVES)),
            _controller_ratio(owner_ships["enemy"], max(1, ships_total)),
            _controller_ratio(owner_ships["friendly"], max(1, ships_total)),
            _controller_ratio(owner_ships["neutral"], max(1, ships_total)),
        ]
    )
    features.extend(
        [
            _controller_clip(trend_counts[trend], float(PROPOSAL_MAX_MOVES))
            for trend in CONTROLLER_TREND_NAMES
        ]
    )
    return features


def _predict_controller_value(features):
    model = _load_controller_model()
    if model is None or np is None:
        return None
    vector = np.asarray(features, dtype=np.float32)
    if vector.ndim != 1 or vector.shape[0] != int(model["weights"].shape[0]):
        return None
    z = ((vector - model["mean"]) / model["std"]) @ model["weights"] + float(model["bias"])
    return _sigmoid(float(z))


def _apply_controller_model_to_proposal(state, proposal, player, planets, fleets, angular_velocity, step):
    features = _controller_proposal_features(state, proposal, player, planets, fleets, angular_velocity, step)
    prediction = _predict_controller_value(features)
    proposal["controller_feature_dim"] = len(features)
    if prediction is None:
        return proposal
    proposal["controller_value"] = float(prediction)
    proposal["score"] = float(proposal["score"]) + CONTROLLER_MODEL_WEIGHT * (float(prediction) - 0.5)
    reason = proposal.get("reason", "")
    suffix = "ctrl:%.2f" % float(prediction)
    proposal["reason"] = f"{reason},{suffix}" if reason else suffix
    return proposal


def _score_proposal(state, name, moves, player, planets, fleets, angular_velocity, step, apply_controller=True):
    cleaned, invalid = _sanitize_proposal_moves(moves, player, planets)
    if not cleaned:
        return None

    by_id = {int(p.id): p for p in planets}
    pressure = _proposal_pressure_lookup(player, planets, fleets)
    claimed_target_ids = _claimed_target_ids(state, player, planets, fleets)
    total = 0.0
    best_single = -999.0
    enemy_attacks = 0
    neutral_captures = 0
    useful_reinforce = 0
    maintenance_moves = 0
    opportunity_total = 0.0
    best_opportunity = -999.0
    ships_total = 0
    target_ids = set()
    reasons = []

    for move in cleaned:
        source = by_id.get(int(move[0]))
        target = _proposal_target_for_move(move, planets)
        ships = int(move[2])
        ships_total += ships
        if source is None or target is None:
            total -= 0.35
            continue
        target_ids.add(int(target.id))
        already_claimed = int(target.id) in claimed_target_ids and int(target.owner) != int(player)
        model_score = _predict_tactical_value(
            state,
            player,
            planets,
            fleets,
            angular_velocity,
            step,
            source,
            target,
            ships,
        )
        if model_score is None:
            model_score = _fallback_action_value(state, player, planets, fleets, angular_velocity, step, source, target, ships)
        action_score = float(model_score)
        opportunity_value, opportunity_reason = _proposal_opportunity_value(
            state,
            player,
            planets,
            fleets,
            angular_velocity,
            step,
            source,
            target,
            ships,
        )
        opportunity_total += float(opportunity_value)
        best_opportunity = max(best_opportunity, float(opportunity_value))

        if int(target.owner) == int(player):
            target_pressure = pressure.get(int(target.id))
            if target_pressure:
                needed = max(1, int(target_pressure["ships"]) - int(target.ships) + 1)
                coverage = min(1.6, float(ships) / float(max(1, needed)))
                action_score += 0.70 + coverage
                useful_reinforce += 1
            else:
                action_score -= 0.85
                maintenance_moves += 1
        elif int(target.owner) == NEUTRAL:
            need = _planned_capture_need(source, target, angular_velocity)
            if ships >= need:
                owned_count = sum(1 for p in planets if int(p.owner) == int(player))
                foundation_scale = 1.0 if owned_count < 3 else 0.30
                action_score += foundation_scale * (0.35 + 0.05 * float(target.production))
                neutral_captures += 1
            else:
                action_score -= 0.75
        else:
            need = _offensive_capture_need(source, target, player, planets, fleets, angular_velocity)
            margin = (float(ships) - float(need)) / max(12.0, float(need))
            action_score += 1.45 + 2.15 * margin + 0.16 * float(target.production)
            action_score += 0.35 if _is_big(target) else 0.0
            action_score += 0.22 if _is_static(target) else 0.0
            action_score += 0.55 * float(opportunity_value)
            if margin < 0.0:
                action_score -= 1.65
            else:
                enemy_attacks += 1

        if already_claimed:
            action_score -= 1.35

        total += action_score
        best_single = max(best_single, action_score)

    diversity = min(3, len(target_ids)) * 0.08
    total += best_single * 0.55
    total += float(PROPOSAL_SOURCE_PRIOR.get(name, 0.0))
    total += enemy_attacks * 0.90
    total += neutral_captures * 0.05
    total += useful_reinforce * 0.12
    total += 0.95 * opportunity_total
    total += 0.65 * best_opportunity
    total += diversity
    total -= invalid * 1.3
    total -= max(0, len(cleaned) - 7) * 0.20
    total -= maintenance_moves * 0.55
    total -= max(0, ships_total - 220) * 0.004

    if enemy_attacks:
        reasons.append("enemy:%d" % enemy_attacks)
    if best_opportunity > 0:
        reasons.append("opp:%.2f" % float(best_opportunity))
    if useful_reinforce:
        reasons.append("reinforce:%d" % useful_reinforce)
    if neutral_captures:
        reasons.append("neutral:%d" % neutral_captures)
    reasons.append("ships:%d" % ships_total)
    proposal = {
        "name": name,
        "moves": cleaned,
        "score": total,
        "reason": ",".join(reasons),
    }
    if apply_controller:
        proposal = _apply_controller_model_to_proposal(
            state,
            proposal,
            player,
            planets,
            fleets,
            angular_velocity,
            step,
        )
    return proposal


def _hold_proposal_score(player, planets, fleets):
    pressure = _proposal_pressure_lookup(player, planets, fleets)
    incoming = sum(int(row["ships"]) for row in pressure.values())
    owned = [p for p in planets if int(p.owner) == int(player)]
    stockpile = max((int(p.ships) for p in owned), default=0)
    owned_count = len(owned)
    neutral_targets = sum(1 for p in planets if int(p.owner) == NEUTRAL)
    if incoming > 0:
        return -0.65 - min(1.5, float(incoming) / 80.0)
    if owned_count < 3 and neutral_targets:
        return -0.25
    if stockpile >= 70:
        return 0.15
    return 0.20


def proposal_selector_agent(obs, config=None):
    player, planets, fleets, angular_velocity, _ = _parse(obs)
    state = _state_for(player, obs)
    state["turn_claimed_target_ids"] = set()
    step = int(_obs_get(obs, "step", _obs_get(obs, "turn", 0)) or 0)
    if not planets:
        return []

    _update_tactical_events(state, player, planets)
    _update_recent_static_capture_focus(state, player, planets)
    _reset_proposal_sources_for_new_game(state, step)

    deadline = time.perf_counter() + PROPOSAL_AGENT_TIME_BUDGET
    proposals = []
    for name, module, kind in _proposal_sources():
        if time.perf_counter() > deadline and proposals:
            break
        raw_moves = _call_proposal_source(name, module, kind, obs, config, state, player, planets, fleets, step)
        scored = _score_proposal(state, name, raw_moves, player, planets, fleets, angular_velocity, step)
        if scored is not None:
            proposals.append(scored)

    hold_proposal = {
        "name": "hold",
        "moves": [],
        "score": _hold_proposal_score(player, planets, fleets),
        "reason": "stockpile",
    }
    proposals.append(
        _apply_controller_model_to_proposal(
            state,
            hold_proposal,
            player,
            planets,
            fleets,
            angular_velocity,
            step,
        )
    )

    selected = max(proposals, key=lambda proposal: proposal["score"])
    state["proposal_source"] = selected["name"]
    state["proposal_reason"] = "score:%.3f %s" % (float(selected["score"]), selected["reason"])
    for move in selected["moves"]:
        target = _proposal_target_for_move(move, planets)
        if target is not None:
            _record_tactical_move(state, player, target, int(move[2]))
    return selected["moves"]


def anchor_feeder_agent(obs, config=None):
    player, planets, fleets, angular_velocity, _ = _parse(obs)
    state = _state_for(player, obs)
    state["turn_claimed_target_ids"] = set()
    step = int(_obs_get(obs, "step", _obs_get(obs, "turn", 0)) or 0)
    moves = []
    reserved = {}

    if not planets:
        return moves

    _update_tactical_events(state, player, planets)
    _update_recent_static_capture_focus(state, player, planets)
    battery_anchor = _primary_anchor_in_battery_mode(state, player, planets)
    blocked_trap_ids = {battery_anchor.id} if battery_anchor is not None else set()
    roles = _our_roles(planets, player, fleets=fleets, state=state)

    _reactive_trap(
        player,
        planets,
        fleets,
        moves,
        reserved,
        blocked_source_ids=blocked_trap_ids,
        angular_velocity=angular_velocity,
    )
    _drain_bursts(state, player, planets, moves, reserved, angular_velocity)
    _initiation_phase(state, player, planets, moves, reserved, angular_velocity)
    owned_count = sum(1 for p in planets if int(p.owner) == player)
    if owned_count < 2 and state.get("opening_target_ids"):
        _unstick_opening_if_stalled(state, player, planets, fleets, moves, reserved, angular_velocity, step)
        return moves
    if owned_count < 2 and not state.get("opened") and len(state.get("opening_launched_ids", [])) < 2:
        _unstick_opening_if_stalled(state, player, planets, fleets, moves, reserved, angular_velocity, step)
        return moves

    if not _operationally_any_established(planets, player):
        _pre_establishment_expansion(state, player, planets, fleets, moves, reserved, angular_velocity, step)
        return moves

    roles = _our_roles(planets, player, fleets=fleets, state=state)
    _primary_anchor_action(state, player, planets, moves, reserved, angular_velocity)
    primary_anchor_id = state.get("primary_anchor_id")
    model_action = _model_controller_action(
        state,
        player,
        planets,
        fleets,
        moves,
        reserved,
        roles,
        angular_velocity,
        step,
        primary_anchor_id=primary_anchor_id,
    )
    if model_action not in ("decisive_attack", "pressure_counter"):
        _attacker_stage_action(state, player, planets, fleets, moves, reserved, angular_velocity, step, primary_anchor_id=primary_anchor_id)
        _striker_mode(state, player, planets, fleets, moves, reserved, roles, angular_velocity, step, primary_anchor_id=primary_anchor_id)
    _feeder_logic(state, player, planets, fleets, moves, reserved, angular_velocity, step, primary_anchor_id=primary_anchor_id)
    collector_id = state.get("static_collector_id")
    collector_blocked_ids = {collector_id} if collector_id is not None else set()
    if model_action not in ("decisive_attack", "pressure_counter"):
        _role_actions(state, player, planets, fleets, moves, reserved, roles, angular_velocity, step, blocked_source_ids=collector_blocked_ids)
        _expansion(state, player, planets, fleets, moves, reserved, roles, angular_velocity, step)
    if not moves:
        _force_nearest_unconquered_move(
            state,
            player,
            planets,
            fleets,
            moves,
            reserved,
            angular_velocity,
            step,
            min_ships=OPENING_STALL_SHIPS,
            require_attack_ready=True,
        )

    return moves


def smith_moveset_agent(obs, config=None):
    if _smith_controller is None or not USE_SMITH_DELEGATION:
        return anchor_feeder_agent(obs, config)

    player, planets, fleets, _, _ = _parse(obs)
    state = _state_for(player, obs)
    step = int(_obs_get(obs, "step", _obs_get(obs, "turn", 0)) or 0)
    _update_tactical_events(state, player, planets)
    intent = _smith_controller_intent(state, player, planets, fleets, step)
    moves = _smith_controller.controller_agent(obs, config=config, intent=intent)

    if _tf_infer_action_target is not None:
        for move in moves:
            target = _tf_infer_action_target(move, planets)
            if target is not None:
                _record_tactical_move(state, player, target, int(move[2]))
    return moves


def agent(obs, config=None):
    return proposal_selector_agent(obs, config)


nearest_planet_sniper = proposal_selector_agent
