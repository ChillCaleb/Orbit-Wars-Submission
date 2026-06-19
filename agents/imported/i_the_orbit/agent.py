"""
Orbit Wars — Phase-based agent

PHASE 1 (steps 0 → PHASE_SWITCH_STEP):
  From the home planet, attack the closest STATIC (non-moving) planets first,
  expanding outward ring by ring. No wasted shots at moving targets.

PHASE 2 (steps PHASE_SWITCH_STEP → end):
  Switch to also targeting ORBITAL (inner, moving) planets.
  intercept_angle handles the leading automatically — we just pick the target.

Home planet: auto-detected at step 0 as the planet we own.
             Tracked each turn as the owned planet nearest to the original
             home position (robust even if the home planet itself is moving).

Moving vs static: a planet is "moving" if its position changed more than
MOVE_THRESHOLD units from the previous turn. Accumulated over turns so the
classifier is stable from turn 2 onward.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor

from i_the_orbit_orbit_lite.geometry import fleet_speed
from i_the_orbit_orbit_lite.intercept_aim import intercept_angle
from i_the_orbit_orbit_lite.movement import MovementConfig
from i_the_orbit_orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from i_the_orbit_orbit_lite.obs import parse_obs
from i_the_orbit_orbit_lite.distance_cache import build_distance_cache
from i_the_orbit_orbit_lite.planner_core import (
    _empty_entries,
    _greedy_select,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from i_the_orbit_orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves

# ─────────────────────────────────────────────────────────────────
# Tunable constants — the only numbers you should ever need to touch
# ─────────────────────────────────────────────────────────────────

TOTAL_STEPS        = 500

# How many steps to spend only on static planets before switching to orbitals
PHASE_SWITCH_STEP  = 80

# A planet is "moving" if it moved more than this per turn (game units)
MOVE_THRESHOLD     = 0.3

# Score multipliers
STATIC_BOOST       = 1.0   # baseline — Phase 1 targets
ORBITAL_BOOST      = 2.5   # Phase 2: orbitals score higher (they produce more)
HOME_DEFENSE_BOOST = 3.0   # reinforce home planet if threatened

# Visitor comets: extra boost during their brief window
COMET_BOOST        = 5.0


# ─────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Config:
    horizon:                  int   = 10
    max_sources:              int   = 8
    max_targets:              int   = 8
    max_waves:                int   = 6
    roi_threshold:            float = 1.20
    min_ships:                float = 4.0
    reinforce_beta:           float = 2.0
    reinforce_eta_free:       float = 3.0
    reinforce_eta_scale:      float = 10.0
    defense_horizon:          float = 10.0
    defense_max_waves:        int   = 2
    defense_margin:           float = 1.05
    prod_rush_steps:          int   = 60
    prod_rush_top_k:          int   = 3
    prod_rush_discount:       float = 0.80
    # Phase 1: prefer nearest static. Phase 2: allow orbitals
    phase_switch_step:        int   = PHASE_SWITCH_STEP
    move_threshold:           float = MOVE_THRESHOLD
    static_boost:             float = STATIC_BOOST
    orbital_boost:            float = ORBITAL_BOOST
    comet_boost:              float = COMET_BOOST
    home_defense_boost:       float = HOME_DEFENSE_BOOST
    near_fraction:            float = 0.80   # 80% waves go to nearest targets

    @property
    def max_offensive_targets(self) -> int:
        return self.max_targets

    @property
    def max_defensive_targets(self) -> int:
        return max(1, min(3, self.max_targets))

    @property
    def min_ships_to_launch(self) -> float:
        return self.min_ships

    @property
    def max_regroup_sources_per_lane(self) -> int:
        return self.max_sources

    @property
    def max_regroup_targets_per_source(self) -> int:
        return self.max_targets

    @property
    def max_regroup_time(self) -> float:
        return float(self.horizon)

    @property
    def regroup_pressure_delta_min(self) -> float:
        return 0.25

    @property
    def regroup_time_penalty_weight(self) -> float:
        return 1e-3


CONFIG_2P = Config()
CONFIG_3P = replace(Config(), horizon=9,  max_sources=6, max_targets=6,  roi_threshold=1.15)
CONFIG_4P = replace(Config(), horizon=8,  max_sources=5, max_targets=5,  roi_threshold=1.10, max_waves=5)

def _pick_config(player_count: int) -> Config:
    return CONFIG_4P if player_count >= 4 else CONFIG_3P if player_count == 3 else CONFIG_2P


# ─────────────────────────────────────────────────────────────────
# Movement / distance helpers
# ─────────────────────────────────────────────────────────────────

def _movement_cfg(cfg: Config, player_count: int) -> MovementConfig:
    return MovementConfig(
        movement_horizon=cfg.horizon,
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=player_count,
        max_tracked_fleets=64,
    )


def _enemy_pressure(obs, cache, *, horizon: float, pid: int) -> Tensor:
    P = obs.P
    device, dtype = obs.device, obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0    = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    spd   = fleet_speed(ships.clamp(min=1e-6))
    reach = (spd.view(P, 1) * horizon).clamp(min=1e-6)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != pid)
    eye   = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    decay = (1.0 - d0 / reach).clamp(min=0.0)
    return torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay)).sum(0)


# ─────────────────────────────────────────────────────────────────
# Memory  — persists across turns
# ─────────────────────────────────────────────────────────────────

class Memory:
    def __init__(self) -> None:
        self.movement            = None
        self.player_count        = None
        self.prev_pos: dict      = {}   # {planet_id: (x, y)} from last turn
        self.is_moving: dict     = {}   # {planet_id: bool} accumulated classifier
        self.home_pos: tuple     = None # (x, y) at step 0 — never changes
        self.home_planet_id: int = -1

    def reset(self) -> None:
        self.movement        = None
        self.player_count    = None
        self.prev_pos        = {}
        self.is_moving       = {}
        self.home_pos        = None
        self.home_planet_id  = -1


# ─────────────────────────────────────────────────────────────────
# Moving-planet classifier
#
# Updates memory.is_moving[planet_id] = True/False each turn.
# A planet is classified as "moving" as soon as it moves > threshold.
# Static planets stay False unless they suddenly move (robustness).
# ─────────────────────────────────────────────────────────────────

def _update_movement_classifier(obs, mem: Memory, threshold: float) -> None:
    raw = obs.planets if hasattr(obs, "planets") else []
    # obs from parse_obs has .x .y tensors indexed by position, not planet id.
    # We iterate planet indices directly.
    P = int(obs.P)
    xs = obs.x; ys = obs.y

    for i in range(P):
        pid_planet = i  # use index as stable id (planet order is fixed per game)
        cx, cy = float(xs[i].item()), float(ys[i].item())
        if pid_planet in mem.prev_pos:
            px, py = mem.prev_pos[pid_planet]
            dist_moved = ((cx - px)**2 + (cy - py)**2) ** 0.5
            if dist_moved > threshold:
                mem.is_moving[pid_planet] = True
            elif pid_planet not in mem.is_moving:
                mem.is_moving[pid_planet] = False
        else:
            # First time seeing this planet — assume static until proven otherwise
            mem.is_moving[pid_planet] = False
        mem.prev_pos[pid_planet] = (cx, cy)


def _moving_mask(obs, mem: Memory) -> Tensor:
    """Returns bool Tensor [P]: True = planet moves (inner orbital or comet)."""
    P = int(obs.P)
    device = obs.device
    result = torch.zeros(P, dtype=torch.bool, device=device)
    for i in range(P):
        if mem.is_moving.get(i, False):
            result[i] = True
    return result


def _static_mask(obs, mem: Memory) -> Tensor:
    return ~_moving_mask(obs, mem)


# ─────────────────────────────────────────────────────────────────
# Home planet tracker
#
# At step 0: record home position = centroid of all planets we own.
# Each turn: find the owned planet closest to that position.
# Works even if our starting planet is itself an orbital (moving).
# ─────────────────────────────────────────────────────────────────

def _init_home(obs, mem: Memory) -> None:
    owned = obs.owned & obs.alive
    if not bool(owned.any()):
        return
    xs = obs.x[owned]; ys = obs.y[owned]
    mem.home_pos = (float(xs.mean().item()), float(ys.mean().item()))


def _find_home_planet(obs, mem: Memory) -> int:
    """Returns planet index of current home planet."""
    owned = obs.owned & obs.alive
    if not bool(owned.any()) or mem.home_pos is None:
        return -1
    hx, hy = mem.home_pos
    xs = obs.x; ys = obs.y
    dists = ((xs - hx)**2 + (ys - hy)**2) ** 0.5
    # Among owned planets, find the one nearest to original home position
    dists_owned = torch.where(owned, dists, torch.full_like(dists, 1e9))
    return int(dists_owned.argmin().item())


# ─────────────────────────────────────────────────────────────────
# Phase-based score multiplier
#
# Phase 1: static planets get STATIC_BOOST, moving planets get 0
#          (effectively filtered out of candidate scoring)
# Phase 2: static keeps STATIC_BOOST, moving gets ORBITAL_BOOST
#          (orbitals score higher because they're inner = more production)
#
# In both phases: nearest targets score higher via distance penalty.
# The home planet is always protected (HOME_DEFENSE_BOOST for defense).
# ─────────────────────────────────────────────────────────────────

def _phase_boost(
    *,
    step: int,
    obs,
    mem: Memory,
    cfg: Config,
    cand_tgt_abs: Tensor,
) -> Tensor:
    P      = int(obs.P)
    device = obs.device
    dtype  = obs.ships.dtype

    moving = _moving_mask(obs, mem)   # [P] bool

    if step < cfg.phase_switch_step:
        # Phase 1: only static planets
        # Moving targets get a near-zero multiplier (not -inf, avoids NaN in score)
        planet_mult = torch.where(
            moving,
            torch.full((P,), 0.01, dtype=dtype, device=device),  # effectively skip
            torch.full((P,), cfg.static_boost, dtype=dtype, device=device),
        )
    else:
        # Phase 2: static + orbital, orbitals boosted
        planet_mult = torch.where(
            moving,
            torch.full((P,), cfg.orbital_boost, dtype=dtype, device=device),
            torch.full((P,), cfg.static_boost,  dtype=dtype, device=device),
        )

    # Comet detection: a moving planet that appeared after step 1 = comet
    # Give it an extra boost on top of orbital_boost (time-limited window)
    if step >= cfg.phase_switch_step:
        for i in range(P):
            if mem.is_moving.get(i, False):
                # Comets are moving but NOT inner planets — they appear/disappear.
                # Heuristic: if they showed up moving from turn 1 they're orbitals;
                # if they started static then began moving, they're comets.
                # We tag them with comet_boost in either case (safe to be generous).
                planet_mult[i] = max(float(planet_mult[i].item()), cfg.comet_boost)

    return planet_mult[cand_tgt_abs].to(dtype)


# ─────────────────────────────────────────────────────────────────
# Distance penalty — fast, uses mean not median
# ─────────────────────────────────────────────────────────────────

def _dist_penalty(dist: Tensor, ref: float) -> Tensor:
    return 1.0 / (1.0 + dist / max(ref, 1.0))


# ─────────────────────────────────────────────────────────────────
# Nearest-first ordering for Phase 1
#
# In Phase 1, from the home planet outward, we want the score to
# decay sharply with distance so the planner always grabs the
# nearest static planet before the next-nearest.
# We apply a steeper distance penalty when step < phase_switch_step.
# ─────────────────────────────────────────────────────────────────

def _phase1_distance_weight(
    *,
    step: int,
    cfg: Config,
    dist: Tensor,
    mean_dist: float,
) -> Tensor:
    if step < cfg.phase_switch_step:
        # Steep falloff: exponent makes near targets overwhelmingly preferred
        ref = max(mean_dist * 0.35, 1.0)
    else:
        # Phase 2: gentler — we want to reach orbitals that may be further away
        ref = max(mean_dist * 0.60, 1.0)
    return _dist_penalty(dist, ref)


# ─────────────────────────────────────────────────────────────────
# Light proactive defense
# ─────────────────────────────────────────────────────────────────

def _build_defense(*, movement, obs, cache, cfg: Config, pid: int, home_idx: int) -> object:
    P = int(obs.P)
    device, dtype = obs.device, obs.ships.dtype
    if P == 0:
        return _empty_entries(device, dtype)
    owned = obs.owned & obs.alive
    if not bool(owned.any()):
        return _empty_entries(device, dtype)

    H_def = int(cfg.defense_horizon)
    status = movement.garrison_status(max_horizon=H_def)
    if status.ships.shape[-1] < 2:
        return _empty_entries(device, dtype)
    ships_at_H = status.ships[:, -1]

    # Home planet gets extra-sensitive defense trigger
    threatened_mask = owned & (ships_at_H < 0)
    if home_idx >= 0 and 0 <= home_idx < P:
        # Home is threatened if garrison will drop below home_defense_boost margin
        home_thresh = ships_at_H[home_idx] < float(cfg.home_defense_boost)
        if home_thresh:
            threatened_mask[home_idx] = True

    if not bool(threatened_mask.any()):
        return _empty_entries(device, dtype)

    tgt_idx = threatened_mask.nonzero(as_tuple=False).squeeze(1)
    src_idx = owned.nonzero(as_tuple=False).squeeze(1)
    if src_idx.numel() == 0 or tgt_idx.numel() == 0:
        return _empty_entries(device, dtype)

    d0        = cache.cross_dist[0].to(dtype)
    src_ships = obs.ships[src_idx].to(dtype)
    entries   = []
    waves     = 0

    for t_i in range(int(tgt_idx.shape[0])):
        if waves >= cfg.defense_max_waves:
            break
        tgt    = int(tgt_idx[t_i].item())
        deficit = float(-ships_at_H[tgt].item())
        need    = max(deficit * cfg.defense_margin, cfg.min_ships)
        dists   = d0[src_idx, tgt]
        spds    = fleet_speed(src_ships.clamp(min=1.0))
        etas    = (dists / spds.clamp(min=1e-6)).ceil()
        ok      = (etas <= H_def) & (src_ships > need + cfg.min_ships) & (src_idx != tgt)
        if not bool(ok.any()):
            continue
        best     = int(torch.where(ok, dists, torch.full_like(dists, 1e9)).argmin().item())
        best_src = int(src_idx[best].item())
        send     = max(min(float(src_ships[best].item()) * 0.55, need + cfg.min_ships), cfg.min_ships)
        entry = make_launch_set(
            source_slots=torch.tensor([[best_src]], dtype=torch.long, device=device),
            target_slots=torch.tensor([[tgt]],      dtype=torch.long, device=device),
            ships=torch.tensor([[send]],             dtype=dtype,      device=device),
            eta=torch.tensor([[float(etas[best].item())]], dtype=dtype, device=device),
            valid=torch.tensor([[True]],             dtype=torch.bool, device=device),
            player_id=pid,
        )
        entries.append(entry)
        waves += 1

    return concat_launch_entries(entries) if entries else _empty_entries(device, dtype)


# ─────────────────────────────────────────────────────────────────
# Late-game suppression  (skip entirely when plenty of time left)
# ─────────────────────────────────────────────────────────────────

def _late_suppress(*, score, obs, target_idx, cand_tgt_short,
                   cand_is_def, cand_eta, step, pid):
    remaining = TOTAL_STEPS - step
    if remaining > 120:
        return score
    P     = int(obs.P)
    device, dtype = score.device, score.dtype
    tgt_abs   = target_idx[cand_tgt_short].clamp(0, P - 1)
    tgt_owner = obs.owner_abs.to(device=device)[tgt_abs].long()
    eta       = cand_eta.reshape(score.shape).to(device=device, dtype=dtype)
    neutral   = tgt_owner < 0
    enemy     = (tgt_owner >= 0) & (tgt_owner != pid) & ~cand_is_def
    score = torch.where(neutral, score * torch.sigmoid((remaining - eta) / 30.0 * 0.5), score)
    score = torch.where(enemy,   score * torch.sigmoid((remaining - eta) / 20.0 * 0.5), score)
    return torch.where(eta >= remaining, torch.full_like(score, float("-inf")), score)


# ─────────────────────────────────────────────────────────────────
# Core wave planner
# ─────────────────────────────────────────────────────────────────

def _plan_waves(*, movement, obs, obs_tensors, cache, status, prod,
                alive_by_step, cfg: Config, player_count: int,
                mem: Memory, pid: int, home_idx: int):

    P = int(obs.P)
    device, dtype = obs.device, obs.ships.dtype
    step  = int(obs_tensors["step"].reshape(-1)[0].item())
    phase = 1 if step < cfg.phase_switch_step else 2

    H_axis = int(status.ships.shape[-1])
    H      = max(H_axis - 1, 0)
    K_eta  = max(1, min(cfg.horizon, H))
    W      = max(1, cfg.max_waves)
    H_eff  = torch.full((), float(H), dtype=dtype, device=device)

    ships    = obs.ships.to(dtype)
    prod_val = prod.to(dtype)

    # ── Sources: all owned planets above min_ships ──
    src_score = ships + 0.4 * prod_val * (ships / (ships + 1.0))
    src_mask  = obs.owned & obs.alive & (ships >= cfg.min_ships)
    src_score = torch.where(src_mask, src_score,
                            torch.tensor(float("-inf"), device=device, dtype=dtype))
    S_cap    = max(1, min(cfg.max_sources, P))
    src_idx  = torch.topk(src_score, min(S_cap, src_score.numel())).indices
    src_ok   = src_mask[src_idx]

    # ── Targets: shortlist from planner_core ──
    tgt_idx, tgt_ok = build_target_shortlist(
        obs, obs_tensors, status, cache,
        config=cfg, K_eta=K_eta, H=H, prod=prod, source_mask=src_mask,
    )

    # In Phase 1: forcibly remove moving planets from target list
    if phase == 1:
        moving = _moving_mask(obs, mem)
        static_ok = ~moving[tgt_idx.clamp(0, P-1)]
        tgt_ok = tgt_ok & static_ok

    if not bool(tgt_ok.any()):
        return _empty_entries(device, dtype)

    S, T = int(src_idx.shape[0]), int(tgt_idx.shape[0])
    tgt_is_mine = obs.owned[tgt_idx.clamp(0, P-1)]
    src_ships   = obs.ships[src_idx.clamp(0, P-1)].to(dtype)
    drain = safe_drain(status, source_idx=src_idx, source_ships=src_ships,
                       H_eff=H_eff, player_id=pid)

    eta_cap    = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    enemy_mass = _enemy_pressure(obs, cache, horizon=float(K_eta), pid=pid)

    reinforcement = None
    if cfg.reinforce_beta > 0.0:
        k_arr = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho   = reinforcement_timing_factor(k_arr,
                    eta_free=cfg.reinforce_eta_free, eta_scale=cfg.reinforce_eta_scale)
        reinforcement = cfg.reinforce_beta * rho.view(1, K_eta) * enemy_mass[tgt_idx.clamp(0,P-1)].view(T, 1)

    floor = capture_floor(status, target_idx=tgt_idx, k_max=K_eta,
                          capture_overhead=1.0, player_id=pid, reinforcement=reinforcement)
    K     = int(floor.shape[-1])
    sizes = drain.view(S, 1).expand(S, T).floor().clamp(min=1.0)

    active = reachable_mask(movement, source_idx=src_idx, target_idx=tgt_idx,
                             fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap).squeeze(-1)
    aim    = intercept_angle(movement, src_idx.unsqueeze(1),
                              tgt_idx.unsqueeze(0), sizes, active=active)
    angle  = aim["angle"]
    eta    = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    if K > 0:
        k_floor = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K-1)
        floor_at = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_floor.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at = torch.ones(S, T, dtype=dtype, device=device)

    neq   = src_idx.view(S, 1) != tgt_idx.view(1, T)
    valid = viable & (sizes >= floor_at) & (sizes >= cfg.min_ships) & neq & src_ok.view(S, 1) & tgt_ok.view(1, T)

    L, C = 1, S * T
    cand_src       = src_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot  = tgt_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send      = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle     = angle.reshape(C, L)
    cand_eta       = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active    = valid.reshape(C, L)
    cand_valid     = valid.reshape(C)
    cand_is_def    = tgt_is_mine[cand_tgt_short]
    cand_src_abs   = cand_src.squeeze(-1)
    cand_tgt_abs   = cand_tgt_slot

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send, eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1), player_id=pid,
    )
    score = score_candidates(
        status, prod=prod, alive_by_step=alive_by_step,
        player_count=player_count, launches=launches, player_id=pid,
    )

    # ── Distance penalty (phase-aware steepness) ──
    d0        = cache.cross_dist[0].to(dtype)
    dist      = d0[cand_src_abs, cand_tgt_abs]
    alive_d   = d0[obs.alive][:, obs.alive]
    mean_dist = float(alive_d.mean().item()) if alive_d.numel() > 0 else 15.0
    score     = score * _phase1_distance_weight(
                    step=step, cfg=cfg, dist=dist, mean_dist=mean_dist
                ).reshape(score.shape)

    # ── Phase / moving-planet boost ──
    phase_mult = _phase_boost(step=step, obs=obs, mem=mem, cfg=cfg,
                              cand_tgt_abs=cand_tgt_abs)
    score = score * phase_mult.reshape(score.shape)

    # ── Early-game prod rush boost ──
    if step <= cfg.prod_rush_steps:
        neutral = obs.owner_abs < 0
        if bool(neutral.any()):
            pn  = torch.where(neutral & obs.alive, prod_val,
                              torch.zeros(P, dtype=dtype, device=device))
            top = float(torch.topk(pn, min(cfg.prod_rush_top_k, pn.numel())).values[-1].item())
            is_top = (obs.owner_abs[cand_tgt_abs] < 0) & (prod_val[cand_tgt_abs] >= top - 1e-6)
            score  = torch.where(is_top.reshape(score.shape),
                                 score * (1.0 / cfg.prod_rush_discount), score)

    # ── Late-game suppression ──
    score = _late_suppress(score=score, obs=obs, target_idx=tgt_idx,
                           cand_tgt_short=cand_tgt_short, cand_is_def=cand_is_def,
                           cand_eta=cand_eta, step=step, pid=pid)

    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    # ── Near-first wave split ──
    W_near = max(1, round(W * cfg.near_fraction))
    W_far  = max(1, W - W_near)
    eta_1d = cand_eta.squeeze(-1) if cand_eta.dim() > 1 else cand_eta
    med    = float(eta_1d[cand_valid].float().median().item()) if bool(cand_valid.any()) else 1e9
    near   = cand_valid & (eta_1d.reshape(cand_valid.shape) <= med)
    far    = cand_valid & (eta_1d.reshape(cand_valid.shape) >  med)

    gkw = dict(P=P, device=device, dtype=dtype, cand_src=cand_src,
               cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
               cand_active=cand_active, cand_tgt_slot=cand_tgt_slot,
               cand_tgt_short=cand_tgt_short, cand_is_def=cand_is_def,
               source_budget=obs.ships.to(dtype).clone(), target_exists=tgt_ok,
               roi_threshold=cfg.roi_threshold)

    near_e, _ = _greedy_select(W=W_near, score=torch.where(near, score, torch.full_like(score, float("-inf"))), **gkw)
    far_e,  _ = _greedy_select(W=W_far,  score=torch.where(far,  score, torch.full_like(score, float("-inf"))), **gkw)
    return concat_launch_entries([near_e, far_e])


# ─────────────────────────────────────────────────────────────────
# Turn pipeline
# ─────────────────────────────────────────────────────────────────

def run_turn(obs_tensors: dict, *, cfg: Config, player_count: int, mem: Memory) -> dict:
    device = obs_tensors["planets"].device
    obs    = parse_obs(obs_tensors)
    if obs.P == 0:
        return empty_action_row(device)

    pid  = int(obs.player_id)
    step = int(obs_tensors["step"].reshape(-1)[0].item())

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_cfg(cfg, player_count),
        cached_movement=mem.movement,
    )
    mem.movement = movement

    # ── Step 0: initialise home position ──
    if step == 0:
        _init_home(obs, mem)

    # ── Update movement classifier every turn ──
    _update_movement_classifier(obs, mem, threshold=cfg.move_threshold)

    # ── Find current home planet ──
    home_idx = _find_home_planet(obs, mem)

    cache         = build_distance_cache(movement, max_k=cfg.horizon)
    H             = cfg.horizon
    status        = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[:H + 1]
    prod          = movement.planet_prod

    # Defense first
    defense = _build_defense(movement=movement, obs=obs, cache=cache,
                              cfg=cfg, pid=pid, home_idx=home_idx)

    # Offensive waves
    waves = _plan_waves(movement=movement, obs=obs, obs_tensors=obs_tensors,
                        cache=cache, status=status, prod=prod,
                        alive_by_step=alive_by_step, cfg=cfg,
                        player_count=player_count, mem=mem,
                        pid=pid, home_idx=home_idx)

    entries = disambiguate_duplicate_launches(concat_launch_entries([defense, waves]))
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement,
        entries=entries, player_id=pid,
    )
    apply_private_planned_launches(
        movement=movement, launches=launches,
        owner_id=pid, obs_tensors=obs_tensors,
    )
    return entries_to_sparse_payload(entries, planet_ids=obs_tensors["planets"][..., 0].long())


# ─────────────────────────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────────────────────────

class Runtime:
    def __init__(self) -> None:
        self.mem = Memory()

    def reset(self) -> None:
        self.mem.reset()

    def act(self, obs_tensors: dict) -> dict:
        mem = self.mem
        if bool((obs_tensors["step"] == 0).all()):
            mem.reset()
        if mem.player_count is None:
            mem.player_count = largest_initial_player_count(obs_tensors)
        cfg = _pick_config(mem.player_count)
        return run_turn(obs_tensors, cfg=cfg, player_count=mem.player_count, mem=mem)


_RT = Runtime()


def agent(obs):
    player      = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id   = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RT.act(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
