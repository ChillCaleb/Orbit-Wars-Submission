#!/usr/bin/env python3
"""Starter self-play PPO loop for Orbit Wars.

This is intentionally not medal-scale infrastructure. It is the first local loop
that gives the repo the same shape as the top writeups: a learnable action
interface, self-play rollouts, PPO updates, checkpoints, and a place to attach a
future fast simulator / PFSP league.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from kaggle_environments import make
from torch import nn
from torch.distributions import Categorical

import main as live_agent
from agent_lab import extract_observation, extract_reward, extract_status
from tactical_features import obs_get


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "data" / "self_play_runs"

ACTION_KINDS = ("send_all", "sortie", "hold", "kill_at_arrival")
ACTION_TO_INDEX = {name: idx for idx, name in enumerate(ACTION_KINDS)}
TOTAL_STEPS = 500.0

GLOBAL_DIM = 10
SOURCE_DIM = 13
TARGET_DIM = 13
RELATION_DIM = 8
EDGE_DIM = SOURCE_DIM + TARGET_DIM + RELATION_DIM + len(ACTION_KINDS)


@dataclass(frozen=True)
class Candidate:
    source_id: int
    target_id: int
    kind: str
    ships: int
    angle: float
    eta: float
    features: tuple[float, ...]

    @property
    def key(self) -> str:
        return f"{self.source_id}:{self.target_id}:{self.kind}"

    def move(self) -> list[Any]:
        return [self.source_id, float(self.angle), int(self.ships)]


@dataclass
class SourceDecision:
    source_id: int
    selected_key: str | None


@dataclass
class Transition:
    obs: dict[str, Any]
    player: int
    decisions: list[SourceDecision]
    old_logprob: float
    old_value: float
    reward: float = 0.0


@dataclass
class EpisodeResult:
    seed: int
    player_count: int
    rewards: list[float]
    shaped_rewards: list[float]
    statuses: list[str]
    winners: list[int]
    steps: int
    transitions: int


def serialize_obs(obs: Any) -> dict[str, Any]:
    return {
        "player": int(obs_get(obs, "player", 0) or 0),
        "step": int(obs_get(obs, "step", 0) or 0),
        "angular_velocity": float(obs_get(obs, "angular_velocity", 0.0) or 0.0),
        "planets": obs_get(obs, "planets", []) or [],
        "fleets": obs_get(obs, "fleets", []) or [],
        "comet_planet_ids": obs_get(obs, "comet_planet_ids", []) or [],
    }


def safe_div(part: float, whole: float) -> float:
    return 0.0 if abs(float(whole)) < 1e-9 else float(part) / float(whole)


def clip(value: float, limit: float = 4.0) -> float:
    return max(-limit, min(limit, float(value)))


def owner_counts(planets: list[Any], fleets: list[Any], player: int) -> dict[str, float]:
    owned = [p for p in planets if int(p.owner) == int(player)]
    neutral = [p for p in planets if int(p.owner) == live_agent.NEUTRAL]
    enemy = [p for p in planets if int(p.owner) not in (int(player), live_agent.NEUTRAL)]
    own_ships = sum(int(p.ships) for p in owned)
    enemy_ships = sum(int(p.ships) for p in enemy)
    own_prod = sum(int(p.production) for p in owned)
    enemy_prod = sum(int(p.production) for p in enemy)
    own_fleet = sum(int(f.ships) for f in fleets if int(f.owner) == int(player))
    enemy_fleet = sum(int(f.ships) for f in fleets if int(f.owner) not in (int(player), live_agent.NEUTRAL))
    return {
        "owned_count": float(len(owned)),
        "neutral_count": float(len(neutral)),
        "enemy_count": float(len(enemy)),
        "own_ships": float(own_ships + own_fleet),
        "enemy_ships": float(enemy_ships + enemy_fleet),
        "own_prod": float(own_prod),
        "enemy_prod": float(enemy_prod),
    }


def player_count_from_state(planets: list[Any], fleets: list[Any], player: int) -> int:
    owners = [int(player)]
    owners.extend(int(p.owner) for p in planets if int(p.owner) >= 0)
    owners.extend(int(f.owner) for f in fleets if int(f.owner) >= 0)
    return max(2, max(owners, default=0) + 1)


def global_features(obs: Any, planets: list[Any], fleets: list[Any], player: int) -> tuple[float, ...]:
    step = float(obs_get(obs, "step", 0) or 0)
    counts = owner_counts(planets, fleets, player)
    player_count = player_count_from_state(planets, fleets, player)
    return (
        clip(step / TOTAL_STEPS),
        clip(float(player_count) / 4.0),
        clip(float(len(planets)) / 44.0),
        clip(float(len(fleets)) / 180.0),
        clip(counts["owned_count"] / 32.0),
        clip(counts["neutral_count"] / 40.0),
        clip(counts["enemy_count"] / 32.0),
        clip(counts["own_ships"] / 2200.0),
        clip(counts["enemy_ships"] / 2200.0),
        clip((counts["own_prod"] - counts["enemy_prod"]) / 160.0),
    )


def quadrant_one_hot(planet: Any) -> tuple[float, float, float, float]:
    q = int(live_agent._quadrant(planet))
    return tuple(1.0 if idx == q else 0.0 for idx in range(4))


def planet_features(planet: Any, player: int) -> tuple[float, ...]:
    owner = int(planet.owner)
    return (
        clip(float(planet.ships) / 220.0),
        clip(float(planet.production) / 5.0),
        clip(float(planet.radius) / 10.0),
        clip(float(planet.x) / 100.0),
        clip(float(planet.y) / 100.0),
        1.0 if live_agent._is_static(planet) else 0.0,
        1.0 if live_agent._is_big(planet) else 0.0,
        1.0 if owner == int(player) else 0.0,
        1.0 if owner == live_agent.NEUTRAL else 0.0,
        *quadrant_one_hot(planet),
    )


def semantic_ship_count(
    kind: str,
    source: Any,
    target: Any,
    player: int,
    planets: list[Any],
    fleets: list[Any],
    angular_velocity: float,
) -> int:
    available = max(0, int(source.ships))
    if available <= 0:
        return 0
    if int(target.owner) == live_agent.NEUTRAL:
        need = live_agent._planned_capture_need(source, target, angular_velocity)
    else:
        need = live_agent._offensive_capture_need(source, target, player, planets, fleets, angular_velocity)

    if kind == "send_all":
        return available
    if kind == "kill_at_arrival":
        return min(available, max(1, int(need)))
    if kind == "hold":
        hold_buffer = max(2, int(math.ceil(float(target.production) * 8.0)))
        return min(available, max(1, int(need) + hold_buffer))
    if kind == "sortie":
        keep = max(6, min(28, int(source.production) * 5 + 4))
        return min(available, max(1, available - keep))
    return 0


def candidate_features(
    source: Any,
    target: Any,
    kind: str,
    ships: int,
    eta: float,
    player: int,
    planets: list[Any],
    fleets: list[Any],
    angular_velocity: float,
) -> tuple[float, ...]:
    if int(target.owner) == live_agent.NEUTRAL:
        need = live_agent._planned_capture_need(source, target, angular_velocity)
    else:
        need = live_agent._offensive_capture_need(source, target, player, planets, fleets, angular_velocity)
    distance = live_agent._distance(source, target)
    source_q = int(live_agent._quadrant(source))
    target_q = int(live_agent._quadrant(target))
    same_quadrant = 1.0 if source_q == target_q else 0.0
    target_enemy = 1.0 if int(target.owner) not in (int(player), live_agent.NEUTRAL) else 0.0
    target_neutral = 1.0 if int(target.owner) == live_agent.NEUTRAL else 0.0
    margin = safe_div(float(ships) - float(need), max(12.0, float(need)))
    action_one_hot = tuple(1.0 if ACTION_TO_INDEX[kind] == idx else 0.0 for idx in range(len(ACTION_KINDS)))
    relation = (
        clip(distance / 120.0),
        clip(float(eta) / 24.0),
        clip(float(ships) / 220.0),
        clip(float(ships) / max(1.0, float(source.ships))),
        clip(margin),
        same_quadrant,
        target_enemy,
        target_neutral,
    )
    return (
        *planet_features(source, player),
        *planet_features(target, player),
        *relation,
        *action_one_hot,
    )


def build_candidates(
    obs: Any,
    player: int,
    *,
    max_targets: int,
    horizon: float,
) -> tuple[tuple[float, ...], dict[int, tuple[float, ...]], dict[int, list[Candidate]]]:
    parsed_player, planets, fleets, angular_velocity, _comets = live_agent._parse(obs)
    player = int(parsed_player if parsed_player is not None else player)
    glob = global_features(obs, planets, fleets, player)
    owned = [p for p in planets if int(p.owner) == int(player) and int(p.ships) > 1]
    targets = [p for p in planets if int(p.owner) != int(player)]
    by_source: dict[int, list[Candidate]] = {}
    source_features: dict[int, tuple[float, ...]] = {}

    for source in owned:
        source_features[int(source.id)] = planet_features(source, player)
        ranked_targets = []
        probe_ships = max(1, min(int(source.ships), 24))
        for target in targets:
            if int(target.id) == int(source.id):
                continue
            try:
                measurement = live_agent._attack_measurement(source, target, probe_ships, angular_velocity, planets=planets)
            except Exception:
                continue
            if not measurement.clear or float(measurement.eta) > float(horizon):
                continue
            target_enemy = int(target.owner) not in (int(player), live_agent.NEUTRAL)
            score = (
                float(measurement.eta),
                0 if target_enemy else 1,
                int(target.ships),
                -int(target.production),
                live_agent._distance(source, target),
            )
            ranked_targets.append((score, target))

        ranked_targets.sort(key=lambda row: row[0])
        candidates: list[Candidate] = []
        seen: set[tuple[int, str, int]] = set()
        for _score, target in ranked_targets[: max(1, int(max_targets))]:
            for kind in ACTION_KINDS:
                ships = semantic_ship_count(kind, source, target, player, planets, fleets, angular_velocity)
                if ships <= 0 or ships > int(source.ships):
                    continue
                marker = (int(target.id), kind, int(ships))
                if marker in seen:
                    continue
                seen.add(marker)
                try:
                    measurement = live_agent._attack_measurement(source, target, ships, angular_velocity, planets=planets)
                except Exception:
                    continue
                if not measurement.clear or float(measurement.eta) > float(horizon):
                    continue
                features = candidate_features(
                    source,
                    target,
                    kind,
                    ships,
                    float(measurement.eta),
                    player,
                    planets,
                    fleets,
                    angular_velocity,
                )
                candidates.append(
                    Candidate(
                        source_id=int(source.id),
                        target_id=int(target.id),
                        kind=kind,
                        ships=int(ships),
                        angle=float(measurement.angle),
                        eta=float(measurement.eta),
                        features=features,
                    )
                )
        by_source[int(source.id)] = candidates

    return glob, source_features, by_source


class SemanticPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 192):
        super().__init__()
        self.edge_net = nn.Sequential(
            nn.Linear(EDGE_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.noop_net = nn.Sequential(
            nn.Linear(GLOBAL_DIM + SOURCE_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_net = nn.Sequential(
            nn.Linear(GLOBAL_DIM, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def edge_logits(self, edge_features: torch.Tensor) -> torch.Tensor:
        return self.edge_net(edge_features).squeeze(-1)

    def noop_logit(self, global_features_t: torch.Tensor, source_features_t: torch.Tensor) -> torch.Tensor:
        return self.noop_net(torch.cat([global_features_t, source_features_t], dim=-1)).squeeze(-1)

    def value(self, global_features_t: torch.Tensor) -> torch.Tensor:
        return self.value_net(global_features_t).squeeze(-1)


def tensor_row(values: tuple[float, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32, device=device).view(1, -1)


def select_or_evaluate(
    model: SemanticPolicy,
    obs: Any,
    player: int,
    *,
    device: torch.device,
    max_targets: int,
    horizon: float,
    temperature: float,
    decisions: list[SourceDecision] | None = None,
) -> tuple[list[list[Any]], list[SourceDecision], torch.Tensor, torch.Tensor, torch.Tensor] | None:
    glob, source_features, candidates_by_source = build_candidates(
        obs,
        player,
        max_targets=max_targets,
        horizon=horizon,
    )
    global_t = tensor_row(glob, device)
    value = model.value(global_t).reshape(())

    decisions_by_source = {int(row.source_id): row.selected_key for row in decisions or []}
    moves: list[list[Any]] = []
    output_decisions: list[SourceDecision] = []
    logprobs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    for source_id in sorted(source_features):
        source_t = tensor_row(source_features[source_id], device)
        noop = model.noop_logit(global_t, source_t).reshape(1)
        candidates = candidates_by_source.get(source_id, [])
        if candidates:
            edge_t = torch.tensor(
                [candidate.features for candidate in candidates],
                dtype=torch.float32,
                device=device,
            )
            logits = torch.cat([noop, model.edge_logits(edge_t)], dim=0)
        else:
            logits = noop

        dist = Categorical(logits=logits / max(1e-4, float(temperature)))
        if decisions is None:
            selected = dist.sample()
            selected_index = int(selected.item())
            selected_key = None
            if selected_index > 0 and selected_index - 1 < len(candidates):
                candidate = candidates[selected_index - 1]
                moves.append(candidate.move())
                selected_key = candidate.key
        else:
            selected_key = decisions_by_source.get(source_id)
            if selected_key is None:
                selected_index = 0
            else:
                index_by_key = {candidate.key: idx + 1 for idx, candidate in enumerate(candidates)}
                if selected_key not in index_by_key:
                    return None
                selected_index = index_by_key[selected_key]
            selected = torch.tensor(selected_index, dtype=torch.long, device=device)

        output_decisions.append(SourceDecision(source_id=source_id, selected_key=selected_key))
        logprobs.append(dist.log_prob(selected))
        entropies.append(dist.entropy())

    if not logprobs:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return moves, output_decisions, zero, zero, value

    return moves, output_decisions, torch.stack(logprobs).sum(), torch.stack(entropies).sum(), value


class PolicyAgent:
    def __init__(
        self,
        model: SemanticPolicy,
        transitions: list[Transition] | None,
        *,
        device: torch.device,
        max_targets: int,
        horizon: float,
        temperature: float,
    ):
        self.model = model
        self.transitions = transitions
        self.device = device
        self.max_targets = max_targets
        self.horizon = horizon
        self.temperature = temperature

    def __call__(self, obs: Any, config: Any = None) -> list[list[Any]]:
        del config
        player = int(obs_get(obs, "player", 0) or 0)
        with torch.no_grad():
            result = select_or_evaluate(
                self.model,
                obs,
                player,
                device=self.device,
                max_targets=self.max_targets,
                horizon=self.horizon,
                temperature=self.temperature,
            )
        if result is None:
            return []
        moves, decisions, logprob, _entropy, value = result
        if self.transitions is not None:
            self.transitions.append(
                Transition(
                    obs=serialize_obs(obs),
                    player=player,
                    decisions=decisions,
                    old_logprob=float(logprob.detach().cpu().item()),
                    old_value=float(value.detach().cpu().item()),
                )
            )
        return moves


def final_shaped_rewards(final_step: list[Any]) -> list[float]:
    raw_rewards = [extract_reward(state) for state in final_step]
    best = max(raw_rewards)
    winners = [idx for idx, reward in enumerate(raw_rewards) if reward == best]
    player_count = len(raw_rewards)
    if len(winners) != 1:
        return [0.0 for _ in raw_rewards]
    winner = winners[0]
    loser_value = -1.0 if player_count <= 2 else -1.0 / float(player_count - 1)
    return [1.0 if idx == winner else loser_value for idx in range(player_count)]


def run_episode(
    model: SemanticPolicy,
    *,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    opponent_model: SemanticPolicy | None = None,
) -> tuple[EpisodeResult, list[Transition]]:
    transitions: list[Transition] = []
    agents = []
    player_count = int(args.players)

    if opponent_model is None or args.opponent_mode == "self":
        for _idx in range(player_count):
            agents.append(
                PolicyAgent(
                    model,
                    transitions,
                    device=device,
                    max_targets=args.max_targets,
                    horizon=args.horizon,
                    temperature=args.temperature,
                )
            )
    else:
        train_slot = seed % player_count
        for idx in range(player_count):
            active_model = model if idx == train_slot else opponent_model
            active_transitions = transitions if idx == train_slot else None
            agents.append(
                PolicyAgent(
                    active_model,
                    active_transitions,
                    device=device,
                    max_targets=args.max_targets,
                    horizon=args.horizon,
                    temperature=args.temperature,
                )
            )

    env = make(
        "orbit_wars",
        configuration={"seed": int(seed), "randomSeed": int(seed)},
        debug=False,
    )
    env.run(agents)
    final_step = env.steps[-1]
    shaped = final_shaped_rewards(final_step)
    for transition in transitions:
        if 0 <= int(transition.player) < len(shaped):
            transition.reward = float(shaped[int(transition.player)])

    rewards = [extract_reward(state) for state in final_step]
    best = max(rewards)
    winners = [idx for idx, reward in enumerate(rewards) if reward == best]
    result = EpisodeResult(
        seed=int(seed),
        player_count=player_count,
        rewards=rewards,
        shaped_rewards=shaped,
        statuses=[extract_status(state) for state in final_step],
        winners=winners,
        steps=len(env.steps),
        transitions=len(transitions),
    )
    return result, transitions


def ppo_update(
    model: SemanticPolicy,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    if not transitions:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "valid": 0.0}

    old_logprobs = torch.tensor([row.old_logprob for row in transitions], dtype=torch.float32, device=device)
    old_values = torch.tensor([row.old_value for row in transitions], dtype=torch.float32, device=device)
    returns = torch.tensor([row.reward for row in transitions], dtype=torch.float32, device=device)
    advantages = returns - old_values
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

    indices = list(range(len(transitions)))
    total_loss = 0.0
    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    total_valid = 0
    updates = 0

    for _epoch in range(int(args.ppo_epochs)):
        random.shuffle(indices)
        for start in range(0, len(indices), int(args.batch_size)):
            batch_indices = indices[start : start + int(args.batch_size)]
            logprob_rows = []
            value_rows = []
            entropy_rows = []
            old_batch = []
            return_batch = []
            advantage_batch = []
            for idx in batch_indices:
                row = transitions[idx]
                evaluated = select_or_evaluate(
                    model,
                    row.obs,
                    row.player,
                    device=device,
                    max_targets=args.max_targets,
                    horizon=args.horizon,
                    temperature=args.temperature,
                    decisions=row.decisions,
                )
                if evaluated is None:
                    continue
                _moves, _decisions, logprob, entropy, value = evaluated
                logprob_rows.append(logprob)
                entropy_rows.append(entropy)
                value_rows.append(value)
                old_batch.append(old_logprobs[idx])
                return_batch.append(returns[idx])
                advantage_batch.append(advantages[idx])

            if not logprob_rows:
                continue

            new_logprobs = torch.stack(logprob_rows)
            new_values = torch.stack(value_rows)
            entropies = torch.stack(entropy_rows)
            old_t = torch.stack(old_batch).detach()
            returns_t = torch.stack(return_batch).detach()
            advantages_t = torch.stack(advantage_batch).detach()

            ratio = torch.exp(new_logprobs - old_t)
            unclipped = ratio * advantages_t
            clipped = torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef) * advantages_t
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(new_values, returns_t)
            entropy = entropies.mean()
            loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            batch_size = len(logprob_rows)
            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_policy += float(policy_loss.detach().cpu().item()) * batch_size
            total_value += float(value_loss.detach().cpu().item()) * batch_size
            total_entropy += float(entropy.detach().cpu().item()) * batch_size
            total_valid += batch_size
            updates += 1

    denom = max(1, total_valid)
    return {
        "loss": total_loss / denom,
        "policy_loss": total_policy / denom,
        "value_loss": total_value / denom,
        "entropy": total_entropy / denom,
        "valid": float(total_valid),
        "optimizer_steps": float(updates),
    }


def save_checkpoint(
    path: Path,
    model: SemanticPolicy,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    stats: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": serializable_args(args),
            "stats": stats,
            "dims": {
                "global_dim": GLOBAL_DIM,
                "source_dim": SOURCE_DIM,
                "edge_dim": EDGE_DIM,
                "actions": ACTION_KINDS,
            },
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device, hidden_dim: int) -> SemanticPolicy:
    model = SemanticPolicy(hidden_dim=hidden_dim).to(device)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("policy_update_*.pt"))
    return candidates[-1] if candidates else None


def sample_pool_checkpoint(checkpoint_dir: Path, rng: random.Random) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("policy_update_*.pt"))
    if not candidates:
        return None
    # Cheap PFSP-ish start: bias toward recent, harder checkpoints without
    # needing a separate win-rate estimator yet.
    weights = np.linspace(1.0, 3.0, num=len(candidates), dtype=np.float64)
    weights = weights / weights.sum()
    index = int(rng.choices(range(len(candidates)), weights=weights.tolist(), k=1)[0])
    return candidates[index]


def make_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    run_dir = RUN_ROOT / f"{timestamp}-semantic-ppo-selfplay"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    serialized: dict[str, Any] = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def write_readme(run_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Semantic PPO Self-Play Run",
        "",
        "This run is produced by `scripts/self_play_ppo.py`.",
        "",
        "## Method",
        "",
        "- Policy: MLP scorer over semantic source-target actions.",
        "- Actions: `send_all`, `sortie`, `hold`, `kill_at_arrival`, plus per-source no-op.",
        "- Rollouts: Kaggle Python `orbit_wars` environment.",
        "- Update: PPO over terminal win/loss rewards.",
        "- Opponents: current-policy self-play by default; optional recent-checkpoint pool.",
        "",
        "## Arguments",
        "",
        "```json",
        json.dumps(serializable_args(args), indent=2, sort_keys=True),
        "```",
        "",
        "## Files",
        "",
        "- `metrics.jsonl`: one row per PPO update.",
        "- `episodes.jsonl`: one row per episode.",
        "- `checkpoints/policy_update_*.pt`: policy and optimizer checkpoints.",
    ]
    (run_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    run_dir = make_run_dir(args)
    checkpoint_dir = run_dir / "checkpoints"
    write_readme(run_dir, args)

    model = SemanticPolicy(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    resume_path = Path(args.resume_from) if args.resume_from else latest_checkpoint(checkpoint_dir)
    if resume_path is not None and resume_path.exists() and args.resume:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    rng = random.Random(args.seed)
    global_episode = 0
    for update_idx in range(1, int(args.updates) + 1):
        model.eval()
        transitions: list[Transition] = []
        episode_rows: list[EpisodeResult] = []
        for game_idx in range(int(args.games_per_update)):
            seed = int(args.seed + update_idx * 100000 + game_idx)
            opponent_model = None
            if args.opponent_mode == "pool":
                selected = sample_pool_checkpoint(checkpoint_dir, rng)
                if selected is not None:
                    opponent_model = load_checkpoint(selected, device, args.hidden_dim)
            episode, rows = run_episode(
                model,
                args=args,
                device=device,
                seed=seed,
                opponent_model=opponent_model,
            )
            transitions.extend(rows)
            episode_rows.append(episode)
            global_episode += 1
            append_jsonl(run_dir / "episodes.jsonl", {"update": update_idx, "episode": global_episode, **asdict(episode)})

        model.train()
        metrics = ppo_update(model, optimizer, transitions, args=args, device=device)
        win_rate = float(np.mean([1.0 if 0 in row.winners and len(row.winners) == 1 else 0.0 for row in episode_rows])) if episode_rows else 0.0
        mean_steps = float(np.mean([row.steps for row in episode_rows])) if episode_rows else 0.0
        mean_reward = float(np.mean([row.shaped_rewards[0] for row in episode_rows if row.shaped_rewards])) if episode_rows else 0.0
        row = {
            "update": update_idx,
            "episodes": len(episode_rows),
            "transitions": len(transitions),
            "mean_steps": mean_steps,
            "slot0_win_rate": win_rate,
            "slot0_mean_reward": mean_reward,
            **metrics,
        }
        append_jsonl(run_dir / "metrics.jsonl", row)
        print(json.dumps(row, sort_keys=True))

        if update_idx % int(args.save_every) == 0 or update_idx == int(args.updates):
            save_checkpoint(
                checkpoint_dir / f"policy_update_{update_idx:06d}.pt",
                model,
                optimizer,
                args,
                row,
            )

    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a starter semantic-action PPO self-play policy.")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--games-per-update", type=int, default=2)
    parser.add_argument("--players", type=int, choices=(2, 4), default=2)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--horizon", type=float, default=20.0)
    parser.add_argument("--max-targets", type=int, default=8)
    parser.add_argument("--opponent-mode", choices=("self", "pool"), default="self")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = train(args)
    print(f"self-play run: {run_dir}")


if __name__ == "__main__":
    main()
