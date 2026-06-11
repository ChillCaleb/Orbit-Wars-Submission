import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from kaggle_environments import make

from agent_lab import (
    capture_events,
    clean_moves,
    extract_observation,
    extract_reward,
    extract_status,
    make_recording_agent,
    new_player_stats,
    serializable_stats,
    summarize_game,
)
from tactical_features import (
    action_feature_vector,
    action_feature_vector_for_state,
    action_features,
    action_penalty_profile_for_state,
    current_control_half,
    distance,
    fleets_from_obs,
    infer_phase_from_state,
    infer_role_assignments_from_state,
    is_static,
    numeric_quadrant_array,
    obs_get,
    overtake_profile_for_target,
    owner_control_scores,
    planets_from_obs,
    quadrant_index,
    role_confidence_map_from_state,
    role_scores,
    same_equator_side,
)
from watch_match import build_lineup


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "TRAINING_RUNS"
TOTAL_STEPS = 500.0

WITH_MINE_LINEUPS = (
    ("mine", "1200"),
    ("mine", "smith"),
    ("mine", "1039"),
    ("mine", "smith", "1039", "1200"),
)

WITHOUT_MINE_LINEUPS = (
    ("smith", "1039"),
    ("smith", "1200"),
    ("1039", "1200"),
    ("smith", "1039", "1200", "random"),
)

WORKER_LINEUP_CACHE = {}
RANKING_DEFAULT_TOP_K = 5


def starter_agent(obs, _rng=None):
    from agent_lab import starter_agent as local_starter

    return local_starter(obs, _rng)


def flatten_quadrant_features(obs, player, player_count):
    rows = numeric_quadrant_array(obs, player=player, player_count=player_count)
    values = []
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
    for row in rows:
        for idx, value in enumerate(row):
            values.append(float(value) / scales[idx])
    return values


def flatten_role_features(obs, player):
    labels = role_scores(obs, player=player, top_n=1)

    def first_score(key, score_name):
        items = labels.get(key, [])
        if not items:
            return 0.0
        return float(items[0].get(score_name, 0.0)) / 12.0

    return [
        len(labels.get("established_quadrants", [])) / 4.0,
        first_score("anchor_candidates", "anchor_score"),
        first_score("feeder_candidates", "feeder_score"),
        first_score("sweeper_candidates", "sweeper_score"),
        first_score("strike_stage_candidates", "strike_stage_score"),
    ]


def empty_tendency():
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


def flatten_tendency(tendency):
    launches = max(1, tendency["launches"])
    return [
        tendency["launches"] / 400.0,
        tendency["ships_launched"] / 20000.0,
        tendency["neutral_targets"] / launches,
        tendency["enemy_targets"] / launches,
        tendency["friendly_targets"] / launches,
        tendency["static_targets"] / launches,
        tendency["rotating_targets"] / launches,
        tendency["central_rotating_big"] / launches,
        tendency["central_rotating_small"] / launches,
        tendency["captures"] / 80.0,
        tendency["losses"] / 80.0,
    ]


def sample_features(obs, player, player_count, tendency, category):
    step = float(obs_get(obs, "step", 0) or 0)
    features = [
        step / TOTAL_STEPS,
        player_count / 4.0,
        1.0 if category == "with_mine" else 0.0,
        1.0 if player_count == 4 else 0.0,
    ]
    features.extend(flatten_quadrant_features(obs, player, player_count))
    features.extend(flatten_role_features(obs, player))
    features.extend(flatten_tendency(tendency))
    return features


def call_builtin_or_agent(agent, obs, config, rng):
    if isinstance(agent, str):
        if agent == "starter":
            return starter_agent(obs, rng)
        if agent == "random":
            from agent_lab import random_agent

            return random_agent(obs, rng)
        return []
    try:
        return agent(obs, config)
    except TypeError:
        return agent(obs)


def make_training_recording_agent(agent, player_index, label, calls_by_step, rng):
    def wrapped(obs, config=None):
        step = int(obs_get(obs, "step", 0) or 0)
        error = None
        try:
            moves = clean_moves(call_builtin_or_agent(agent, obs, config, rng))
        except Exception as exc:
            moves = []
            error = f"{type(exc).__name__}: {exc}"
        calls_by_step[step][player_index] = {
            "label": label,
            "player": player_index,
            "moves": moves,
            "error": error,
        }
        return moves

    return wrapped


def run_env_game(agents, labels, seed):
    rng = random.Random(seed)
    calls_by_step = defaultdict(dict)
    wrapped_agents = [
        make_training_recording_agent(agent, idx, labels[idx], calls_by_step, rng)
        for idx, agent in enumerate(agents)
    ]
    env = make(
        "orbit_wars",
        configuration={"seed": int(seed), "randomSeed": int(seed)},
        debug=False,
    )
    env.run(wrapped_agents)
    return env, calls_by_step


def update_action_tendencies(tendencies, step_states, calls_by_step, labels):
    board_obs = extract_observation(step_states[0])
    step = int(obs_get(board_obs, "step", 0) or 0)
    for player, label in enumerate(labels):
        obs = extract_observation(step_states[player])
        calls = calls_by_step.get(step, {}).get(player, {"moves": []})
        for move in calls.get("moves", []):
            info = action_features(move, obs, player=player)
            tendencies[label]["launches"] += 1
            tendencies[label]["ships_launched"] += int(info["ships"])
            if info["target_owner_group"] == "neutral":
                tendencies[label]["neutral_targets"] += 1
            elif info["target_owner_group"] == "enemy":
                tendencies[label]["enemy_targets"] += 1
            elif info["target_owner_group"] == "friendly":
                tendencies[label]["friendly_targets"] += 1
            if info["target_kind"] == "static":
                tendencies[label]["static_targets"] += 1
            elif info["target_kind"] == "rotating":
                tendencies[label]["rotating_targets"] += 1
            if info["central_rotating_big"]:
                tendencies[label]["central_rotating_big"] += 1
            if info["central_rotating_small"]:
                tendencies[label]["central_rotating_small"] += 1


def update_capture_tendencies(tendencies, before_step, after_step, labels):
    before_obs = extract_observation(before_step[0])
    after_obs = extract_observation(after_step[0])
    for event in capture_events(before_obs, after_obs):
        to_owner = int(event["to_owner"])
        from_owner = int(event["from_owner"])
        if 0 <= to_owner < len(labels):
            tendencies[labels[to_owner]]["captures"] += 1
        if 0 <= from_owner < len(labels):
            tendencies[labels[from_owner]]["losses"] += 1


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _shaped_action_target(winner_flag, penalty_profile):
    return _clamp01((0.2 + 0.8 * float(winner_flag)) * float(penalty_profile["quality_score"]))


def _overtake_focus_weight(overtake_profile):
    return (
        1.0
        + 2.40 * float(overtake_profile["overtake_bonus"])
        + 1.35 * float(overtake_profile["board_ownership_bonus"])
        + 0.90 * float(overtake_profile["projected_overtake_count"])
        + 0.60 * float(overtake_profile["leader_target"])
        + 0.40 * float(overtake_profile["ahead_owner_target"])
    )


def _overtake_action_boost(overtake_profile):
    return _clamp01(
        0.30 * float(overtake_profile["overtake_bonus"])
        + 0.18 * float(overtake_profile["board_ownership_bonus"])
        + 0.10 * float(overtake_profile["projected_overtake_count"])
        + 0.08 * float(overtake_profile["leader_target"])
        + 0.05 * float(overtake_profile["ahead_owner_target"])
    )


def _ranking_candidate_sort_key(source, candidate, phase_name):
    source_quadrant = quadrant_index(source)
    candidate_quadrant = quadrant_index(candidate)
    static_penalty = 0 if phase_name in ("initiation", "expansion") and is_static(candidate) else 1
    enemy_penalty = 0 if phase_name == "attack" and int(candidate.owner) != -1 else 1
    return (
        candidate_quadrant != source_quadrant,
        0 if int(candidate.owner) == -1 else 1,
        static_penalty,
        enemy_penalty,
        int(candidate.ships),
        distance(source, candidate),
        -int(candidate.production),
        0 if is_static(candidate) else 1,
        int(candidate.id),
    )


def _shortlist_ranking_candidates(planets, fleets, player, source, chosen_target, source_role, phase_name, top_k, owner_scores=None):
    candidates = [
        planet
        for planet in planets
        if int(planet.id) != int(source.id)
        and int(planet.owner) != int(player)
        and same_equator_side(source, planet)
    ]
    if not candidates:
        return []

    control_half = current_control_half(planets, fleets, player)
    if control_half:
        if source_role == "attacker":
            filtered = [planet for planet in candidates if quadrant_index(planet) in control_half]
        else:
            filtered = [
                planet
                for planet in candidates
                if quadrant_index(planet) in control_half
                and (int(planet.owner) == -1 or quadrant_index(planet) == quadrant_index(source))
            ]
        candidates = filtered or candidates

    ordered = sorted(candidates, key=lambda candidate: _ranking_candidate_sort_key(source, candidate, phase_name))
    same_quadrant = [candidate for candidate in ordered if quadrant_index(candidate) == quadrant_index(source)]
    static_targets = [candidate for candidate in ordered if is_static(candidate)]
    chosen_class = []
    if chosen_target is not None:
        chosen_class = [candidate for candidate in ordered if is_static(candidate) == is_static(chosen_target)]
    player_score = float(owner_scores.get(int(player), 0.0)) if owner_scores is not None else 0.0
    ahead_owner_targets = [
        candidate
        for candidate in ordered
        if int(candidate.owner) >= 0 and int(candidate.owner) != int(player) and float(owner_scores.get(int(candidate.owner), 0.0)) > player_score
    ] if owner_scores is not None else []
    overtake_targets = (
        sorted(
            ordered,
            key=lambda candidate: -overtake_profile_for_target(
                planets,
                fleets,
                player,
                candidate,
                owner_scores=owner_scores,
            )["overtake_bonus"],
        )
        if owner_scores is not None
        else []
    )

    shortlist = []
    seen = set()
    for candidate in ([chosen_target] if chosen_target is not None else []) + ahead_owner_targets[:top_k] + overtake_targets[:top_k] + same_quadrant[:top_k] + static_targets[:top_k] + chosen_class[:top_k] + ordered[:top_k]:
        if candidate is None or int(candidate.id) in seen:
            continue
        seen.add(int(candidate.id))
        shortlist.append(candidate)
        if len(shortlist) >= max(2, top_k):
            break
    return shortlist


def _ranking_pair_weight(winner_flag, chosen_profile, alt_profile, chosen_overtake, alt_overtake):
    opportunity_margin = max(
        0.0,
        float(chosen_profile["chosen_value"]) - float(alt_profile["chosen_value"]),
    )
    overtake_margin = max(
        0.0,
        float(chosen_overtake["overtake_bonus"]) - float(alt_overtake["overtake_bonus"]),
    )
    board_margin = max(
        0.0,
        float(chosen_overtake["board_ownership_bonus"]) - float(alt_overtake["board_ownership_bonus"]),
    )
    count_margin = max(
        0.0,
        float(chosen_overtake["projected_overtake_count"]) - float(alt_overtake["projected_overtake_count"]),
    )
    base_weight = (
        0.35
        + 0.30 * float(chosen_profile["quality_score"])
        + 0.20 * float(winner_flag)
        + 0.15 * min(1.0, opportunity_margin / 3.0)
        + 0.95 * overtake_margin
        + 0.55 * board_margin
        + 0.40 * count_margin
    )
    return base_weight * (0.80 + 0.25 * _overtake_focus_weight(chosen_overtake))


def game_samples(env, calls_by_step, labels, category, split, sample_stride, training_mode="ranking", ranking_top_k=RANKING_DEFAULT_TOP_K):
    final_rewards = [extract_reward(state) for state in env.steps[-1]]
    best_reward = max(final_rewards)
    winner_flags = [1 if reward == best_reward and final_rewards.count(best_reward) == 1 else 0 for reward in final_rewards]
    tendencies = {label: empty_tendency() for label in labels}
    rows = []
    targets = []
    groups = []
    weights = []
    meta_rows = []

    for step_index, step_states in enumerate(env.steps):
        update_action_tendencies(tendencies, step_states, calls_by_step, labels)
        if step_index + 1 < len(env.steps):
            update_capture_tendencies(tendencies, step_states, env.steps[step_index + 1], labels)

        if step_index % sample_stride != 0 and step_index != len(env.steps) - 1:
            continue

        for player, label in enumerate(labels):
            obs = extract_observation(step_states[player])
            planets = planets_from_obs(obs)
            fleets = fleets_from_obs(obs)
            player_count = max(2, len(labels))
            calls = calls_by_step.get(int(obs_get(obs, "step", step_index) or step_index), {}).get(player, {"moves": []})
            for move in calls.get("moves", []):
                info = action_features(move, obs, player=player)
                target = next((planet for planet in planets if planet.id == info.get("target_id")), None)
                source = next((planet for planet in planets if planet.id == info.get("source_id")), None)
                if source is None:
                    continue
                phase_name = infer_phase_from_state(planets, fleets, player, target=target)
                role_profiles = role_confidence_map_from_state(
                    planets,
                    fleets,
                    player,
                    phase_name=phase_name,
                )
                roles = infer_role_assignments_from_state(
                    planets,
                    fleets,
                    player,
                    phase_name=phase_name,
                    role_profiles=role_profiles,
                )
                source_role = roles.get(int(info["source_id"])) if info.get("source_id") is not None else "unknown"
                ships = int(info.get("ships", 0))
                owner_scores = owner_control_scores(planets, fleets, player_count=player_count)
                chosen_overtake = overtake_profile_for_target(
                    planets,
                    fleets,
                    player,
                    target,
                    owner_scores=owner_scores,
                    player_count=player_count,
                )
                overtake_focus_weight = _overtake_focus_weight(chosen_overtake)
                penalty_profile = action_penalty_profile_for_state(
                    planets,
                    fleets,
                    player,
                    source,
                    target,
                    ships,
                    source_role=source_role,
                    phase_name=phase_name,
                    action_angle=float(info.get("angle")) if info.get("angle") is not None else None,
                )
                chosen_features = action_feature_vector_for_state(
                    planets,
                    fleets,
                    player,
                    source,
                    target,
                    ships,
                    step=int(obs_get(obs, "step", step_index) or step_index),
                    angular_velocity=float(obs_get(obs, "angular_velocity", 0.0) or 0.0),
                    tendency=tendencies[label],
                    source_role=source_role,
                    phase_name=phase_name,
                    player_count=player_count,
                    role_profiles=role_profiles,
                    roles=roles,
                    owner_scores=owner_scores,
                    overtake_profile=chosen_overtake,
                    action_angle=float(info.get("angle")) if info.get("angle") is not None else None,
                )
                if not chosen_features:
                    continue
                action_target = _shaped_action_target(winner_flags[player], penalty_profile)
                action_target = _clamp01(action_target + _overtake_action_boost(chosen_overtake))

                if training_mode == "outcome":
                    sample_weight = overtake_focus_weight
                    rows.append(chosen_features)
                    targets.append(action_target)
                    groups.append(1 if category == "with_mine" else 0)
                    weights.append(sample_weight)
                    meta_rows.append(
                        {
                            "split": split,
                            "category": category,
                            "lineup": " ".join(labels),
                            "player": label,
                            "step": int(obs_get(obs, "step", step_index) or step_index),
                            "target_win": int(winner_flags[player]),
                            "role": source_role,
                            "phase": phase_name,
                            "target_owner_group": info.get("target_owner_group", "unknown"),
                            "target_quadrant": info.get("target_quadrant", "unknown"),
                            "action_target": round(action_target, 6),
                            "sample_weight": round(sample_weight, 6),
                            "sample_kind": "chosen_action",
                            "compare_target_id": "",
                            "sun_penalty": round(float(penalty_profile["sun_penalty"]), 6),
                            "long_flight_penalty": round(float(penalty_profile["long_flight_penalty"]), 6),
                            "opportunity_penalty": round(float(penalty_profile["opportunity_penalty"]), 6),
                            "overtake_bonus": round(float(chosen_overtake["overtake_bonus"]), 6),
                            "board_ownership_bonus": round(float(chosen_overtake["board_ownership_bonus"]), 6),
                            "projected_overtake_count": round(float(chosen_overtake["projected_overtake_count"]), 6),
                            "overtake_focus_weight": round(float(sample_weight), 6),
                        }
                    )
                    continue

                shortlist = _shortlist_ranking_candidates(
                    planets,
                    fleets,
                    player,
                    source,
                    target,
                    source_role,
                    phase_name,
                    ranking_top_k,
                    owner_scores=owner_scores,
                )
                for candidate in shortlist:
                    if target is not None and int(candidate.id) == int(target.id):
                        continue
                    alt_penalty = action_penalty_profile_for_state(
                        planets,
                        fleets,
                        player,
                        source,
                        candidate,
                        ships,
                        source_role=source_role,
                        phase_name=phase_name,
                    )
                    alt_overtake = overtake_profile_for_target(
                        planets,
                        fleets,
                        player,
                        candidate,
                        owner_scores=owner_scores,
                        player_count=player_count,
                    )
                    alt_features = action_feature_vector_for_state(
                        planets,
                        fleets,
                        player,
                        source,
                        candidate,
                        ships,
                        step=int(obs_get(obs, "step", step_index) or step_index),
                        angular_velocity=float(obs_get(obs, "angular_velocity", 0.0) or 0.0),
                        tendency=tendencies[label],
                        source_role=source_role,
                        phase_name=phase_name,
                        player_count=player_count,
                        role_profiles=role_profiles,
                        roles=roles,
                        owner_scores=owner_scores,
                        overtake_profile=alt_overtake,
                    )
                    if not alt_features or len(alt_features) != len(chosen_features):
                        continue

                    pair_weight = _ranking_pair_weight(
                        winner_flags[player],
                        penalty_profile,
                        alt_penalty,
                        chosen_overtake,
                        alt_overtake,
                    )
                    if pair_weight <= 0.0:
                        continue

                    positive_row = [float(a) - float(b) for a, b in zip(chosen_features, alt_features)]
                    negative_row = [float(b) - float(a) for a, b in zip(chosen_features, alt_features)]
                    base_meta = {
                        "split": split,
                        "category": category,
                        "lineup": " ".join(labels),
                        "player": label,
                        "step": int(obs_get(obs, "step", step_index) or step_index),
                        "target_win": int(winner_flags[player]),
                        "role": source_role,
                        "phase": phase_name,
                        "target_owner_group": info.get("target_owner_group", "unknown"),
                        "target_quadrant": info.get("target_quadrant", "unknown"),
                        "sample_weight": round(pair_weight, 6),
                        "compare_target_id": int(candidate.id),
                        "sun_penalty": round(float(penalty_profile["sun_penalty"]), 6),
                        "long_flight_penalty": round(float(penalty_profile["long_flight_penalty"]), 6),
                        "opportunity_penalty": round(float(penalty_profile["opportunity_penalty"]), 6),
                        "overtake_bonus": round(float(chosen_overtake["overtake_bonus"]), 6),
                        "board_ownership_bonus": round(float(chosen_overtake["board_ownership_bonus"]), 6),
                        "projected_overtake_count": round(float(chosen_overtake["projected_overtake_count"]), 6),
                        "overtake_focus_weight": round(float(overtake_focus_weight), 6),
                    }
                    rows.append(positive_row)
                    targets.append(1.0)
                    groups.append(1 if category == "with_mine" else 0)
                    weights.append(pair_weight)
                    meta_rows.append(
                        {
                            **base_meta,
                            "action_target": 1.0,
                            "sample_kind": "rank_pref",
                        }
                    )
                    rows.append(negative_row)
                    targets.append(0.0)
                    groups.append(1 if category == "with_mine" else 0)
                    weights.append(pair_weight)
                    meta_rows.append(
                        {
                            **base_meta,
                            "action_target": 0.0,
                            "sample_kind": "rank_inverse",
                        }
                    )

    return rows, targets, groups, weights, meta_rows


def lineup_for(category, index):
    choices = WITH_MINE_LINEUPS if category == "with_mine" else WITHOUT_MINE_LINEUPS
    return choices[index % len(choices)]


def build_tasks(args):
    tasks = []
    global_game_index = 0
    for category, total in (("without_mine", args.without_mine_games), ("with_mine", args.with_mine_games)):
        train_cutoff = int(round(total * args.train_ratio))
        for category_index in range(total):
            split = "train" if category_index < train_cutoff else "test"
            lineup_specs = lineup_for(category, category_index)
            tasks.append(
                {
                    "global_game_index": global_game_index,
                    "category_index": category_index,
                    "category": category,
                    "split": split,
                    "lineup_specs": lineup_specs,
                    "seed": args.seed + global_game_index,
                    "sample_stride": args.sample_stride,
                    "training_mode": args.training_mode,
                    "ranking_top_k": args.ranking_top_k,
                }
            )
            global_game_index += 1
    return tasks


def cached_worker_lineup(lineup_specs):
    key = tuple(lineup_specs)
    cached = WORKER_LINEUP_CACHE.get(key)
    if cached is not None:
        return cached
    agents, labels = build_lineup(list(lineup_specs))
    WORKER_LINEUP_CACHE[key] = (agents, labels)
    return agents, labels


def collect_one_game(task):
    agents, labels = cached_worker_lineup(task["lineup_specs"])
    env, calls_by_step = run_env_game(agents, labels, task["seed"])
    rows, targets, groups, weights, meta_rows = game_samples(
        env,
        calls_by_step,
        labels,
        task["category"],
        task["split"],
        task["sample_stride"],
        training_mode=task.get("training_mode", "ranking"),
        ranking_top_k=task.get("ranking_top_k", RANKING_DEFAULT_TOP_K),
    )
    final_rewards = [extract_reward(state) for state in env.steps[-1]]
    best = max(final_rewards)
    winners = [labels[idx] for idx, reward in enumerate(final_rewards) if reward == best]
    statuses = [extract_status(state) for state in env.steps[-1]]
    return {
        "task": task,
        "labels": labels,
        "rows": rows,
        "targets": targets,
        "groups": groups,
        "weights": weights,
        "meta_rows": meta_rows,
        "game_row": {
            "game_index": task["global_game_index"] + 1,
            "category": task["category"],
            "split": task["split"],
            "seed": task["seed"],
            "lineup": " ".join(labels),
            "rewards": json.dumps(final_rewards),
            "statuses": json.dumps(statuses),
            "winners": json.dumps(winners),
        },
    }


def collect_games(args, run_dir):
    train_x, train_y, train_group, train_w = [], [], [], []
    test_x, test_y, test_group, test_w = [], [], [], []
    meta_path = run_dir / "samples_meta.csv"
    values_path = run_dir / "sample_values.csv"
    game_log_path = run_dir / "game_log.csv"
    stats_by_lineup = Counter()
    stats_by_category = Counter()
    tasks = build_tasks(args)

    with (
        meta_path.open("w", newline="", encoding="utf-8") as meta_handle,
        values_path.open("w", newline="", encoding="utf-8") as values_handle,
        game_log_path.open("w", newline="", encoding="utf-8") as game_handle,
    ):
        meta_writer = csv.DictWriter(
            meta_handle,
            fieldnames=[
                "split",
                "category",
                "lineup",
                "player",
                "step",
                "target_win",
                "role",
                "phase",
                "target_owner_group",
                "target_quadrant",
                "action_target",
                "sample_weight",
                "sample_kind",
                "compare_target_id",
                "sun_penalty",
                "long_flight_penalty",
                "opportunity_penalty",
                "overtake_bonus",
                "board_ownership_bonus",
                "projected_overtake_count",
                "overtake_focus_weight",
            ],
        )
        values_writer = None
        game_writer = csv.DictWriter(
            game_handle,
            fieldnames=["game_index", "category", "split", "seed", "lineup", "rewards", "statuses", "winners"],
        )
        meta_writer.writeheader()
        game_writer.writeheader()

        completed = 0

        def absorb_result(result):
            nonlocal values_writer
            task = result["task"]
            if task["split"] == "train":
                train_x.extend(result["rows"])
                train_y.extend(result["targets"])
                train_group.extend(result["groups"])
                train_w.extend(result["weights"])
            else:
                test_x.extend(result["rows"])
                test_y.extend(result["targets"])
                test_group.extend(result["groups"])
                test_w.extend(result["weights"])

            if values_writer is None and result["rows"]:
                values_writer = csv.DictWriter(
                    values_handle,
                    fieldnames=[
                        "split",
                        "category",
                        "lineup",
                        "player",
                        "step",
                        "target_win",
                        "role",
                        "phase",
                        "target_owner_group",
                        "target_quadrant",
                        "action_target",
                        "sample_weight",
                        "sample_kind",
                        "compare_target_id",
                        "sun_penalty",
                        "long_flight_penalty",
                        "opportunity_penalty",
                        "overtake_bonus",
                        "board_ownership_bonus",
                        "projected_overtake_count",
                        "overtake_focus_weight",
                        "group",
                    ]
                    + [f"f{idx}" for idx in range(len(result["rows"][0]))],
                )
                values_writer.writeheader()

            for row, target, group, sample_weight, meta in zip(
                result["rows"],
                result["targets"],
                result["groups"],
                result["weights"],
                result["meta_rows"],
            ):
                meta_writer.writerow(meta)
                if values_writer is not None:
                    values_writer.writerow(
                        {
                            **meta,
                            "group": int(group),
                            "sample_weight": round(float(sample_weight), 6),
                            **{f"f{idx}": value for idx, value in enumerate(row)},
                        }
                    )
            game_writer.writerow(result["game_row"])
            meta_handle.flush()
            values_handle.flush()
            game_handle.flush()
            stats_by_lineup[result["game_row"]["lineup"]] += 1
            stats_by_category[f"{task['category']}_{task['split']}"] += 1

        if args.workers <= 1:
            for task in tasks:
                result = collect_one_game(task)
                absorb_result(result)
                completed += 1
                if completed == 1 or completed % args.progress_every == 0 or completed == len(tasks):
                    print(
                        f"[{completed}/{len(tasks)}] latest={task['category']} {task['split']} "
                        f"seed={task['seed']} lineup={result['game_row']['lineup']}",
                        flush=True,
                    )
        else:
            print(f"Collecting {len(tasks)} games with {args.workers} workers", flush=True)
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(collect_one_game, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    absorb_result(result)
                    completed += 1
                    if completed == 1 or completed % args.progress_every == 0 or completed == len(tasks):
                        task = result["task"]
                        print(
                            f"[{completed}/{len(tasks)}] latest={task['category']} {task['split']} "
                            f"seed={task['seed']} lineup={result['game_row']['lineup']}",
                            flush=True,
                        )

    arrays = {
        "train_x": np.asarray(train_x, dtype=np.float32),
        "train_y": np.asarray(train_y, dtype=np.float32),
        "train_group": np.asarray(train_group, dtype=np.int8),
        "train_w": np.asarray(train_w, dtype=np.float32),
        "test_x": np.asarray(test_x, dtype=np.float32),
        "test_y": np.asarray(test_y, dtype=np.float32),
        "test_group": np.asarray(test_group, dtype=np.int8),
        "test_w": np.asarray(test_w, dtype=np.float32),
    }
    np.savez_compressed(run_dir / "dataset.npz", **arrays)
    return arrays, {
        "lineups": dict(stats_by_lineup),
        "categories": dict(stats_by_category),
    }


def load_arrays_from_sample_values(run_dir):
    values_path = run_dir / "sample_values.csv"
    if not values_path.exists():
        raise FileNotFoundError(f"No sample_values.csv found in {run_dir}")

    train_x, train_y, train_group, train_w = [], [], [], []
    test_x, test_y, test_group, test_w = [], [], [], []
    with values_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        feature_names = [name for name in reader.fieldnames or [] if name.startswith("f")]
        feature_names.sort(key=lambda name: int(name[1:]))
        for row in reader:
            features = [float(row[name]) for name in feature_names]
            target = float(row.get("action_target", row["target_win"]))
            group = int(row["group"])
            sample_weight = float(row.get("sample_weight", 1.0))
            if row["split"] == "train":
                train_x.append(features)
                train_y.append(target)
                train_group.append(group)
                train_w.append(sample_weight)
            else:
                test_x.append(features)
                test_y.append(target)
                test_group.append(group)
                test_w.append(sample_weight)

    arrays = {
        "train_x": np.asarray(train_x, dtype=np.float32),
        "train_y": np.asarray(train_y, dtype=np.float32),
        "train_group": np.asarray(train_group, dtype=np.int8),
        "train_w": np.asarray(train_w, dtype=np.float32),
        "test_x": np.asarray(test_x, dtype=np.float32),
        "test_y": np.asarray(test_y, dtype=np.float32),
        "test_group": np.asarray(test_group, dtype=np.int8),
        "test_w": np.asarray(test_w, dtype=np.float32),
    }
    np.savez_compressed(run_dir / "dataset.npz", **arrays)
    return arrays


def load_arrays_from_dataset(run_dir):
    dataset_path = run_dir / "dataset.npz"
    if not dataset_path.exists():
        raise FileNotFoundError(f"No dataset.npz found in {run_dir}")
    with np.load(dataset_path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_arrays_from_run_dir(run_dir):
    dataset_path = run_dir / "dataset.npz"
    if dataset_path.exists():
        return load_arrays_from_dataset(run_dir)
    return load_arrays_from_sample_values(run_dir)


def _concat_array_pair(left, right):
    if left.size == 0:
        return np.asarray(right).copy()
    if right.size == 0:
        return np.asarray(left).copy()
    left = np.asarray(left)
    right = np.asarray(right)
    if left.ndim != right.ndim:
        raise ValueError(f"Array rank mismatch: {left.shape} vs {right.shape}")
    if left.ndim > 1 and left.shape[1:] != right.shape[1:]:
        raise ValueError(f"Array shape mismatch: {left.shape} vs {right.shape}")
    return np.concatenate([left, right], axis=0)


def concat_dataset_arrays(base_arrays, extra_arrays):
    merged = {}
    keys = sorted(set(base_arrays.keys()) | set(extra_arrays.keys()))
    for key in keys:
        if key not in base_arrays:
            merged[key] = np.asarray(extra_arrays[key]).copy()
            continue
        if key not in extra_arrays:
            merged[key] = np.asarray(base_arrays[key]).copy()
            continue
        merged[key] = _concat_array_pair(base_arrays[key], extra_arrays[key])
    return merged


def _resolve_local_path(path_value):
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_promoted_run_dir():
    root_model = ROOT / "model_weights.npz"
    if not root_model.exists():
        return None
    root_digest = _file_sha256(root_model)
    for candidate in sorted(ROOT.glob("TRAINING_RUNS/*/model_weights.npz"), reverse=True):
        try:
            if _file_sha256(candidate) == root_digest:
                return candidate.parent
        except OSError:
            continue
    return None


def _load_model_bundle(model_path):
    path = _resolve_local_path(model_path)
    if path is None or not path.exists():
        return None
    try:
        with np.load(path) as model:
            return {
                "path": str(path),
                "weights": np.asarray(model["weights"], dtype=np.float32),
                "bias": float(np.asarray(model["bias"], dtype=np.float32).reshape(-1)[0]),
                "mean": np.asarray(model["mean"], dtype=np.float32),
                "std": np.asarray(model["std"], dtype=np.float32),
            }
    except Exception:
        return None


def _adapt_model_to_normalization(model_bundle, mean, std):
    weights = np.asarray(model_bundle["weights"], dtype=np.float32)
    old_mean = np.asarray(model_bundle["mean"], dtype=np.float32)
    old_std = np.asarray(model_bundle["std"], dtype=np.float32).copy()
    new_mean = np.asarray(mean, dtype=np.float32)
    new_std = np.asarray(std, dtype=np.float32).copy()
    if weights.ndim != 1 or old_mean.shape != weights.shape or old_std.shape != weights.shape:
        raise ValueError("Stored model normalization shape mismatch")
    if new_mean.shape != weights.shape or new_std.shape != weights.shape:
        raise ValueError("Current normalization shape mismatch")
    old_std[old_std < 1e-6] = 1.0
    new_std[new_std < 1e-6] = 1.0
    scaled_weights = weights * (new_std / old_std)
    scaled_bias = float(model_bundle["bias"]) + float(np.sum(((new_mean - old_mean) / old_std) * weights))
    return scaled_weights.astype(np.float32), np.float32(scaled_bias)


def sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def bce_loss(pred, y, sample_weight=None):
    eps = 1e-7
    pred = np.clip(pred, eps, 1.0 - eps)
    losses = -(y * np.log(pred) + (1.0 - y) * np.log(1.0 - pred))
    if sample_weight is None:
        return float(np.mean(losses))
    weight = np.asarray(sample_weight, dtype=np.float32)
    denom = max(1e-6, float(weight.sum()))
    return float(np.sum(losses * weight) / denom)


def accuracy(pred, y, sample_weight=None):
    if y.size == 0:
        return 0.0
    correct = ((pred >= 0.5) == (y >= 0.5)).astype(np.float32)
    if sample_weight is None:
        return float(np.mean(correct))
    weight = np.asarray(sample_weight, dtype=np.float32)
    denom = max(1e-6, float(weight.sum()))
    return float(np.sum(correct * weight) / denom)


def train_logistic(arrays, args, run_dir, init_model_path=None):
    x = arrays["train_x"]
    y = arrays["train_y"]
    w = arrays.get("train_w")
    tx = arrays["test_x"]
    ty = arrays["test_y"]
    tw = arrays.get("test_w")
    if x.shape[0] == 0:
        raise ValueError("No training samples were collected")

    if args.training_mode == "ranking":
        mean = np.zeros(x.shape[1], dtype=np.float32)
        std = x.std(axis=0)
    else:
        mean = x.mean(axis=0)
        std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    xz = (x - mean) / std
    txz = (tx - mean) / std if tx.shape[0] else tx

    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0.0, 0.01, size=xz.shape[1]).astype(np.float32)
    bias = np.float32(0.0)
    training_info = {
        "warm_start_requested": bool(init_model_path),
        "warm_start_used": False,
        "warm_start_model": str(_resolve_local_path(init_model_path)) if init_model_path is not None else None,
        "warm_start_reason": "random_init",
    }
    if init_model_path is not None:
        model_bundle = _load_model_bundle(init_model_path)
        if model_bundle is None:
            training_info["warm_start_reason"] = "init_model_unavailable"
        elif int(model_bundle["weights"].shape[0]) != int(xz.shape[1]):
            training_info["warm_start_reason"] = (
                f"feature_dim_mismatch stored={model_bundle['weights'].shape[0]} current={xz.shape[1]}"
            )
        else:
            try:
                weights, bias = _adapt_model_to_normalization(model_bundle, mean, std)
                training_info["warm_start_used"] = True
                training_info["warm_start_model"] = str(model_bundle["path"])
                training_info["warm_start_reason"] = "initialized_from_existing_model"
            except ValueError as exc:
                training_info["warm_start_reason"] = str(exc)
    progress_path = run_dir / "training_progress.csv"
    batch_size = min(args.batch_size, xz.shape[0])
    pos_rate = max(1e-3, float(y.mean()))
    neg_rate = max(1e-3, 1.0 - pos_rate)
    pos_weight = neg_rate / pos_rate
    if w is None or w.shape[0] != xz.shape[0]:
        w = np.ones(xz.shape[0], dtype=np.float32)
    if tw is None or tw.shape[0] != tx.shape[0]:
        tw = np.ones(tx.shape[0], dtype=np.float32)

    with progress_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "test_loss", "train_acc", "test_acc", "lr"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(xz.shape[0])
            for start in range(0, xz.shape[0], batch_size):
                idx = order[start : start + batch_size]
                bx = xz[idx]
                by = y[idx]
                bw = w[idx]
                pred = sigmoid(bx @ weights + bias)
                sample_weight = bw * np.where(by > 0.5, pos_weight, 1.0)
                grad = (pred - by) * sample_weight
                denom = max(1.0, float(sample_weight.sum()))
                weights -= args.lr * ((bx.T @ grad) / denom + args.l2 * weights)
                if args.training_mode != "ranking":
                    bias -= np.float32(args.lr * (grad.sum() / denom))

            train_pred = sigmoid(xz @ weights + bias)
            test_pred = sigmoid(txz @ weights + bias) if txz.shape[0] else np.asarray([], dtype=np.float32)
            row = {
                "epoch": epoch,
                "train_loss": round(bce_loss(train_pred, y, w), 6),
                "test_loss": round(bce_loss(test_pred, ty, tw), 6) if ty.size else 0.0,
                "train_acc": round(accuracy(train_pred, y, w), 6),
                "test_acc": round(accuracy(test_pred, ty, tw), 6) if ty.size else 0.0,
                "lr": args.lr,
            }
            writer.writerow(row)
            if epoch == 1 or epoch % max(1, args.log_every) == 0 or epoch == args.epochs:
                print(
                    f"epoch={epoch} train_loss={row['train_loss']} test_loss={row['test_loss']} "
                    f"train_acc={row['train_acc']} test_acc={row['test_acc']}",
                    flush=True,
                )

    np.savez_compressed(
        run_dir / "model_weights.npz",
        weights=weights.astype(np.float32),
        bias=np.asarray([0.0 if args.training_mode == "ranking" else bias], dtype=np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
    )
    training_info["progress_path"] = str(progress_path)
    return progress_path, training_info


def evaluate_groups(arrays, run_dir):
    model = np.load(run_dir / "model_weights.npz")
    weights = model["weights"]
    bias = float(model["bias"][0])
    mean = model["mean"]
    std = model["std"]
    metrics = {}
    for split in ("train", "test"):
        x = arrays[f"{split}_x"]
        y = arrays[f"{split}_y"]
        group = arrays[f"{split}_group"]
        weight = arrays.get(f"{split}_w")
        if x.shape[0] == 0:
            continue
        if weight is None or weight.shape[0] != x.shape[0]:
            weight = np.ones(x.shape[0], dtype=np.float32)
        pred = sigmoid(((x - mean) / std) @ weights + bias)
        metrics[split] = {
            "samples": int(x.shape[0]),
            "positive_rate": float(np.sum(y * weight) / max(1e-6, float(weight.sum()))),
            "loss": bce_loss(pred, y, weight),
            "accuracy": accuracy(pred, y, weight),
            "with_mine": {},
            "without_mine": {},
        }
        for group_value, group_name in ((1, "with_mine"), (0, "without_mine")):
            mask = group == group_value
            if not mask.any():
                continue
            metrics[split][group_name] = {
                "samples": int(mask.sum()),
                "positive_rate": float(np.sum(y[mask] * weight[mask]) / max(1e-6, float(weight[mask].sum()))),
                "loss": bce_loss(pred[mask], y[mask], weight[mask]),
                "accuracy": accuracy(pred[mask], y[mask], weight[mask]),
            }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_run_report(run_dir, args, collection_stats, arrays, metrics, training_info=None):
    training_info = training_info or {}
    lines = []
    lines.append("# Tactical Training Run")
    lines.append("")
    lines.append(f"Run directory: `{run_dir.name}`")
    lines.append(f"Seed: `{args.seed}`")
    lines.append(f"Games: without_mine={args.without_mine_games}, with_mine={args.with_mine_games}")
    lines.append(f"Train ratio: {args.train_ratio}")
    lines.append(f"Sample stride: every {args.sample_stride} environment steps")
    lines.append(f"Training mode: {args.training_mode}")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Train samples: {arrays['train_x'].shape[0]}")
    lines.append(f"- Test samples: {arrays['test_x'].shape[0]}")
    lines.append(f"- Feature dimension: {arrays['train_x'].shape[1] if arrays['train_x'].size else 0}")
    lines.append(f"- Category counts: {collection_stats['categories']}")
    lines.append(f"- Lineups: {collection_stats['lineups']}")
    lines.append("")
    lines.append("## Progression")
    lines.append(
        f"- Warm start requested: {bool(training_info.get('warm_start_requested', False))}"
    )
    lines.append(
        f"- Warm start used: {bool(training_info.get('warm_start_used', False))}"
    )
    if training_info.get("warm_start_model"):
        lines.append(f"- Warm start model: `{training_info['warm_start_model']}`")
    if training_info.get("warm_start_reason"):
        lines.append(f"- Warm start note: {training_info['warm_start_reason']}")
    history_runs = training_info.get("history_runs", [])
    if history_runs:
        lines.append(f"- History merged from: {history_runs}")
        lines.append(f"- History train samples added: {training_info.get('history_train_samples', 0)}")
        lines.append(f"- History test samples added: {training_info.get('history_test_samples', 0)}")
        if training_info.get("history_skips"):
            lines.append(f"- History skips: {training_info['history_skips']}")
    else:
        lines.append("- History merged from: []")
    lines.append("")
    lines.append("## Model")
    lines.append("- Model type: numpy logistic action-value predictor")
    if args.training_mode == "ranking":
        lines.append("- Target: pairwise target preference using chosen-vs-alternative candidate differences.")
        lines.append("- Inputs: action feature differences between the chosen target and nearby viable alternatives.")
        lines.append("- The saved linear weights are used as a raw utility scorer during live target reranking.")
    else:
        lines.append("- Target: shaped action quality using final winner, sun/flight penalties, and production opportunity cost")
        lines.append("- Inputs: board quadrant state, role label, phase label, target context, recent tactical tendencies, and action penalty features")
        lines.append("- This is a role-conditioned action scorer layered onto the heuristic controller.")
    lines.append("")
    lines.append("## Metrics")
    for split, split_metrics in metrics.items():
        lines.append(f"### {split}")
        lines.append(
            f"- samples={split_metrics['samples']} positive_rate={split_metrics['positive_rate']:.4f} "
            f"loss={split_metrics['loss']:.4f} accuracy={split_metrics['accuracy']:.4f}"
        )
        for group_name in ("with_mine", "without_mine"):
            group = split_metrics.get(group_name, {})
            if not group:
                continue
            lines.append(
                f"- {group_name}: samples={group['samples']} positive_rate={group['positive_rate']:.4f} "
                f"loss={group['loss']:.4f} accuracy={group['accuracy']:.4f}"
            )
    lines.append("")
    lines.append("## Artifacts")
    lines.append("- `dataset.npz`: train/test arrays")
    lines.append("- `model_weights.npz`: learned weights and normalization")
    lines.append("- `training_progress.csv`: epoch log")
    lines.append("- `game_log.csv`: per-game lineup/winner log")
    lines.append("- `samples_meta.csv`: per-sample metadata")
    lines.append("- `metrics.json`: final metrics")
    (run_dir / "RUN_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Orbit Wars training games and train a tactical outcome model.")
    parser.add_argument("--without-mine-games", type=int, default=500)
    parser.add_argument("--with-mine-games", type=int, default=500)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--training-mode", choices=("ranking", "outcome"), default="ranking")
    parser.add_argument("--ranking-top-k", type=int, default=RANKING_DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=20261100)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.04)
    parser.add_argument("--l2", type=float, default=0.0005)
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--out", type=Path, default=TRAIN_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--from-run", type=Path, default=None, help="Train from an existing run directory's sample_values.csv.")
    parser.add_argument("--append-from-run", type=Path, action="append", default=[], help="Append dataset history from another run directory.")
    parser.add_argument("--init-from-model", type=Path, default=None, help="Warm-start from an existing model_weights.npz file.")
    parser.add_argument("--no-progressive-history", action="store_true", help="Do not append the currently promoted run dataset.")
    parser.add_argument("--no-warm-start", action="store_true", help="Do not initialize from the currently promoted model.")
    return parser.parse_args()


def main():
    args = parse_args()
    training_info = {}
    if args.from_run is not None:
        run_dir = _resolve_local_path(args.from_run)
        print(f"Training from existing run: {run_dir}", flush=True)
        arrays = load_arrays_from_run_dir(run_dir)
        collection_stats = {"lineups": {}, "categories": {"loaded_from_sample_values": 1}}
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = args.out / f"{timestamp}-tactical-outcome"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Training run: {run_dir}", flush=True)
        arrays, collection_stats = collect_games(args, run_dir)

    history_run_dirs = []
    promoted_run_dir = None if args.no_progressive_history else find_promoted_run_dir()
    if promoted_run_dir is not None and promoted_run_dir.resolve() != run_dir.resolve():
        history_run_dirs.append(promoted_run_dir)
    for history_path in args.append_from_run:
        resolved = _resolve_local_path(history_path)
        if resolved is None or not resolved.exists():
            continue
        history_run_dirs.append(resolved)
    deduped_history = []
    seen_history = set()
    for history_dir in history_run_dirs:
        key = str(history_dir.resolve())
        if key in seen_history:
            continue
        seen_history.add(key)
        deduped_history.append(history_dir)
    history_run_dirs = deduped_history

    history_train_samples = 0
    history_test_samples = 0
    history_skips = []
    for history_dir in history_run_dirs:
        try:
            history_arrays = load_arrays_from_run_dir(history_dir)
            merged_arrays = concat_dataset_arrays(arrays, history_arrays)
        except (FileNotFoundError, ValueError) as exc:
            history_skips.append(f"{history_dir.name}: {exc}")
            continue
        history_train_samples += int(history_arrays.get("train_x", np.asarray([])).shape[0])
        history_test_samples += int(history_arrays.get("test_x", np.asarray([])).shape[0])
        arrays = merged_arrays

    if history_run_dirs:
        np.savez_compressed(run_dir / "dataset.npz", **arrays)
        collection_stats["categories"]["history_runs_merged"] = len(history_run_dirs)
        collection_stats["categories"]["history_train_samples"] = history_train_samples
        collection_stats["categories"]["history_test_samples"] = history_test_samples
        if history_skips:
            collection_stats["categories"]["history_skips"] = len(history_skips)

    init_model_path = None
    if not args.no_warm_start:
        if args.init_from_model is not None:
            init_model_path = _resolve_local_path(args.init_from_model)
        elif promoted_run_dir is not None:
            init_model_path = promoted_run_dir / "model_weights.npz"
        else:
            root_model = ROOT / "model_weights.npz"
            if root_model.exists():
                init_model_path = root_model

    progress_path, training_info = train_logistic(arrays, args, run_dir, init_model_path=init_model_path)
    training_info["history_runs"] = [history_dir.name for history_dir in history_run_dirs]
    training_info["history_train_samples"] = history_train_samples
    training_info["history_test_samples"] = history_test_samples
    training_info["history_skips"] = history_skips
    metrics = evaluate_groups(arrays, run_dir)
    write_run_report(run_dir, args, collection_stats, arrays, metrics, training_info=training_info)
    print(f"Progress: {progress_path}", flush=True)
    print(f"Metrics: {run_dir / 'metrics.json'}", flush=True)
    print(f"Run log: {run_dir / 'RUN_LOG.md'}", flush=True)


if __name__ == "__main__":
    main()
