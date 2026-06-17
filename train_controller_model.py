#!/usr/bin/env python3
"""Train a lightweight controller selector over whole-agent proposals.

The tactical model scores individual actions. This trainer builds a second layer:
given the same board state, ask each available controller family for a complete
move proposal, featurize that proposal, and learn which source style matched the
eventual winner in elite-heavy sparring games.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import main as live_agent
from agent_lab import extract_observation, extract_reward, extract_status
from train_tactical_model import build_lineup, lineup_for, run_env_game


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "data" / "training_runs"
SOURCE_TO_LABEL = {
    "best": "best",
    "intruder": "intruder",
    "ppo1200": "1200",
    "smith": "smith",
    "hold": "hold",
}


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
    return float(np.sum(losses * weight) / max(1e-6, float(weight.sum())))


def accuracy(pred, y, sample_weight=None):
    if y.size == 0:
        return 0.0
    correct = ((pred >= 0.5) == (y >= 0.5)).astype(np.float32)
    if sample_weight is None:
        return float(np.mean(correct))
    weight = np.asarray(sample_weight, dtype=np.float32)
    return float(np.sum(correct * weight) / max(1e-6, float(weight.sum())))


def load_controller_model(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
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


def latest_controller_model():
    root_model = ROOT / "controller_weights.npz"
    if root_model.exists():
        return root_model
    candidates = sorted((ROOT / "data" / "training_runs").glob("*/controller_weights.npz"), reverse=True)
    return candidates[0] if candidates else None


def adapt_model_to_normalization(model, mean, std):
    weights = np.asarray(model["weights"], dtype=np.float32)
    old_mean = np.asarray(model["mean"], dtype=np.float32)
    old_std = np.asarray(model["std"], dtype=np.float32).copy()
    new_mean = np.asarray(mean, dtype=np.float32)
    new_std = np.asarray(std, dtype=np.float32).copy()
    if weights.ndim != 1 or old_mean.shape != weights.shape or old_std.shape != weights.shape:
        raise ValueError("Stored controller normalization shape mismatch")
    if new_mean.shape[0] < weights.shape[0]:
        raise ValueError("Current controller feature vector is shorter than stored model")
    old_std[old_std < 1e-6] = 1.0
    new_std[new_std < 1e-6] = 1.0
    old_dim = weights.shape[0]
    scaled_weights = np.zeros(new_mean.shape[0], dtype=np.float32)
    scaled_weights[:old_dim] = weights * (new_std[:old_dim] / old_std)
    scaled_bias = float(model["bias"]) + float(
        np.sum(((new_mean[:old_dim] - old_mean) / old_std) * weights)
    )
    return scaled_weights.astype(np.float32), np.float32(scaled_bias)


def reset_source(module):
    runtime = getattr(module, "_RUNTIME", None)
    if runtime is not None and hasattr(runtime, "reset"):
        try:
            runtime.reset()
        except Exception:
            pass


def _clip(value, lo, hi):
    return max(float(lo), min(float(hi), float(value)))


def _proposal_future_quality(state, proposal, player, planets, fleets, angular_velocity, step):
    """Score a proposal by expected near-future usefulness, not source identity."""
    moves = proposal.get("moves", []) or []
    if not moves:
        pressure = live_agent._proposal_pressure_lookup(player, planets, fleets)
        owned = [p for p in planets if int(p.owner) == int(player)]
        stockpile = max((int(p.ships) for p in owned), default=0)
        if pressure:
            return 0.18
        return 0.46 if stockpile >= 85 else 0.34

    by_id = {int(p.id): p for p in planets}
    pressure = live_agent._proposal_pressure_lookup(player, planets, fleets)
    quality = 0.0
    ships_total = 0
    enemy_count = 0
    enemy_ships = 0
    enemy_max = 0
    friendly_maintenance = 0
    pressure_covers = 0
    thin_enemy = 0
    decisive_enemy = 0
    positive_opportunity = 0

    for move in moves:
        if not move or len(move) < 3:
            quality -= 0.65
            continue
        source = by_id.get(int(move[0]))
        target = live_agent._proposal_target_for_move(move, planets)
        ships = int(move[2])
        ships_total += ships
        if source is None or target is None:
            quality -= 0.65
            continue

        opportunity_value, opportunity_reason = live_agent._proposal_opportunity_value(
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
        if opportunity_value > 0.0:
            positive_opportunity += 1

        if int(target.owner) == int(player):
            target_pressure = pressure.get(int(target.id))
            if target_pressure:
                needed = max(1, int(target_pressure["ships"]) - int(target.ships) + 1)
                coverage = min(2.0, float(ships) / float(max(1, needed)))
                quality += 0.75 + 0.75 * coverage
                pressure_covers += 1
            else:
                quality -= 0.55
                friendly_maintenance += 1
        elif int(target.owner) == live_agent.NEUTRAL:
            need = live_agent._planned_capture_need(source, target, angular_velocity)
            margin = (float(ships) - float(need)) / max(10.0, float(need))
            if ships >= need:
                quality += 0.18 + 0.04 * float(target.production) + 0.20 * min(1.5, margin)
            else:
                quality -= 0.28
        else:
            need = live_agent._offensive_capture_need(source, target, player, planets, fleets, angular_velocity)
            margin = (float(ships) - float(need)) / max(12.0, float(need))
            enemy_count += 1
            enemy_ships += ships
            enemy_max = max(enemy_max, ships)
            quality += 1.25 + 1.65 * _clip(margin, -1.0, 1.65)
            quality += 0.08 * float(target.production)
            quality += 0.22 if live_agent._is_static(target) else 0.0
            quality += 0.28 if live_agent._is_big(target) else 0.0
            quality += 0.45 * float(opportunity_value)
            if ships >= 50:
                quality += 0.32
            if ships >= 80:
                quality += 0.36
            if margin >= 0.0:
                quality += 0.58
                decisive_enemy += 1
            else:
                quality -= 0.85
                thin_enemy += 1

        if int(source.id) in pressure and not (target is not None and int(target.owner) == int(player)):
            quality -= 0.28
        if opportunity_reason in ("overtake_window", "cash_in", "chasing_leader"):
            quality += 0.42

    if enemy_count:
        concentration = float(enemy_ships) / float(max(1, ships_total))
        quality += 0.65 * _clip(concentration, 0.0, 1.0)
        quality += 0.45 * min(1.0, float(enemy_max) / 95.0)
        if enemy_count <= 3 and enemy_ships >= 70:
            quality += 0.55
        if decisive_enemy:
            quality += 0.25 * min(3, decisive_enemy)
        quality -= 0.20 * max(0, enemy_count - 5)
    elif pressure_covers:
        quality += 0.35
    else:
        quality -= 0.25

    quality -= 0.22 * friendly_maintenance
    quality -= 0.18 * max(0, len(moves) - 7)
    quality -= 0.12 * thin_enemy
    quality += 0.10 * min(3, positive_opportunity)
    quality -= 0.0025 * max(0, ships_total - 240)

    return float(1.0 / (1.0 + np.exp(-_clip((quality - 1.75) / 2.25, -8.0, 8.0))))


def proposal_rows_for_state(obs, player, labels, winners, category, split, state, step, target_mode):
    planets, fleets, angular_velocity = live_agent._parse(obs)[1:4]
    rows = []
    known_winners = {winner for winner in winners if winner in set(SOURCE_TO_LABEL.values())}
    if target_mode == "winner-source" and not known_winners:
        return rows

    live_agent._update_tactical_events(state, player, planets)
    live_agent._update_recent_static_capture_focus(state, player, planets)
    state["turn_claimed_target_ids"] = set()

    proposals = []
    for name, module, kind in live_agent._proposal_sources():
        reset_source(module)
        raw_moves = live_agent._call_proposal_source(
            name,
            module,
            kind,
            obs,
            None,
            state,
            player,
            planets,
            fleets,
            step,
        )
        scored = live_agent._score_proposal(
            state,
            name,
            raw_moves,
            player,
            planets,
            fleets,
            angular_velocity,
            step,
            apply_controller=False,
        )
        if scored is not None:
            proposals.append(scored)

    proposals.append(
        {
            "name": "hold",
            "moves": [],
            "score": live_agent._hold_proposal_score(player, planets, fleets),
            "reason": "stockpile",
        }
    )

    actual_label = labels[player] if player < len(labels) else f"player{player}"
    for proposal in proposals:
        source_label = SOURCE_TO_LABEL.get(proposal["name"], proposal["name"])
        quality_target = _proposal_future_quality(state, proposal, player, planets, fleets, angular_velocity, step)
        winner_target = 1.0 if source_label in known_winners else 0.0
        target = winner_target if target_mode == "winner-source" else quality_target
        features = live_agent._controller_proposal_features(
            state,
            proposal,
            player,
            planets,
            fleets,
            angular_velocity,
            step,
        )
        if not features:
            continue
        move_count = len(proposal.get("moves", []) or [])
        ships_total = sum(int(move[2]) for move in proposal.get("moves", []) or [] if move and len(move) >= 3)
        weight = 1.0 + 1.2 * abs(float(target) - 0.5)
        if winner_target > 0.5:
            weight += 1.0
        if actual_label == source_label:
            weight += 0.35
        if actual_label in known_winners:
            weight += 0.35
        rows.append(
            {
                "features": features,
                "target": target,
                "weight": weight,
                "meta": {
                    "split": split,
                    "category": category,
                    "lineup": " ".join(labels),
                    "player": actual_label,
                    "step": step,
                    "target_mode": target_mode,
                    "proposal_source": proposal["name"],
                    "proposal_winner_style": source_label,
                    "winners": " ".join(winners),
                    "target": round(float(target), 6),
                    "quality_target": round(float(quality_target), 6),
                    "winner_source_target": round(float(winner_target), 6),
                    "sample_weight": round(weight, 6),
                    "base_score": round(float(proposal.get("score", 0.0)), 6),
                    "move_count": move_count,
                    "ships_total": ships_total,
                    "reason": proposal.get("reason", ""),
                },
            }
        )
    return rows


def collect_one_game(task):
    agents, labels = build_lineup(list(task["lineup_specs"]))
    env, _calls_by_step = run_env_game(agents, labels, task["seed"])
    final_rewards = [extract_reward(state) for state in env.steps[-1]]
    best_reward = max(final_rewards)
    winners = [
        labels[idx]
        for idx, reward in enumerate(final_rewards)
        if reward == best_reward and final_rewards.count(best_reward) == 1
    ]
    statuses = [extract_status(state) for state in env.steps[-1]]
    state_by_player = defaultdict(live_agent._fresh_state)
    rows = []
    for step_index, step_states in enumerate(env.steps):
        if step_index % task["sample_stride"] != 0 and step_index != len(env.steps) - 1:
            continue
        for player, _label in enumerate(labels):
            obs = extract_observation(step_states[player])
            step = int(live_agent._obs_get(obs, "step", step_index) or step_index)
            state = state_by_player[player]
            state["last_turn"] = step
            rows.extend(
                proposal_rows_for_state(
                    obs,
                    player,
                    labels,
                    winners,
                    task["category"],
                    task["split"],
                    state,
                    step,
                    task["target_mode"],
                )
            )
    return {
        "task": task,
        "labels": labels,
        "rows": rows,
        "game_row": {
            "game_index": task["global_game_index"] + 1,
            "category": task["category"],
            "split": task["split"],
            "seed": task["seed"],
            "lineup": " ".join(labels),
            "rewards": json.dumps(final_rewards),
            "statuses": json.dumps(statuses),
            "winners": json.dumps(winners),
            "controller_samples": len(rows),
        },
    }


def build_tasks(args):
    tasks = []
    global_game_index = 0
    for category, total in (("without_mine", args.without_mine_games), ("with_mine", args.with_mine_games)):
        train_cutoff = int(round(total * args.train_ratio))
        for category_index in range(total):
            split = "train" if category_index < train_cutoff else "test"
            tasks.append(
                {
                    "global_game_index": global_game_index,
                    "category_index": category_index,
                    "category": category,
                    "split": split,
                    "lineup_specs": lineup_for(category, category_index, preset=args.lineup_preset),
                    "seed": args.seed + global_game_index,
                    "sample_stride": args.sample_stride,
                    "target_mode": args.target_mode,
                }
            )
            global_game_index += 1
    return tasks


def collect_games(args, run_dir):
    tasks = build_tasks(args)
    train_x, train_y, train_w = [], [], []
    test_x, test_y, test_w = [], [], []
    meta_rows = []
    game_rows = []
    stats_by_lineup = Counter()
    stats_by_category = Counter()

    for task in tasks:
        result = collect_one_game(task)
        game_rows.append(result["game_row"])
        stats_by_lineup[result["game_row"]["lineup"]] += 1
        stats_by_category[f"{task['category']}_{task['split']}"] += 1
        for row in result["rows"]:
            if task["split"] == "train":
                train_x.append(row["features"])
                train_y.append(row["target"])
                train_w.append(row["weight"])
            else:
                test_x.append(row["features"])
                test_y.append(row["target"])
                test_w.append(row["weight"])
            meta_rows.append(row["meta"])
        completed = len(game_rows)
        if completed == 1 or completed % max(1, args.progress_every) == 0 or completed == len(tasks):
            print(
                f"[{completed}/{len(tasks)}] latest={task['category']} {task['split']} "
                f"seed={task['seed']} samples={len(result['rows'])} lineup={result['game_row']['lineup']}",
                flush=True,
            )

    arrays = {
        "train_x": np.asarray(train_x, dtype=np.float32),
        "train_y": np.asarray(train_y, dtype=np.float32),
        "train_w": np.asarray(train_w, dtype=np.float32),
        "test_x": np.asarray(test_x, dtype=np.float32),
        "test_y": np.asarray(test_y, dtype=np.float32),
        "test_w": np.asarray(test_w, dtype=np.float32),
    }
    np.savez_compressed(run_dir / "controller_dataset.npz", **arrays)

    with (run_dir / "controller_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        if meta_rows:
            fieldnames = list(meta_rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(meta_rows)

    with (run_dir / "controller_game_log.csv").open("w", newline="", encoding="utf-8") as handle:
        if game_rows:
            fieldnames = list(game_rows[0].keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(game_rows)

    return arrays, {
        "lineups": dict(stats_by_lineup),
        "categories": dict(stats_by_category),
        "games": len(game_rows),
        "samples": len(meta_rows),
    }


def train_controller(arrays, args, run_dir, init_model_path=None):
    x = arrays["train_x"]
    y = arrays["train_y"]
    w = arrays["train_w"]
    tx = arrays["test_x"]
    ty = arrays["test_y"]
    tw = arrays["test_w"]
    if x.shape[0] == 0:
        raise ValueError("No controller samples were collected")

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
        "warm_start_model": str(init_model_path) if init_model_path is not None else None,
        "warm_start_reason": "random_init",
    }
    if init_model_path is not None:
        model = load_controller_model(init_model_path)
        if model is None:
            training_info["warm_start_reason"] = "init_controller_unavailable"
        elif int(model["weights"].shape[0]) > int(xz.shape[1]):
            training_info["warm_start_reason"] = (
                f"feature_dim_mismatch stored={model['weights'].shape[0]} current={xz.shape[1]}"
            )
        else:
            try:
                weights, bias = adapt_model_to_normalization(model, mean, std)
                training_info["warm_start_used"] = True
                training_info["warm_start_model"] = model["path"]
                training_info["warm_start_reason"] = "initialized_from_existing_controller"
            except ValueError as exc:
                training_info["warm_start_reason"] = str(exc)

    batch_size = min(args.batch_size, xz.shape[0])
    pos_rate = max(1e-3, float(y.mean()))
    neg_rate = max(1e-3, 1.0 - pos_rate)
    pos_weight = neg_rate / pos_rate
    progress_path = run_dir / "controller_training_progress.csv"
    with progress_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "test_loss", "train_acc", "test_acc", "lr"])
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
        run_dir / "controller_weights.npz",
        weights=weights.astype(np.float32),
        bias=np.asarray([bias], dtype=np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        source_names=np.asarray(live_agent.CONTROLLER_SOURCE_NAMES),
        trend_names=np.asarray(live_agent.CONTROLLER_TREND_NAMES),
    )
    training_info["progress_path"] = str(progress_path)
    return progress_path, training_info


def evaluate(arrays, run_dir):
    with np.load(run_dir / "controller_weights.npz") as model:
        weights = model["weights"]
        bias = float(model["bias"][0])
        mean = model["mean"]
        std = model["std"]
    metrics = {}
    for split in ("train", "test"):
        x = arrays[f"{split}_x"]
        y = arrays[f"{split}_y"]
        w = arrays[f"{split}_w"]
        if x.shape[0] == 0:
            continue
        pred = sigmoid(((x - mean) / std) @ weights + bias)
        metrics[split] = {
            "samples": int(x.shape[0]),
            "positive_rate": float(np.sum(y * w) / max(1e-6, float(w.sum()))),
            "loss": bce_loss(pred, y, w),
            "accuracy": accuracy(pred, y, w),
        }
    (run_dir / "controller_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def write_report(run_dir, args, collection_stats, arrays, metrics, training_info):
    lines = [
        "# Controller Selector Training Run",
        "",
        f"Run directory: `{run_dir.name}`",
        f"Seed: `{args.seed}`",
        f"Games: without_mine={args.without_mine_games}, with_mine={args.with_mine_games}",
        f"Lineup preset: `{args.lineup_preset}`",
        f"Sample stride: every {args.sample_stride} environment steps",
        f"Target mode: `{args.target_mode}`",
        "",
        "## Dataset",
        f"- Train samples: {arrays['train_x'].shape[0]}",
        f"- Test samples: {arrays['test_x'].shape[0]}",
        f"- Feature dimension: {arrays['train_x'].shape[1] if arrays['train_x'].size else 0}",
        f"- Category counts: {collection_stats['categories']}",
        f"- Lineups: {collection_stats['lineups']}",
        "",
        "## Progression",
        f"- Warm start requested: {bool(training_info.get('warm_start_requested', False))}",
        f"- Warm start used: {bool(training_info.get('warm_start_used', False))}",
    ]
    if training_info.get("warm_start_model"):
        lines.append(f"- Warm start model: `{training_info['warm_start_model']}`")
    lines.append(f"- Warm start note: {training_info.get('warm_start_reason', 'unknown')}")
    lines.extend(
        [
            "",
            "## Model",
            "- Model type: numpy logistic proposal-source selector",
            "- Target: "
            + (
                "whether a proposal source's style matched the final winning model in the game"
                if args.target_mode == "winner-source"
                else "near-future proposal quality: decisive enemy payloads, pressure coverage, and low-churn movement"
            ),
            "- Inputs: whole-proposal attack mass, pressure coverage, target mix, source style, and opportunity/future proxy features",
            "",
            "## Metrics",
        ]
    )
    for split, row in metrics.items():
        lines.append(
            f"- {split}: samples={row['samples']} positive_rate={row['positive_rate']:.4f} "
            f"loss={row['loss']:.4f} accuracy={row['accuracy']:.4f}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "- `controller_dataset.npz`: train/test feature arrays",
            "- `controller_weights.npz`: learned selector weights and normalization",
            "- `controller_samples.csv`: proposal-source metadata",
            "- `controller_game_log.csv`: per-game lineup/winner log",
            "- `controller_metrics.json`: final metrics",
        ]
    )
    (run_dir / "CONTROLLER_RUN_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a proposal-source controller selector.")
    parser.add_argument("--without-mine-games", type=int, default=50)
    parser.add_argument("--with-mine-games", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--lineup-preset", choices=("all", "new-agents", "opportunity"), default="opportunity")
    parser.add_argument("--target-mode", choices=("winner-source", "future-quality"), default="winner-source")
    parser.add_argument("--seed", type=int, default=20261200)
    parser.add_argument("--sample-stride", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--l2", type=float, default=0.0007)
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--out", type=Path, default=TRAIN_DIR)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--init-from-controller", type=Path, default=None)
    parser.add_argument("--no-warm-start", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out / f"{timestamp}-controller-selector"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "controller_config.json").write_text(json.dumps(vars(args), indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Controller run: {run_dir}", flush=True)
    arrays, collection_stats = collect_games(args, run_dir)
    init_model_path = None
    if not args.no_warm_start:
        init_model_path = args.init_from_controller or latest_controller_model()
    progress_path, training_info = train_controller(arrays, args, run_dir, init_model_path=init_model_path)
    metrics = evaluate(arrays, run_dir)
    write_report(run_dir, args, collection_stats, arrays, metrics, training_info)
    print(f"Progress: {progress_path}", flush=True)
    print(f"Metrics: {run_dir / 'controller_metrics.json'}", flush=True)
    print(f"Run log: {run_dir / 'CONTROLLER_RUN_LOG.md'}", flush=True)


if __name__ == "__main__":
    main()
