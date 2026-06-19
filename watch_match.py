import argparse
import importlib.util
import math
from pathlib import Path
import re
import sys
import webbrowser

import imageio.v2 as imageio
from kaggle_environments import make
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "data" / "replays"

BUILTIN_AGENTS = {"random", "starter"}
LOCAL_AGENT_ALIASES = {
    "mine": "main.py",
    "main": "main.py",
    "smith": "agents/agent_smith.py",
    "agent_smith": "agents/agent_smith.py",
    "1039": "agents/agent_1039_launch_safety.py",
    "safety": "agents/agent_1039_launch_safety.py",
    "1200": "agents/agent_1200_ppo_strategy.py",
    "ppo": "agents/agent_1200_ppo_strategy.py",
    "best": "agents/best_orbit/agent_best_orbit.py",
    "best_orbit": "agents/best_orbit/agent_best_orbit.py",
    "intruder": "agents/light_intruder/agent_light_intruder.py",
    "light_intruder": "agents/light_intruder/agent_light_intruder.py",
}


def _discover_imported_agent_aliases():
    imported_root = ROOT / "agents" / "imported"
    aliases = {}
    if not imported_root.exists():
        return aliases
    for agent_path in sorted(imported_root.glob("*/agent.py")):
        aliases[agent_path.parent.name] = agent_path.relative_to(ROOT).as_posix()
    return aliases


LOCAL_AGENT_ALIASES.update(_discover_imported_agent_aliases())

COLORS = {
    -1: "#888888",
    0: "#0072B2",
    1: "#D55E00",
    2: "#009E73",
    3: "#F0E442",
}


def load_agent_file(path, slot):
    module_name = f"_orbit_agent_{slot}_{re.sub(r'[^a-zA-Z0-9_]', '_', path.stem)}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent file: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    loaded_agent = getattr(module, "agent", None)
    if loaded_agent is None:
        raise AttributeError(f"{path} does not define an agent(obs, config=None) function")
    return loaded_agent


def resolve_local_agent_path(agent_spec):
    alias = LOCAL_AGENT_ALIASES.get(agent_spec.lower())
    raw_path = Path(alias if alias is not None else agent_spec).expanduser()
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    return raw_path


def load_agent(agent_spec, slot):
    normalized = agent_spec.lower()
    if normalized in BUILTIN_AGENTS:
        return normalized, normalized

    if normalized in LOCAL_AGENT_ALIASES:
        path = resolve_local_agent_path(agent_spec)
        if not path.exists():
            raise FileNotFoundError(f"Agent file not found: {path}")
        return load_agent_file(path, slot), normalized

    path = resolve_local_agent_path(agent_spec)
    if path.exists() or path.suffix == ".py":
        if not path.exists():
            raise FileNotFoundError(f"Agent file not found: {path}")
        return load_agent_file(path, slot), path.stem

    return agent_spec, agent_spec


def build_lineup(agent_specs):
    if len(agent_specs) not in (2, 4):
        raise ValueError("--players must contain exactly 2 or 4 agents")
    agents = []
    labels = []
    for slot, agent_spec in enumerate(agent_specs):
        loaded_agent, label = load_agent(agent_spec, slot)
        agents.append(loaded_agent)
        labels.append(label)
    return agents, labels


def run_one(agents):
    env = make("orbit_wars", debug=True)
    env.run(agents)
    return env


def summarize(final_step, labels):
    return [
        (idx, labels[idx] if idx < len(labels) else f"player {idx}", state.reward, state.status)
        for idx, state in enumerate(final_step)
    ]


def winners(final_step):
    rewards = [state.reward for state in final_step]
    best_reward = max(rewards)
    return [idx for idx, reward in enumerate(rewards) if reward == best_reward]


def font(size):
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def obs_get(obs, name, default=None):
    if isinstance(obs, dict):
        return obs.get(name, default)
    return getattr(obs, name, default)


def draw_centered(draw, text, x, y, fill, size):
    label_font = font(size)
    bbox = draw.textbbox((0, 0), str(text), font=label_font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text((x - width / 2, y - height / 2), str(text), fill=fill, font=label_font)


def draw_circle(draw, x, y, r, fill=None, outline=None, width=2):
    box = (x - r, y - r, x + r, y + r)
    draw.ellipse(box, fill=fill, outline=outline, width=width)


def rotated(points, angle, origin):
    ox, oy = origin
    ca, sa = math.cos(angle), math.sin(angle)
    return [(ox + px * ca - py * sa, oy + px * sa + py * ca) for px, py in points]


def render_frame(obs, step_idx, size):
    scale = size / 100.0
    image = Image.new("RGB", (size, size), "#000000")
    draw = ImageDraw.Draw(image)

    sun_x = sun_y = 50 * scale
    for radius, fill in (
        (25, "#1d1300"),
        (19, "#2d1d00"),
        (14, "#5a3500"),
        (10, "#FFB800"),
    ):
        draw_circle(draw, sun_x, sun_y, radius * scale / 10, fill=fill)
    draw_circle(draw, sun_x, sun_y, 10 * scale, fill="#FFB800", outline="#FFD700", width=max(2, size // 260))

    comet_ids = set(obs_get(obs, "comet_planet_ids", []) or [])
    comets = obs_get(obs, "comets", []) or []
    for group in comets:
        path_index = group.get("path_index", 0) if isinstance(group, dict) else group.path_index
        paths = group.get("paths", []) if isinstance(group, dict) else group.paths
        for path in paths:
            tail_len = min(path_index + 1, len(path), 3)
            for tail_idx in range(1, tail_len):
                pos_idx = path_index - tail_idx
                if pos_idx < 0 or pos_idx + 1 >= len(path):
                    continue
                alpha = 170 - tail_idx * 45
                color = (200, 220, 255)
                x0, y0 = path[pos_idx]
                x1, y1 = path[pos_idx + 1]
                draw.line(
                    (x1 * scale, y1 * scale, x0 * scale, y0 * scale),
                    fill=color,
                    width=max(1, int((4 - tail_idx) * scale / 2)),
                )

    for planet in obs_get(obs, "planets", []) or []:
        pid, owner, x, y, radius, ships, _production = planet
        color = COLORS.get(owner, "#FFFFFF")
        px, py, pr = x * scale, y * scale, max(3, radius * scale)
        draw_circle(draw, px, py, pr, fill=color, outline="#FFFFFF" if pid in comet_ids else color, width=2)
        draw_centered(draw, int(ships), px, py, "#FFFFFF", max(9, int(12 * scale / 4)))

    for fleet in obs_get(obs, "fleets", []) or []:
        _fid, owner, x, y, angle, _from_planet_id, ships = fleet
        color = COLORS.get(owner, "#FFFFFF")
        px, py = x * scale, y * scale
        fleet_size = max(ships, 1)
        marker = (0.5 + (2.5 * math.log(fleet_size)) / math.log(1000)) * scale
        points = rotated(
            [(marker, 0), (-marker, -marker * 0.7), (-marker * 0.3, 0), (-marker, marker * 0.7)],
            angle,
            (px, py),
        )
        draw.polygon(points, fill=color)
        label_y = py + (-3 if y >= 50 else 3) * scale
        draw_centered(draw, int(ships), px, label_y, color, max(8, int(8 * scale / 4)))

    draw.text((10, 10), f"Step: {step_idx}", fill="#FFFFFF", font=font(max(14, size // 42)))
    return image


def next_replay_number(video_dir):
    highest = 0
    pattern = re.compile(r"^replay (\d+)\.mp4$")
    for path in video_dir.glob("replay *.mp4"):
        match = pattern.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest + 1


def video_path(args, first_number):
    replay_number = first_number + len(args._saved_paths)
    return VIDEO_DIR / f"replay {replay_number}.mp4"


def save_video(env, path, size, fps, frame_step):
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = env.steps[:: max(1, frame_step)]
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=7, macro_block_size=16) as writer:
        for frame_idx, step in enumerate(steps):
            obs = step[0].observation
            frame = render_frame(obs, frame_idx * max(1, frame_step), size)
            writer.append_data(np.asarray(frame))

    return path.resolve()


def main():
    parser = argparse.ArgumentParser(description="Run the Orbit Wars bot locally.")
    parser.add_argument("--games", type=int, default=1, help="Number of games to run.")
    parser.add_argument(
        "--players",
        nargs="+",
        help="Explicit 2- or 4-player lineup. Use mine, random, starter, aliases, or .py files.",
    )
    parser.add_argument(
        "--opponent",
        default=None,
        help="Backward-compatible shortcut for mine vs this opponent.",
    )
    parser.add_argument("--save-all", action="store_true", help="Save every replay when --games is above 1.")
    parser.add_argument("--no-save", action="store_true", help="Do not save an MP4 replay.")
    parser.add_argument("--open", action="store_true", help="Compatibility option; videos open by default.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the saved video in your browser.")
    parser.add_argument("--fps", type=int, default=12, help="Video frames per second.")
    parser.add_argument("--size", type=int, default=720, help="Square video size in pixels.")
    parser.add_argument("--frame-step", type=int, default=1, help="Save every Nth environment step as a frame.")
    args = parser.parse_args()

    if args.players and args.opponent is not None:
        parser.error("Use either --players or --opponent, not both.")

    agent_specs = args.players if args.players else ["mine", args.opponent or "random"]
    try:
        agents, labels = build_lineup(agent_specs)
    except (AttributeError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    win_counts = [0] * len(labels)
    tie_games = 0
    saved_paths = []
    args._saved_paths = saved_paths
    first_number = next_replay_number(VIDEO_DIR)
    lineup = ", ".join(f"P{idx}: {label}" for idx, label in enumerate(labels))
    print(f"Lineup: {lineup}")

    for game_idx in range(args.games):
        env = run_one(agents)
        game_winners = winners(env.steps[-1])
        if len(game_winners) == 1:
            win_counts[game_winners[0]] += 1
        else:
            tie_games += 1
        winner_label = ", ".join(f"P{idx} {labels[idx]}" for idx in game_winners)
        print(f"Game {game_idx + 1}: {summarize(env.steps[-1], labels)} Winner: {winner_label}")

        should_save_this = not args.no_save and (args.save_all or game_idx == args.games - 1)
        if should_save_this:
            saved_path = save_video(env, video_path(args, first_number), args.size, args.fps, args.frame_step)
            saved_paths.append(saved_path)
            print(f"Video saved to: {saved_path}")

    record = ", ".join(f"P{idx} {labels[idx]}: {wins}" for idx, wins in enumerate(win_counts))
    print(f"Wins: {record}; ties: {tie_games} over {args.games} game(s)")

    if saved_paths and not args.no_open:
        webbrowser.open(saved_paths[-1].as_uri())


if __name__ == "__main__":
    main()
