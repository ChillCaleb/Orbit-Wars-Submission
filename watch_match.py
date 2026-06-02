import argparse
import math
from pathlib import Path
import re
import webbrowser

import imageio.v2 as imageio
from kaggle_environments import make
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from main import agent


VIDEO_DIR = Path("MP4")

COLORS = {
    -1: "#888888",
    0: "#0072B2",
    1: "#D55E00",
    2: "#009E73",
    3: "#F0E442",
}


def run_one(opponent):
    env = make("orbit_wars", debug=True)
    env.run([agent, opponent])
    return env


def summarize(final_step):
    return [(idx, state.reward, state.status) for idx, state in enumerate(final_step)]


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
    parser.add_argument("--opponent", default="random", help="Opponent agent name or file.")
    parser.add_argument("--save-all", action="store_true", help="Save every replay when --games is above 1.")
    parser.add_argument("--no-save", action="store_true", help="Do not save an MP4 replay.")
    parser.add_argument("--open", action="store_true", help="Compatibility option; videos open by default.")
    parser.add_argument("--no-open", action="store_true", help="Do not open the saved video in your browser.")
    parser.add_argument("--fps", type=int, default=12, help="Video frames per second.")
    parser.add_argument("--size", type=int, default=720, help="Square video size in pixels.")
    parser.add_argument("--frame-step", type=int, default=1, help="Save every Nth environment step as a frame.")
    args = parser.parse_args()

    wins = losses = ties = 0
    saved_paths = []
    args._saved_paths = saved_paths
    first_number = next_replay_number(VIDEO_DIR)

    for game_idx in range(args.games):
        env = run_one(args.opponent)
        player_reward = env.steps[-1][0].reward
        if player_reward > 0:
            wins += 1
        elif player_reward < 0:
            losses += 1
        else:
            ties += 1
        print(f"Game {game_idx + 1}: {summarize(env.steps[-1])}")

        should_save_this = not args.no_save and (args.save_all or game_idx == args.games - 1)
        if should_save_this:
            saved_path = save_video(env, video_path(args, first_number), args.size, args.fps, args.frame_step)
            saved_paths.append(saved_path)
            print(f"Video saved to: {saved_path}")

    print(f"Record: {wins}-{losses}-{ties} over {args.games} game(s)")

    if saved_paths and not args.no_open:
        webbrowser.open(saved_paths[-1].as_uri())


if __name__ == "__main__":
    main()
