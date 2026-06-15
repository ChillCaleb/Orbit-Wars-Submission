import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "kaggle_training"
SOURCE_FILES = (
    "main.py",
    "tactical_features.py",
    "train_tactical_model.py",
    "agent_lab.py",
    "watch_match.py",
)


def newest_model():
    root_model = ROOT / "model_weights.npz"
    if root_model.exists():
        return root_model
    candidates = sorted((ROOT / "data" / "training_runs").glob("*/model_weights.npz"), reverse=True)
    return candidates[0] if candidates else None


def build_bundle():
    payload = io.BytesIO()
    manifest = []
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        paths = [ROOT / name for name in SOURCE_FILES]
        paths.extend(path for path in (ROOT / "agents").rglob("*") if path.is_file() and "__pycache__" not in path.parts)
        for path in paths:
            relative = path.relative_to(ROOT)
            archive.write(path, relative.as_posix())
            manifest.append(relative.as_posix())

        model_path = newest_model()
        if model_path is not None:
            archive.write(model_path, "model_weights.npz")
            manifest.append(f"model_weights.npz <- {model_path.relative_to(ROOT)}")

    return base64.b64encode(payload.getvalue()).decode("ascii"), manifest


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def markdown_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def build_notebook(bundle_b64, with_games, without_games, workers):
    unpack = f"""import base64
import io
from pathlib import Path
import zipfile

BUNDLE_B64 = {bundle_b64!r}
WORKDIR = Path("/kaggle/working/orbit_training")
WORKDIR.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as archive:
    archive.extractall(WORKDIR)
print("Prepared", WORKDIR)
"""
    train = f"""import subprocess
import sys

command = [
    sys.executable,
    "train_tactical_model.py",
    "--with-mine-games", "{with_games}",
    "--without-mine-games", "{without_games}",
    "--workers", "{workers}",
    "--out", "/kaggle/working/training_output",
]
model = WORKDIR / "model_weights.npz"
if model.exists():
    command.extend(["--init-from-model", str(model)])
print("Running:", " ".join(command))
subprocess.run(command, cwd=WORKDIR, check=True)
"""
    summarize = """import json
from pathlib import Path
import shutil

runs = sorted(Path("/kaggle/working/training_output").glob("*"))
latest = runs[-1]
print(json.dumps(json.loads((latest / "metrics.json").read_text()), indent=2))
archive = shutil.make_archive("/kaggle/working/orbit_training_output", "zip", latest)
print("Output archive:", archive)
"""
    return {
        "cells": [
            markdown_cell("# Orbit Wars Tactical Training\n\nSelf-contained 100-with/100-without training run."),
            code_cell(unpack),
            code_cell(train),
            code_cell(summarize),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description="Build the self-contained Kaggle training notebook without uploading it.")
    parser.add_argument("--with-mine-games", type=int, default=100)
    parser.add_argument("--without-mine-games", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    bundle_b64, manifest = build_bundle()
    notebook = build_notebook(bundle_b64, args.with_mine_games, args.without_mine_games, args.workers)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notebook_path = OUTPUT_DIR / "training.ipynb"
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")

    digest = hashlib.sha256(notebook_path.read_bytes()).hexdigest()
    manifest_data = {
        "notebook": str(notebook_path.relative_to(ROOT)),
        "sha256": digest,
        "with_mine_games": args.with_mine_games,
        "without_mine_games": args.without_mine_games,
        "workers": args.workers,
        "files": manifest,
    }
    (OUTPUT_DIR / "bundle_manifest.json").write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {notebook_path}")
    print(f"SHA256 {digest}")
    print("No Kaggle upload was performed.")


if __name__ == "__main__":
    main()
