#!/usr/bin/env python3
"""Build Kaggle submission artifacts.

The primary artifact is `submission.zip`, which carries the agent source and
trained model weights together. A legacy single-file wrapper is also generated
from the same bundle for environments that still want a `.py` upload.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ZIP = ROOT / "submission.zip"
DEFAULT_OUTPUT_PY = ROOT / "submission.py"
SOURCE_FILES = (
    "main.py",
    "tactical_features.py",
)
SOURCE_DIRS = (
    "agents",
)
SOURCE_DIR_EXCLUDES = (
    ("agents", "imported"),
)


WRAPPER_TEMPLATE = '''\
import base64
import importlib
import sys
import tempfile
import zipfile
from pathlib import Path


_ZIP_B64 = """{payload}"""
_ENTRY = None


def _ensure_payload_loaded():
    global _ENTRY
    if _ENTRY is not None:
        return _ENTRY

    payload = base64.b64decode(_ZIP_B64.encode("ascii"))
    temp_dir = Path(tempfile.mkdtemp(prefix="orbit_wars_submission_"))
    archive = temp_dir / "bundle.zip"
    archive.write_bytes(payload)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(temp_dir)

    sys.path.insert(0, str(temp_dir))
    try:
        module = importlib.import_module("main")
    finally:
        if sys.path and sys.path[0] == str(temp_dir):
            sys.path.pop(0)
    _ENTRY = module.agent
    return _ENTRY


def agent(obs, config=None):
    return _ensure_payload_loaded()(obs, config)
'''


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def build_bundle(model_weights_path: Path, controller_weights_path: Path | None = None) -> tuple[bytes, list[tuple[str, int, str]]]:
    metadata: list[tuple[str, int, str]] = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        relative_names = list(SOURCE_FILES)
        for source_dir in SOURCE_DIRS:
            directory = ROOT / source_dir
            if not directory.exists():
                raise FileNotFoundError(directory)
            relative_names.extend(
                str(path.relative_to(ROOT))
                for path in sorted(directory.rglob("*.py"))
                if "__pycache__" not in path.parts
                and not any(tuple(path.relative_to(ROOT).parts[: len(exclude)]) == exclude for exclude in SOURCE_DIR_EXCLUDES)
            )

        for relative_name in dict.fromkeys(relative_names):
            path = ROOT / relative_name
            if not path.exists():
                raise FileNotFoundError(path)
            data = path.read_bytes()
            zf.writestr(relative_name, data)
            metadata.append((relative_name, len(data), _sha256(data)))

        model_weights_path = _resolve_path(model_weights_path)
        if not model_weights_path.exists():
            raise FileNotFoundError(model_weights_path)
        data = model_weights_path.read_bytes()
        zf.writestr("model_weights.npz", data)
        metadata.append(("model_weights.npz", len(data), _sha256(data)))
        if controller_weights_path is not None:
            controller_weights_path = _resolve_path(controller_weights_path)
            if controller_weights_path.exists():
                data = controller_weights_path.read_bytes()
                zf.writestr("controller_weights.npz", data)
                metadata.append(("controller_weights.npz", len(data), _sha256(data)))
    return buffer.getvalue(), metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Build Orbit Wars Kaggle submission artifacts.")
    parser.add_argument("--model-weights", type=Path, default=ROOT / "model_weights.npz")
    parser.add_argument("--controller-weights", type=Path, default=ROOT / "controller_weights.npz")
    parser.add_argument("--zip-output", type=Path, default=DEFAULT_OUTPUT_ZIP)
    parser.add_argument("--py-output", type=Path, default=DEFAULT_OUTPUT_PY)
    parser.add_argument("--no-py", action="store_true", help="Only write the zip artifact.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    zip_output = _resolve_path(args.zip_output)
    py_output = _resolve_path(args.py_output)
    bundle, metadata = build_bundle(args.model_weights, args.controller_weights)
    zip_output.parent.mkdir(parents=True, exist_ok=True)
    zip_output.write_bytes(bundle)
    print(f"Wrote {zip_output.relative_to(ROOT)}")
    if not args.no_py:
        py_output.parent.mkdir(parents=True, exist_ok=True)
        payload = base64.b64encode(bundle).decode("ascii")
        py_output.write_text(WRAPPER_TEMPLATE.format(payload=payload), encoding="utf-8")
        print(f"Wrote {py_output.relative_to(ROOT)}")
    for name, size, digest in metadata:
        print(f"{name}\t{size}\t{digest}")


if __name__ == "__main__":
    main()
