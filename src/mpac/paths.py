"""Where artifacts live. Uses Google Drive on Colab so caches survive runtime resets."""

from __future__ import annotations

import os
from pathlib import Path

DRIVE_MOUNT = Path("/content/drive/MyDrive")


def in_colab() -> bool:
    return "COLAB_RELEASE_TAG" in os.environ


def resolve_root(cfg: dict) -> Path:
    """Artifact root: $MPAC_ROOT, else Drive if mounted, else project.root from config."""
    if env := os.environ.get("MPAC_ROOT"):
        root = Path(env)
    elif DRIVE_MOUNT.exists():
        root = DRIVE_MOUNT / cfg["project"].get("drive_subdir", "mpac")
    else:
        root = Path(cfg["project"]["root"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def subdir(cfg: dict, name: str) -> Path:
    """e.g. subdir(cfg, "generations") -> <root>/generations, created if missing."""
    path = resolve_root(cfg) / name
    path.mkdir(parents=True, exist_ok=True)
    return path
