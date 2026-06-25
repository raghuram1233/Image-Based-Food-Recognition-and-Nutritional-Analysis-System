"""Default project paths.

These resolve relative to the repository root so the tools work the same way
whether they are launched from the repo or installed as a package. Every path
is overridable from the CLI, so nothing here is load-bearing at runtime.
"""

from __future__ import annotations

from pathlib import Path

# src/visionkit/paths.py -> repo root is three levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "Data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

MODELS_DIR = REPO_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "Fruits_model.h5"

IMGS_DIR = REPO_ROOT / "imgs"


def labels_sidecar(model_path: str | Path) -> Path:
    """Return the conventional labels file that sits next to a model.

    e.g. ``models/fruit_mobilenetv2.keras`` -> ``models/fruit_mobilenetv2.labels.json``.
    Keeping labels coupled to each model means a model never clobbers another
    model's class metadata.
    """
    p = Path(model_path)
    return p.with_name(p.stem + ".labels.json")
