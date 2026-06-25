"""Food volume and nutrition estimation.

Combines object dimension measurement, fruit classification,
ellipsoid volume calculation, and nutritional value lookup.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from importlib.resources import files
from pathlib import Path

from visionkit.classify import FruitClassifier
from visionkit.logging_utils import configure_logging, get_logger
from visionkit.measure import MeasurementConfig, load_image, measure_objects
from visionkit.paths import DEFAULT_MODEL_PATH

log = get_logger("visionkit.nutrition")

_TF_HINT = "TensorFlow is required for this command. Install it with: pip install '.[ml]'"


def _tensorflow_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


# =========================================================================== #
# Volume Calculation
# =========================================================================== #

def ellipsoid_volume_cm3(width_mm: float, height_mm: float, shape_factor: float = 1.0) -> float:
    """Estimate volume (cm^3) of an item from its width and height in millimetres.

    We model each item as an ellipsoid revolved about its vertical axis.
    V = shape_factor * (pi/6) * w^2 * h
    """
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("width_mm and height_mm must be positive")
    if shape_factor <= 0:
        raise ValueError("shape_factor must be positive")

    volume_mm3 = shape_factor * (math.pi / 6.0) * (width_mm**2) * height_mm
    return volume_mm3 / 1000.0  # 1 cm^3 = 1000 mm^3


# =========================================================================== #
# Nutrition Database Loader
# =========================================================================== #

@dataclass(frozen=True)
class FoodInfo:
    """Density, geometry and nutrition for one food class."""

    name: str
    density_g_per_cm3: float
    shape_factor: float
    edible_fraction: float
    per_100g: dict[str, float] = field(default_factory=dict)

    def nutrients_for_mass(self, edible_mass_g: float) -> dict[str, float]:
        """Scale the per-100g nutrients to a given edible mass (grams)."""
        factor = edible_mass_g / 100.0
        return {k: round(v * factor, 3) for k, v in self.per_100g.items()}


class NutritionDatabase:
    """Lookup of :class:`FoodInfo` by class name."""

    def __init__(self, foods: dict[str, FoodInfo], meta: dict | None = None):
        self._foods = foods
        self.meta = meta or {}

    def __contains__(self, name: str) -> bool:
        return name in self._foods

    @property
    def names(self) -> list[str]:
        return sorted(self._foods)

    def get(self, name: str) -> FoodInfo:
        """Return the :class:`FoodInfo` for ``name``.

        Raises:
            KeyError: if the food is not in the database.
        """
        try:
            return self._foods[name]
        except KeyError as exc:
            raise KeyError(
                f"No nutrition data for {name!r}. Known foods: {', '.join(self.names)}"
            ) from exc

    @classmethod
    def from_dict(cls, raw: dict) -> NutritionDatabase:
        meta = raw.get("_meta", {})
        foods = {
            name: FoodInfo(
                name=name,
                density_g_per_cm3=float(entry["density_g_per_cm3"]),
                shape_factor=float(entry["shape_factor"]),
                edible_fraction=float(entry["edible_fraction"]),
                per_100g={k: float(v) for k, v in entry["per_100g"].items()},
            )
            for name, entry in raw.items()
            if not name.startswith("_")
        }
        return cls(foods, meta)

    @classmethod
    def load(cls, path: str | Path | None = None) -> NutritionDatabase:
        """Load the database from ``path`` or the packaged default table."""
        if path is None:
            text = (
                files("visionkit")
                .joinpath("data", "nutrition.json")
                .read_text(encoding="utf-8")
            )
        else:
            text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))


# =========================================================================== #
# Orchestrator
# =========================================================================== #

@dataclass
class FoodEstimate:
    """Full estimate for a single food item."""

    fruit: str
    confidence: float
    width_mm: float
    height_mm: float
    volume_cm3: float
    density_g_per_cm3: float
    mass_g: float
    edible_fraction: float
    edible_mass_g: float
    nutrients: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def estimate_food(
    image_path: str,
    ref_width_mm: float,
    *,
    model_path: str = str(DEFAULT_MODEL_PATH),
    labels_path: str | None = None,
    scale: float = 1.0,
    min_area: float = 100.0,
    database: NutritionDatabase | None = None,
    classifier: FruitClassifier | None = None,
) -> FoodEstimate:
    """Estimate volume and nutrition for the single food item in ``image_path``."""
    database = database or NutritionDatabase.load()

    # 1) Measure objects.
    image = load_image(image_path)
    config = MeasurementConfig(ref_width_mm=ref_width_mm, scale=scale, min_area=min_area)
    result = measure_objects(image, config)

    food_objects = [obj for obj in result.objects if not obj.is_reference]
    if not food_objects:
        raise ValueError(
            "No food object detected. The image must contain a reference object "
            "(left-most) and at least one food item. Try lowering --min-area or "
            "adjusting --scale/--canny."
        )
    # Largest non-reference object is the food item (single-item assumption).
    food = max(food_objects, key=lambda o: o.width_mm * o.height_mm)

    # 2) Identify the fruit (whole image).
    classifier = classifier or FruitClassifier(model_path, labels_path)
    prediction = classifier.predict(image_path)[0]
    info = database.get(prediction.label)

    # 3) Volume -> mass -> nutrients.
    volume_cm3 = ellipsoid_volume_cm3(food.width_mm, food.height_mm, info.shape_factor)
    mass_g = volume_cm3 * info.density_g_per_cm3
    edible_mass_g = mass_g * info.edible_fraction
    nutrients = info.nutrients_for_mass(edible_mass_g)

    log.info(
        "%s (%.1f%%): %.0fx%.0f mm -> %.0f cm3 -> %.0f g (%.0f g edible)",
        prediction.label,
        prediction.confidence * 100,
        food.width_mm,
        food.height_mm,
        volume_cm3,
        mass_g,
        edible_mass_g,
    )

    return FoodEstimate(
        fruit=prediction.label,
        confidence=prediction.confidence,
        width_mm=round(food.width_mm, 1),
        height_mm=round(food.height_mm, 1),
        volume_cm3=round(volume_cm3, 1),
        density_g_per_cm3=info.density_g_per_cm3,
        mass_g=round(mass_g, 1),
        edible_fraction=info.edible_fraction,
        edible_mass_g=round(edible_mass_g, 1),
        nutrients=nutrients,
    )


# =========================================================================== #
# CLI Logic
# =========================================================================== #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="food-estimate",
        description="Estimate a food item's volume and nutrition from a photo. "
        "The image needs a reference object (left-most, known width) and one food item.",
    )
    p.add_argument("image", help="Path to the input image.")
    p.add_argument(
        "--ref-width",
        type=float,
        default=24.0,
        help="Width of the reference object in millimetres (default: 24).",
    )
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Classifier model path.")
    p.add_argument(
        "--labels", default=None, help="Labels file (default: auto-detect '<model>.labels.json')."
    )
    p.add_argument(
        "--scale", type=float, default=1.0, help="Resize factor before measurement (default: 1.0)."
    )
    p.add_argument(
        "--min-area",
        type=float,
        default=100.0,
        help="Minimum contour area in pixels (default: 100).",
    )
    p.add_argument("--json", dest="json_out", help="Write the full estimate to this JSON file.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def _nice_nutrient(key: str) -> tuple[str, str]:
    """('vitamin_c_mg') -> ('Vitamin C', 'mg')."""
    name, _, unit = key.rpartition("_")
    return name.replace("_", " ").title(), unit


def format_report(est: FoodEstimate) -> str:
    lines = [
        f"Food:           {est.fruit}  ({est.confidence * 100:.1f}% confidence)",
        f"Dimensions:     {est.width_mm:.0f} x {est.height_mm:.0f} mm",
        f"Est. volume:    {est.volume_cm3:.0f} cm3  ({est.volume_cm3:.0f} mL)",
        f"Est. mass:      {est.mass_g:.0f} g  (whole)",
        f"Edible mass:    {est.edible_mass_g:.0f} g  ({est.edible_fraction * 100:.0f}% edible)",
        "Nutrition (for the edible portion):",
    ]
    for key, value in est.nutrients.items():
        name, unit = _nice_nutrient(key)
        lines.append(f"    {name:<14} {value:>8.1f} {unit}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    if not _tensorflow_available():
        log.error(_TF_HINT)
        return 1

    try:
        est = estimate_food(
            args.image,
            ref_width_mm=args.ref_width,
            model_path=args.model,
            labels_path=args.labels,
            scale=args.scale,
            min_area=args.min_area,
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    except KeyError as exc:
        log.error("%s", exc.args[0] if exc.args else exc)
        return 3

    print(format_report(est))

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(est.to_dict(), fh, indent=2)
        log.info("Wrote estimate to %s", args.json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
