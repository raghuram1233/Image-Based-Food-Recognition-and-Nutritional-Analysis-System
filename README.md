# visionkit

A small, professional computer-vision toolkit with two independent tools:

1. **Object Dimension Measurement** — measure real-world object sizes from a
   single image using a reference object for scale calibration (pure OpenCV).
2. **Fruit Classification** — train and run a CNN image classifier for five
   fruit classes (Apple, Banana, Mango, Orange, Pineapple), with transfer
   learning from a pretrained MobileNetV2 backbone.

The project is packaged as an installable Python package (`src/visionkit`) with
console entry points, structured logging, configurable parameters, and clean,
testable, side-effect-free core logic.

## Project structure

```
visionkit/
├── pyproject.toml            # packaging, dependencies, console scripts, ruff config
├── requirements.txt          # convenience mirror of pyproject deps
├── src/visionkit/
│   ├── __init__.py           # package version & exports
│   ├── paths.py              # default project paths (all CLI-overridable)
│   ├── logging_utils.py      # centralised logging
│   ├── app.py                # Flask web UI
│   ├── measure.py            # object dimension measurement (logic + CLI)
│   ├── classify.py           # fruit classification (datasets, models, training, inference, CLI)
│   ├── nutrition.py          # food volume + nutrition estimation (volume, database, orchestrator, CLI)
│   └── data/
│       └── nutrition.json    # per-food density / shape / per-100g nutrients
├── models/
│   ├── Fruits_model.h5            # legacy baseline model (128x128 CNN)
│   └── Fruits_model.labels.json   # sidecar: class order + input size + preprocessing
├── Data/{train,test}/<class> # dataset
└── imgs/                     # sample images
```

## Installation

```bash
# Measurement tool only (light: numpy + opencv)
pip install -e .

# + Fruit classification & training (adds TensorFlow)
pip install -e ".[ml]"

# + development tools (ruff, pytest)
pip install -e ".[ml,dev]"
```

Python 3.10+ is required.

## Usage

### Object dimension measurement

The left-most detected object is used as the reference for scale calibration.

```bash
# Measure and print sizes (headless — no GUI needed)
measure-object imgs/test1.png --ref-width 25 --scale 0.2

# Save an annotated image
measure-object imgs/test1.png --ref-width 25 --scale 0.2 -o measured.png

# Show in a window (requires a desktop session)
measure-object imgs/test1.png --show
```

Key options: `--ref-width` (reference width in mm), `--scale` (resize factor —
measurements are scale-invariant), `--min-area`, `--canny LOW HIGH`, `--blur`.

Programmatic use:

```python
from visionkit.measure import MeasurementConfig, load_image, measure_objects

image = load_image("imgs/test1.png")
result = measure_objects(image, MeasurementConfig(ref_width_mm=25, scale=0.2))
for obj in result.objects:
    print(obj.width_mm, obj.height_mm, obj.is_reference)
```

### Fruit classification (inference)

Each model carries a sidecar `<model>.labels.json` (class order + input size +
preprocessing), which is auto-detected — so labels always match the model and
are never hard-coded.

```bash
# Default model is the committed baseline (models/Fruits_model.h5)
fruit-classify imgs/apple.jpg

# Use the trained transfer-learning model (auto-finds fruit_mobilenetv2.labels.json)
fruit-classify imgs/apple.jpg --model models/fruit_mobilenetv2.keras --top 3
```

Programmatic use:

```python
from visionkit.classify import FruitClassifier

clf = FruitClassifier("models/fruit_mobilenetv2.keras")  # labels auto-detected
pred = clf.predict("imgs/apple.jpg")[0]
print(pred.label, pred.confidence)      # e.g. "Apple" 0.999
```

### Training (transfer learning)

Run on a machine with TensorFlow installed (GPU strongly recommended):

```bash
# Transfer learning from MobileNetV2 (default), then optional fine-tuning
fruit-train --epochs 20 --fine-tune-epochs 10

# Reproduce the original small CNN for comparison
fruit-train --backbone cnn --img-size 128 --epochs 50
```

Outputs are written next to the model, named after it: e.g.
`fruit_mobilenetv2.keras`, `fruit_mobilenetv2.labels.json` (class order + input
size + preprocessing), `fruit_mobilenetv2.metrics.json` (per-epoch history +
test accuracy), and `fruit_mobilenetv2_curves.png`. Naming artefacts per model
means training one model never overwrites another's metadata.

### Food volume & nutrition

Estimate a food item's real-world volume and nutritional statistics from a
single photo. This chains the tools above: measure → identify → volume → mass →
nutrition.

**The photo must contain a reference object of known width placed left-most**
(e.g. a coin) plus the food item — the reference calibrates pixels to
millimetres (there is no way to recover real size from a 2D photo otherwise).

```bash
food-estimate plate.jpg --ref-width 24 --model models/fruit_mobilenetv2.keras
food-estimate plate.jpg --ref-width 24 --json result.json   # also dump JSON
```

Example output:

```
Food:           Apple  (98.8% confidence)
Dimensions:     74 x 49 mm
Est. volume:    120 cm3  (120 mL)
Est. mass:      101 g  (whole)
Edible mass:    90 g  (90% edible)
Nutrition (for the edible portion):
    Calories           47.1 kcal
    Carbohydrate       12.5 g
    ...
```

How it works:

- **Volume**: each item is modelled as an ellipsoid revolved about its vertical
  axis, `V = shape_factor · (π/6) · width² · height` (depth assumed ≈ width).
- **Mass**: `volume × density × edible_fraction`, so peel/core are excluded.
- **Nutrition**: per-100g values scaled to the edible mass.

Density, `shape_factor`, `edible_fraction`, and per-100g nutrients live in
`src/visionkit/data/nutrition.json` — add a row to support a new food.

Programmatic use:

```python
from visionkit.nutrition import estimate_food

est = estimate_food("plate.jpg", ref_width_mm=24,
                    model_path="models/fruit_mobilenetv2.keras")
print(est.fruit, est.volume_cm3, est.nutrients["calories_kcal"])
```

> **Approximate by design.** Volume comes from a single view plus literature
> densities/edible-fractions, so treat the numbers as estimates. Accuracy could
> be improved later with per-object cropping, contour/silhouette integration, a
> depth camera, or a learned image→volume regressor.

## Why transfer learning?

The original model was a small CNN trained from scratch on a few thousand
images. Transfer learning from an ImageNet-pretrained MobileNetV2 backbone
yields higher accuracy and far better robustness to lighting, background, and
viewpoint, while training faster. The original CNN remains available via
`--backbone cnn` for comparison.

A run of `fruit-train --epochs 20 --fine-tune-epochs 10` on the included dataset
reached **~97.9% test accuracy** (CPU training, ~20 minutes). The legacy CNN
baseline remains the committed default model; the trained MobileNetV2 model is
gitignored (large) and used via `--model`.

## Development

```bash
ruff check .      # lint
ruff format .     # format
```

## Notes & limitations

- **Measurement** assumes objects lie on a flat plane, the reference object is
  fully visible and left-most, and edges are well defined; lighting affects
  accuracy.
- **Classification** is limited to the five trained fruit classes; class names
  and ordering come from each model's `*.labels.json` sidecar (never
  hard-coded), so they always match the model that produced them.
- **Volume & nutrition** require a reference object for scale and assume a
  single food item per image from the five known classes. Volume uses an
  ellipsoid approximation and nutrition uses approximate literature values
  (`nutrition.json`), so results are estimates, not measurements.
