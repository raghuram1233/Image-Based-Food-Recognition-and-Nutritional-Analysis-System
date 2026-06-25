"""Fruit classification: datasets, models, training, inference, and CLIs.

Uses MobileNetV2 transfer learning (default) or a custom CNN.
TensorFlow imports are kept lazy inside functions/methods to prevent
import-time performance penalty on light dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from visionkit.logging_utils import configure_logging, get_logger
from visionkit.paths import DEFAULT_MODEL_PATH, TEST_DIR, TRAIN_DIR, labels_sidecar

log = get_logger("visionkit.classify")

_TF_HINT = "TensorFlow is required for this command. Install it with: pip install '.[ml]'"


def _tensorflow_available() -> bool:
    return importlib.util.find_spec("tensorflow") is not None


# =========================================================================== #
# Label Mapping Sidecar
# =========================================================================== #

@dataclass
class LabelMap:
    """Maps model output indices to human-readable class names plus metadata.

    Attributes:
        classes: Class names in model-output index order.
        img_size: (height, width) the model expects.
        preprocessing: How raw pixels must be prepared before the model:
            ``"model"``  -- preprocessing is baked into the model (feed 0-255);
            ``"rescale"`` -- divide pixel values by 255 before predicting.
    """

    classes: list[str]
    img_size: tuple[int, int] = (224, 224)
    preprocessing: str = "model"
    metadata: dict = field(default_factory=dict)

    def name(self, index: int) -> str:
        return self.classes[index]

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classes": self.classes,
            "img_size": list(self.img_size),
            "preprocessing": self.preprocessing,
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> LabelMap:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        size = data.get("img_size", [224, 224])
        return cls(
            classes=list(data["classes"]),
            img_size=(int(size[0]), int(size[1])),
            preprocessing=data.get("preprocessing", "model"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_directory(cls, train_dir: str | Path, **kwargs) -> LabelMap:
        """Infer class names from sub-directory names (sorted, as Keras does)."""
        train_dir = Path(train_dir)
        classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
        if not classes:
            raise ValueError(f"No class sub-directories found under {train_dir}")
        return cls(classes=classes, **kwargs)


# =========================================================================== #
# Dataset Building
# =========================================================================== #

@dataclass
class Datasets:
    """Container for the prepared datasets and the discovered class names."""

    train: object  # tf.data.Dataset
    val: object  # tf.data.Dataset
    test: object | None  # tf.data.Dataset or None
    class_names: list[str]


def build_datasets(
    train_dir: str | Path,
    test_dir: str | Path | None = None,
    img_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
    seed: int = 42,
) -> Datasets:
    """Build prefetched train/val (and optional test) datasets."""
    import tensorflow as tf

    train_dir = str(train_dir)
    common = dict(image_size=img_size, batch_size=batch_size, label_mode="int")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=validation_split, subset="training", seed=seed, **common
    )
    class_names = list(train_ds.class_names)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, validation_split=validation_split, subset="validation", seed=seed, **common
    )

    test_ds = None
    if test_dir is not None and Path(test_dir).exists():
        test_ds = tf.keras.utils.image_dataset_from_directory(
            str(test_dir), shuffle=False, **common
        )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(autotune)

    return Datasets(train=train_ds, val=val_ds, test=test_ds, class_names=class_names)


# =========================================================================== #
# Model Architectures
# =========================================================================== #

BACKBONES = ("mobilenetv2", "cnn")


def _augmentation_layers():
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


def build_mobilenetv2(num_classes: int, img_size: tuple[int, int] = (224, 224)):
    """Build a MobileNetV2 transfer-learning classifier (backbone frozen)."""
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import (  # type: ignore
        MobileNetV2,
        preprocess_input,
    )

    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = _augmentation_layers()(inputs)
    x = preprocess_input(x)
    base = MobileNetV2(input_shape=(*img_size, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="fruit_mobilenetv2")
    model._visionkit_base = base  # stash for fine-tuning
    return model


def build_cnn(num_classes: int, img_size: tuple[int, int] = (128, 128)):
    """The original small CNN, with rescaling baked in."""
    import tensorflow as tf
    from tensorflow.keras import layers

    return tf.keras.Sequential(
        [
            tf.keras.Input(shape=(*img_size, 3)),
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(2),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="fruit_cnn",
    )


def build_model(backbone: str, num_classes: int, img_size: tuple[int, int]):
    """Dispatch to the requested architecture."""
    if backbone == "mobilenetv2":
        return build_mobilenetv2(num_classes, img_size)
    if backbone == "cnn":
        return build_cnn(num_classes, img_size)
    raise ValueError(f"Unknown backbone {backbone!r}; choose from {BACKBONES}")


def enable_fine_tuning(model, num_layers: int = 30) -> None:
    """Unfreeze the top ``num_layers`` of a MobileNetV2 backbone in place."""
    base = getattr(model, "_visionkit_base", None)
    if base is None:
        return
    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False


# =========================================================================== #
# Model Training Pipeline
# =========================================================================== #

@dataclass
class TrainConfig:
    train_dir: str
    test_dir: str | None = None
    backbone: str = "mobilenetv2"
    img_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 20
    fine_tune_epochs: int = 0
    learning_rate: float = 1e-3
    fine_tune_lr: float = 1e-5
    validation_split: float = 0.2
    seed: int = 42
    model_out: str = "models/fruit_mobilenetv2.keras"
    labels_out: str | None = None
    metrics_out: str | None = None
    plot_out: str | None = None


def _callbacks(model_out: str):
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_out, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]


def _plot_history(history: dict, path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping training-curve plot.")
        return

    epochs = range(1, len(history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(epochs, history["accuracy"], label="train")
    ax1.plot(epochs, history.get("val_accuracy", []), label="val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("epoch")
    ax1.legend()
    ax2.plot(epochs, history["loss"], label="train")
    ax2.plot(epochs, history.get("val_loss", []), label="val")
    ax2.set_title("Loss")
    ax2.set_xlabel("epoch")
    ax2.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    log.info("Saved training curves to %s", path)


def train(config: TrainConfig) -> dict:
    """Run the full training pipeline and persist artefacts. Returns metrics."""
    import tensorflow as tf

    stem = str(Path(config.model_out).with_suffix(""))
    labels_out = config.labels_out or str(labels_sidecar(config.model_out))
    metrics_out = config.metrics_out or f"{stem}.metrics.json"
    plot_out = config.plot_out or f"{stem}_curves.png"

    log.info("Building datasets from %s", config.train_dir)
    datasets = build_datasets(
        config.train_dir,
        test_dir=config.test_dir,
        img_size=config.img_size,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        seed=config.seed,
    )
    num_classes = len(datasets.class_names)
    log.info("Classes (%d): %s", num_classes, datasets.class_names)

    model = build_model(config.backbone, num_classes, config.img_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=lambda s: log.info(s))

    log.info("Phase 1: training for up to %d epochs", config.epochs)
    history = model.fit(
        datasets.train,
        validation_data=datasets.val,
        epochs=config.epochs,
        callbacks=_callbacks(config.model_out),
    )
    merged = {k: list(v) for k, v in history.history.items()}

    if config.fine_tune_epochs > 0 and config.backbone == "mobilenetv2":
        log.info("Phase 2: fine-tuning for up to %d epochs", config.fine_tune_epochs)
        enable_fine_tuning(model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.fine_tune_lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        ft = model.fit(
            datasets.train,
            validation_data=datasets.val,
            epochs=config.epochs + config.fine_tune_epochs,
            initial_epoch=len(merged["loss"]),
            callbacks=_callbacks(config.model_out),
        )
        for k, v in ft.history.items():
            merged.setdefault(k, []).extend(list(v))

    # Persist artefacts.
    Path(config.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(config.model_out)
    log.info("Saved model to %s", config.model_out)

    LabelMap(
        classes=datasets.class_names,
        img_size=config.img_size,
        preprocessing="model",  # preprocessing is baked into the graph
        metadata={"backbone": config.backbone},
    ).save(labels_out)
    log.info("Saved labels to %s", labels_out)

    metrics: dict = {"config": asdict(config), "history": merged}
    if datasets.test is not None:
        test_loss, test_acc = model.evaluate(datasets.test, verbose=0)
        metrics["test"] = {"loss": float(test_loss), "accuracy": float(test_acc)}
        log.info("Test accuracy: %.4f", test_acc)

    Path(metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _plot_history(merged, plot_out)
    return metrics


# =========================================================================== #
# Model Inference Wrapper
# =========================================================================== #

class Prediction:
    """A single image's prediction."""

    def __init__(self, label: str, confidence: float, probabilities: dict[str, float]):
        self.label = label
        self.confidence = confidence
        self.probabilities = probabilities

    def __repr__(self) -> str:
        return f"Prediction(label={self.label!r}, confidence={self.confidence:.4f})"


class FruitClassifier:
    """Load a trained model once and classify images."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        labels_path: str | Path | None = None,
    ):
        import tensorflow as tf

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        log.info("Loading model from %s", model_path)
        self.model = tf.keras.models.load_model(model_path)
        self.labels = self._resolve_labels(model_path, labels_path)
        self.img_size = self._resolve_img_size()
        log.info(
            "Ready: %d classes, input %s, preprocessing=%s",
            self.labels.num_classes,
            self.img_size,
            self.labels.preprocessing,
        )

    def _resolve_labels(self, model_path: Path, labels_path: str | Path | None) -> LabelMap:
        # 1) An explicitly provided labels file wins.
        if labels_path and Path(labels_path).exists():
            return LabelMap.load(labels_path)
        # 2) Otherwise look for the sidecar that sits next to the model.
        sidecar = labels_sidecar(model_path)
        if sidecar.exists():
            log.info("Using labels sidecar %s", sidecar.name)
            return LabelMap.load(sidecar)
        # 3) Fall back to the model's output dimension with generic names.
        n = int(self.model.output_shape[-1])
        log.warning(
            "No labels file found next to %s; using generic names class_0..class_%d",
            model_path.name,
            n - 1,
        )
        return LabelMap(classes=[f"class_{i}" for i in range(n)], preprocessing="rescale")

    def _resolve_img_size(self) -> tuple[int, int]:
        shape = self.model.input_shape  # (None, H, W, C)
        h, w = shape[1], shape[2]
        if h and w:
            return (int(h), int(w))
        return self.labels.img_size

    def _preprocess(self, paths: list[str]) -> np.ndarray:
        batch = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {p!r}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size[1], self.img_size[0]))
            batch.append(img.astype("float32"))
        arr = np.stack(batch)
        if self.labels.preprocessing == "rescale":
            arr = arr / 255.0
        return arr

    def predict(self, image_paths: str | list[str]) -> list[Prediction]:
        """Classify one path or a list of paths (batched in a single call)."""
        paths = [image_paths] if isinstance(image_paths, str) else list(image_paths)
        probs = self.model.predict(self._preprocess(paths), verbose=0)
        results = []
        for row in probs:
            idx = int(np.argmax(row))
            results.append(
                Prediction(
                    label=self.labels.name(idx),
                    confidence=float(row[idx]),
                    probabilities={self.labels.name(i): float(v) for i, v in enumerate(row)},
                )
            )
        return results


# =========================================================================== #
# CLI Parsers and Entry Points
# =========================================================================== #

def _classify_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fruit-classify", description="Classify fruit images with a trained model."
    )
    p.add_argument("images", nargs="+", help="One or more image paths.")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="Path to the trained model.")
    p.add_argument(
        "--labels",
        default=None,
        help="Path to a labels file (default: auto-detect '<model>.labels.json').",
    )
    p.add_argument("--top", type=int, default=1, help="Show the top-K classes (default: 1).")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def classify_main(argv: list[str] | None = None) -> int:
    args = _classify_parser().parse_args(argv)
    configure_logging(args.verbose)

    if not _tensorflow_available():
        log.error(_TF_HINT)
        return 1

    try:
        clf = FruitClassifier(args.model, args.labels)
        predictions = clf.predict(args.images)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    for path, pred in zip(args.images, predictions, strict=True):
        ranked = sorted(pred.probabilities.items(), key=lambda kv: kv[1], reverse=True)
        top = ", ".join(f"{name} {prob * 100:.1f}%" for name, prob in ranked[: args.top])
        print(f"{path}: {top}")
    return 0


def _train_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fruit-train", description="Train a fruit classifier (transfer learning by default)."
    )
    p.add_argument("--train-dir", default=str(TRAIN_DIR), help="Training data directory.")
    p.add_argument("--test-dir", default=str(TEST_DIR), help="Test data directory.")
    p.add_argument("--backbone", choices=["mobilenetv2", "cnn"], default="mobilenetv2")
    p.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Square input size (default: 224 for mobilenetv2, 128 for cnn).",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=0,
        help="Extra epochs unfreezing the backbone (mobilenetv2 only).",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--model-out", default=None, help="Where to save the model.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def train_main(argv: list[str] | None = None) -> int:
    args = _train_parser().parse_args(argv)
    configure_logging(args.verbose)

    if not _tensorflow_available():
        log.error(_TF_HINT)
        return 1

    img = args.img_size or (224 if args.backbone == "mobilenetv2" else 128)
    model_out = args.model_out or f"models/fruit_{args.backbone}.keras"
    config = TrainConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        backbone=args.backbone,
        img_size=(img, img),
        batch_size=args.batch_size,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.lr,
        model_out=model_out,
    )
    metrics = train(config)
    if "test" in metrics:
        log.info("Done. Test accuracy: %.4f", metrics["test"]["accuracy"])
    else:
        log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(classify_main())
