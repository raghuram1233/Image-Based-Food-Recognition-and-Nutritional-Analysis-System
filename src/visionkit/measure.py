"""Object dimension measurement core and CLI.

Pipeline:
1. Pre-process (grayscale, blur, Canny edges, morphological close).
2. Find external contours and sort them left-to-right.
3. Fit rotated bounding box for each contour above minimum area.
4. Calibrate pixels-per-millimetre ratio from the leftmost (reference) object,
   then convert all objects' pixel sizes to millimetres.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from visionkit.logging_utils import configure_logging, get_logger

log = get_logger("visionkit.measure")

# Annotation colours (BGR).
_COLOR_BOX = (0, 255, 0)
_COLOR_CORNER = (0, 0, 255)
_COLOR_MIDPOINT = (255, 0, 0)
_COLOR_LINE = (255, 0, 255)
_COLOR_TEXT = (255, 255, 255)


@dataclass(frozen=True)
class MeasurementConfig:
    """Tunable parameters for the measurement pipeline."""

    ref_width_mm: float = 25.0
    """Real-world width of the left-most (reference) object, in millimetres."""

    scale: float = 1.0
    """Resize factor applied before processing. Measurements are scale
    invariant (the ratio cancels), but downscaling speeds up large images."""

    min_area: float = 100.0
    """Minimum contour area (in pixels) to be considered an object."""

    blur_kernel: int = 7
    """Gaussian blur kernel size (forced odd)."""

    canny_low: int = 50
    canny_high: int = 100

    def __post_init__(self) -> None:
        if self.ref_width_mm <= 0:
            raise ValueError("ref_width_mm must be positive")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.blur_kernel < 1:
            raise ValueError("blur_kernel must be >= 1")


@dataclass
class ObjectMeasurement:
    """A single measured object."""

    width_mm: float
    height_mm: float
    box: np.ndarray  # ordered 4x2 corner points (tl, tr, br, bl) in image coords
    is_reference: bool = False


@dataclass
class MeasurementResult:
    """Result of measuring all objects in an image."""

    objects: list[ObjectMeasurement] = field(default_factory=list)
    pixels_per_metric: float | None = None
    image_size: tuple[int, int] = (0, 0)  # (width, height) after resize


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as a BGR array.

    Raises:
        FileNotFoundError: if OpenCV cannot read the file.
    """
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path!r}")
    return image


def _midpoint(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left has the smallest x + y
    rect[2] = pts[np.argmax(s)]  # bottom-right has the largest x + y
    diff = np.diff(pts, axis=1).ravel()  # y - x
    rect[1] = pts[np.argmin(diff)]  # top-right has the smallest y - x
    rect[3] = pts[np.argmax(diff)]  # bottom-left has the largest y - x
    return rect


def _find_contours_sorted_ltr(edged: np.ndarray) -> list[np.ndarray]:
    """Find external contours and sort them left-to-right.

    Handles both OpenCV 3 (3-tuple) and OpenCV 4 (2-tuple) return signatures.
    """
    found = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    return sorted(contours, key=lambda c: cv2.boundingRect(c)[0])


def preprocess(image: np.ndarray, config: MeasurementConfig) -> np.ndarray:
    """Produce a cleaned edge map from a BGR image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = config.blur_kernel | 1  # force odd
    gray = cv2.GaussianBlur(gray, (k, k), 0)
    edged = cv2.Canny(gray, config.canny_low, config.canny_high)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged


def measure_objects(image: np.ndarray, config: MeasurementConfig) -> MeasurementResult:
    """Measure every object in ``image``.

    The left-most qualifying contour is treated as the reference object and is
    used to calibrate the pixels-per-millimetre ratio.
    """
    if config.scale != 1.0:
        h, w = image.shape[:2]
        image = cv2.resize(image, (max(1, int(w * config.scale)), max(1, int(h * config.scale))))

    h, w = image.shape[:2]
    result = MeasurementResult(image_size=(w, h))

    edged = preprocess(image, config)
    pixels_per_metric: float | None = None

    for c in _find_contours_sorted_ltr(edged):
        if cv2.contourArea(c) < config.min_area:
            continue

        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        box = order_points(np.array(box, dtype="float32"))
        (tl, tr, br, bl) = box

        # Pixel distances along the two axes of the box.
        width_px = float(np.hypot(*(np.array(_midpoint(tl, bl)) - np.array(_midpoint(tr, br)))))
        height_px = float(np.hypot(*(np.array(_midpoint(tl, tr)) - np.array(_midpoint(bl, br)))))

        is_reference = pixels_per_metric is None
        if is_reference:
            pixels_per_metric = width_px / config.ref_width_mm

        result.objects.append(
            ObjectMeasurement(
                width_mm=width_px / pixels_per_metric,
                height_mm=height_px / pixels_per_metric,
                box=box,
                is_reference=is_reference,
            )
        )

    result.pixels_per_metric = pixels_per_metric
    return result


def annotate(image: np.ndarray, result: MeasurementResult) -> np.ndarray:
    """Return a copy of ``image`` with every measurement drawn on it."""
    out = image.copy()
    if result.image_size != (out.shape[1], out.shape[0]):
        out = cv2.resize(out, result.image_size)

    for obj in result.objects:
        box = obj.box
        cv2.drawContours(out, [box.astype("int")], -1, _COLOR_BOX, 2)
        for x, y in box:
            cv2.circle(out, (int(x), int(y)), 5, _COLOR_CORNER, -1)

        (tl, tr, br, bl) = box
        tltr = _midpoint(tl, tr)
        blbr = _midpoint(bl, br)
        tlbl = _midpoint(tl, bl)
        trbr = _midpoint(tr, br)
        for px, py in (tltr, blbr, tlbl, trbr):
            cv2.circle(out, (int(px), int(py)), 5, _COLOR_MIDPOINT, -1)

        cv2.line(out, _to_int(tltr), _to_int(blbr), _COLOR_LINE, 2)
        cv2.line(out, _to_int(tlbl), _to_int(trbr), _COLOR_LINE, 2)

        cv2.putText(
            out,
            f"{obj.height_mm:.1f}mm",
            (int(tltr[0] - 15), int(tltr[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            _COLOR_TEXT,
            2,
        )
        cv2.putText(
            out,
            f"{obj.width_mm:.1f}mm",
            (int(trbr[0] + 10), int(trbr[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            _COLOR_TEXT,
            2,
        )
    return out


def _to_int(pt: tuple[float, float]) -> tuple[int, int]:
    return (int(pt[0]), int(pt[1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="measure-object",
        description="Measure real-world object dimensions in an image using a "
        "reference object (the left-most detected object) for scale.",
    )
    p.add_argument("image", help="Path to the input image.")
    p.add_argument(
        "--ref-width",
        type=float,
        default=25.0,
        help="Width of the reference object in millimetres (default: 25).",
    )
    p.add_argument(
        "--scale", type=float, default=1.0, help="Resize factor before processing (default: 1.0)."
    )
    p.add_argument(
        "--min-area",
        type=float,
        default=100.0,
        help="Minimum contour area in pixels (default: 100).",
    )
    p.add_argument("--blur", type=int, default=7, help="Gaussian blur kernel size (default: 7).")
    p.add_argument(
        "--canny",
        type=int,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=(50, 100),
        help="Canny edge thresholds (default: 50 100).",
    )
    p.add_argument("-o", "--output", help="Save the annotated image to this path.")
    p.add_argument(
        "--show", action="store_true", help="Display the annotated image in a window (needs a GUI)."
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    config = MeasurementConfig(
        ref_width_mm=args.ref_width,
        scale=args.scale,
        min_area=args.min_area,
        blur_kernel=args.blur,
        canny_low=args.canny[0],
        canny_high=args.canny[1],
    )

    try:
        image = load_image(args.image)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    result = measure_objects(image, config)
    if not result.objects:
        log.warning("No objects detected. Try lowering --min-area or adjusting --canny.")
        return 2

    log.info(
        "Detected %d object(s); pixels/mm = %.3f",
        len(result.objects),
        result.pixels_per_metric or 0.0,
    )
    for i, obj in enumerate(result.objects):
        tag = " (reference)" if obj.is_reference else ""
        log.info("Object %d: %.1f x %.1f mm%s", i + 1, obj.width_mm, obj.height_mm, tag)

    if args.output or args.show:
        annotated = annotate(image, result)
        if args.output:
            if cv2.imwrite(args.output, annotated):
                log.info("Saved annotated image to %s", args.output)
            else:
                log.error("Failed to write output image to %s", args.output)
                return 1
        if args.show:
            cv2.imshow("Measurements", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
