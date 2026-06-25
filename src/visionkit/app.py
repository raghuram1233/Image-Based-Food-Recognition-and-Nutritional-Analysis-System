"""Flask web UI for visionkit.

Image-only input — all measurement and model settings are automated.
Three tabs: Measure Objects · Classify Fruit · Food Estimate.

Launch:
    visionkit-app          (console script)
    python -m visionkit.app
"""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import cv2
from flask import Flask, render_template_string, request

from visionkit.measure import MeasurementConfig, annotate, measure_objects
from visionkit.nutrition import format_report
from visionkit.paths import DEFAULT_MODEL_PATH, MODELS_DIR

app = Flask(__name__)

# Prefer the trained MobileNetV2 model when available; fall back to legacy CNN.
_MOBILENET = MODELS_DIR / "fruit_mobilenetv2.keras"
_MODEL = str(_MOBILENET) if _MOBILENET.exists() else str(DEFAULT_MODEL_PATH)

# All settings automated — not exposed in the UI.
_MEAS_CFG = MeasurementConfig(ref_width_mm=25.0, scale=0.2, min_area=100)
_FOOD_REF_MM = 24.0


def _img_to_b64(bgr) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    return base64.b64encode(buf.tobytes()).decode() if ok else ""


def _save_upload(fs) -> str:
    suffix = Path(fs.filename).suffix or ".jpg"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    fs.save(path)
    return path


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>visionkit</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: #f0f2f5; }
    .card { border: none; border-radius: 12px; box-shadow: 0 1px 6px rgba(0,0,0,.08); }
    .upload-label {
      display: block; border: 2px dashed #ced4da; border-radius: 10px;
      padding: 2.5rem 1rem; text-align: center; cursor: pointer;
      background: #fafafa; transition: border-color .2s, background .2s;
    }
    .upload-label:hover { border-color: #0d6efd; background: #f0f6ff; }
    .upload-label input[type=file] { display: none; }
    #m-preview, #c-preview, #f-preview {
      max-width: 100%; max-height: 260px; border-radius: 8px;
      margin-top: 1rem; display: none;
    }
    .result-img { max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.12); }
    pre.result {
      background: #1e1e2e; color: #cdd6f4; padding: 1.25rem;
      border-radius: 8px; font-size: .85rem; white-space: pre-wrap; margin: 0;
    }
    .nav-tabs .nav-link { color: #495057; }
    .nav-tabs .nav-link.active { font-weight: 600; }
  </style>
</head>
<body>

<nav class="navbar navbar-dark bg-dark">
  <div class="container-fluid px-4">
    <span class="navbar-brand fw-semibold fs-5">visionkit</span>
    <span class="text-secondary small">Computer Vision Toolkit</span>
  </div>
</nav>

<div class="container py-4" style="max-width: 900px">

  <ul class="nav nav-tabs mb-4" role="tablist">
    <li class="nav-item">
      <button class="nav-link {% if tab == 'measure' %}active{% endif %}"
              data-bs-toggle="tab" data-bs-target="#pane-measure" type="button">
        Measure Objects
      </button>
    </li>
    <li class="nav-item">
      <button class="nav-link {% if tab == 'classify' %}active{% endif %}"
              data-bs-toggle="tab" data-bs-target="#pane-classify" type="button">
        Classify Fruit
      </button>
    </li>
    <li class="nav-item">
      <button class="nav-link {% if tab == 'food' %}active{% endif %}"
              data-bs-toggle="tab" data-bs-target="#pane-food" type="button">
        Food Estimate
      </button>
    </li>
  </ul>

  <div class="tab-content">

    <!-- ── MEASURE ─────────────────────────────────────────────────── -->
    <div class="tab-pane fade {% if tab == 'measure' %}show active{% endif %}" id="pane-measure">
      <div class="card p-4">
        <h5 class="mb-1">Measure Objects</h5>
        <p class="text-muted small mb-4">
          Place a <strong>reference object of known width (25 mm) on the left</strong> of
          the frame. The tool calibrates scale from it and measures every other object.
        </p>

        <form method="post" action="/measure" enctype="multipart/form-data">
          <label class="upload-label" for="m-file">
            <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="#adb5bd"
                 viewBox="0 0 16 16" class="mb-2">
              <path d="M4.406 1.342A5.53 5.53 0 0 1 8 0c2.69 0 4.923 2 5.166
                       4.579C14.758 4.804 16 6.137 16 7.773 16 9.569 14.502 11
                       12.687 11H10a.5.5 0 0 1 0-1h2.688C13.979 10 15 8.988 15
                       7.773c0-1.216-1.02-2.228-2.313-2.228h-.5v-.5C12.188 2.825
                       10.328 1 8 1a4.53 4.53 0 0 0-2.941 1.1c-.757.652-1.153
                       1.438-1.153 2.055v.448l-.445.049C2.064 4.805 1 5.952 1
                       7.318 1 8.785 2.23 10 3.781 10H6a.5.5 0 0 1 0 1H3.781C1.708
                       11 0 9.366 0 7.318c0-1.763 1.266-3.223 2.942-3.593.143-.863
                       .698-1.723 1.464-2.383z"/>
              <path d="M7.646 4.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5
                       5.707V14.5a.5.5 0 0 1-1 0V5.707L5.354 7.854a.5.5 0 1
                       1-.708-.708l3-3z"/>
            </svg>
            <p class="mb-0 text-muted">Click to select an image</p>
            <input type="file" id="m-file" name="image" accept="image/*"
                   onchange="preview(this,'m-preview'); this.form.querySelector('button').disabled=false">
            <img id="m-preview" alt="preview">
          </label>
          <button class="btn btn-primary mt-3 w-100" type="submit" disabled>Measure</button>
        </form>

        {% if error %}<div class="alert alert-danger mt-3 mb-0">{{ error }}</div>{% endif %}

        {% if measurements %}
        <hr class="mt-4">
        <h6 class="mb-3">Results</h6>
        <div class="row g-3 align-items-start">
          {% if img_b64 %}
          <div class="col-md-7">
            <img src="data:image/png;base64,{{ img_b64 }}" class="result-img" alt="annotated">
          </div>
          {% endif %}
          <div class="col-md-5">
            <pre class="result">{{ measurements }}</pre>
          </div>
        </div>
        {% endif %}
      </div>
    </div>

    <!-- ── CLASSIFY ────────────────────────────────────────────────── -->
    <div class="tab-pane fade {% if tab == 'classify' %}show active{% endif %}" id="pane-classify">
      <div class="card p-4">
        <h5 class="mb-1">Classify Fruit</h5>
        <p class="text-muted small mb-4">
          Upload a photo of a single fruit. Supported classes:
          <strong>Apple · Banana · Mango · Orange · Pineapple</strong>.
        </p>

        <form method="post" action="/classify" enctype="multipart/form-data">
          <label class="upload-label" for="c-file">
            <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="#adb5bd"
                 viewBox="0 0 16 16" class="mb-2">
              <path d="M4.406 1.342A5.53 5.53 0 0 1 8 0c2.69 0 4.923 2 5.166
                       4.579C14.758 4.804 16 6.137 16 7.773 16 9.569 14.502 11
                       12.687 11H10a.5.5 0 0 1 0-1h2.688C13.979 10 15 8.988 15
                       7.773c0-1.216-1.02-2.228-2.313-2.228h-.5v-.5C12.188 2.825
                       10.328 1 8 1a4.53 4.53 0 0 0-2.941 1.1c-.757.652-1.153
                       1.438-1.153 2.055v.448l-.445.049C2.064 4.805 1 5.952 1
                       7.318 1 8.785 2.23 10 3.781 10H6a.5.5 0 0 1 0 1H3.781C1.708
                       11 0 9.366 0 7.318c0-1.763 1.266-3.223 2.942-3.593.143-.863
                       .698-1.723 1.464-2.383z"/>
              <path d="M7.646 4.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5
                       5.707V14.5a.5.5 0 0 1-1 0V5.707L5.354 7.854a.5.5 0 1
                       1-.708-.708l3-3z"/>
            </svg>
            <p class="mb-0 text-muted">Click to select an image</p>
            <input type="file" id="c-file" name="image" accept="image/*"
                   onchange="preview(this,'c-preview'); this.form.querySelector('button').disabled=false">
            <img id="c-preview" alt="preview">
          </label>
          <button class="btn btn-success mt-3 w-100" type="submit" disabled>Classify</button>
        </form>

        {% if error %}<div class="alert alert-danger mt-3 mb-0">{{ error }}</div>{% endif %}

        {% if predictions %}
        <hr class="mt-4">
        <h6 class="mb-3">Predictions</h6>
        <pre class="result">{{ predictions }}</pre>
        {% endif %}
      </div>
    </div>

    <!-- ── FOOD ESTIMATE ───────────────────────────────────────────── -->
    <div class="tab-pane fade {% if tab == 'food' %}show active{% endif %}" id="pane-food">
      <div class="card p-4">
        <h5 class="mb-1">Food Estimate</h5>
        <p class="text-muted small mb-4">
          Place a <strong>24 mm reference object on the left</strong> (e.g. a standard coin)
          next to a single fruit. The pipeline measures dimensions, classifies the fruit,
          and estimates volume, mass, and full nutritional breakdown.
        </p>

        <form method="post" action="/food" enctype="multipart/form-data">
          <label class="upload-label" for="f-file">
            <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="#adb5bd"
                 viewBox="0 0 16 16" class="mb-2">
              <path d="M4.406 1.342A5.53 5.53 0 0 1 8 0c2.69 0 4.923 2 5.166
                       4.579C14.758 4.804 16 6.137 16 7.773 16 9.569 14.502 11
                       12.687 11H10a.5.5 0 0 1 0-1h2.688C13.979 10 15 8.988 15
                       7.773c0-1.216-1.02-2.228-2.313-2.228h-.5v-.5C12.188 2.825
                       10.328 1 8 1a4.53 4.53 0 0 0-2.941 1.1c-.757.652-1.153
                       1.438-1.153 2.055v.448l-.445.049C2.064 4.805 1 5.952 1
                       7.318 1 8.785 2.23 10 3.781 10H6a.5.5 0 0 1 0 1H3.781C1.708
                       11 0 9.366 0 7.318c0-1.763 1.266-3.223 2.942-3.593.143-.863
                       .698-1.723 1.464-2.383z"/>
              <path d="M7.646 4.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5
                       5.707V14.5a.5.5 0 0 1-1 0V5.707L5.354 7.854a.5.5 0 1
                       1-.708-.708l3-3z"/>
            </svg>
            <p class="mb-0 text-muted">Click to select an image</p>
            <input type="file" id="f-file" name="image" accept="image/*"
                   onchange="preview(this,'f-preview'); this.form.querySelector('button').disabled=false">
            <img id="f-preview" alt="preview">
          </label>
          <button class="btn btn-warning mt-3 w-100" type="submit" disabled>Estimate</button>
        </form>

        {% if error %}<div class="alert alert-danger mt-3 mb-0">{{ error }}</div>{% endif %}

        {% if report %}
        <hr class="mt-4">
        <h6 class="mb-3">Nutrition Estimate</h6>
        <pre class="result">{{ report }}</pre>
        {% endif %}
      </div>
    </div>

  </div><!-- tab-content -->
</div><!-- container -->

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
<script>
function preview(input, id) {
  var img = document.getElementById(id);
  if (input.files && input.files[0]) {
    var r = new FileReader();
    r.onload = function(e) { img.src = e.target.result; img.style.display = 'block'; };
    r.readAsDataURL(input.files[0]);
  }
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(TEMPLATE, tab="measure")


@app.route("/measure", methods=["POST"])
def measure_route():
    f = request.files.get("image")
    if not f or not f.filename:
        return render_template_string(TEMPLATE, tab="measure", error="Please select an image.")

    tmp = _save_upload(f)
    try:
        bgr = cv2.imread(tmp)
        if bgr is None:
            return render_template_string(TEMPLATE, tab="measure", error="Cannot read the uploaded image.")

        result = measure_objects(bgr, _MEAS_CFG)
        if not result.objects:
            return render_template_string(
                TEMPLATE,
                tab="measure",
                error="No objects detected. Make sure the reference object is clearly visible and left-most.",
            )

        ann_b64 = _img_to_b64(annotate(bgr, result))
        lines = [f"Scale: {result.pixels_per_metric:.3f} px/mm\n"]
        for i, obj in enumerate(result.objects):
            tag = " (reference)" if obj.is_reference else ""
            lines.append(f"Object {i + 1}:  {obj.width_mm:.1f} x {obj.height_mm:.1f} mm{tag}")

        return render_template_string(
            TEMPLATE, tab="measure", img_b64=ann_b64, measurements="\n".join(lines)
        )
    finally:
        os.unlink(tmp)


@app.route("/classify", methods=["POST"])
def classify_route():
    f = request.files.get("image")
    if not f or not f.filename:
        return render_template_string(TEMPLATE, tab="classify", error="Please select an image.")

    tmp = _save_upload(f)
    try:
        try:
            from visionkit.classify import FruitClassifier
        except ImportError:
            return render_template_string(
                TEMPLATE, tab="classify",
                error="TensorFlow is not installed. Run: pip install '.[ml]'",
            )

        clf = FruitClassifier(_MODEL)
        pred = clf.predict(tmp)[0]
        ranked = sorted(pred.probabilities.items(), key=lambda x: x[1], reverse=True)
        lines = [f"{label:<12}  {conf * 100:.1f}%" for label, conf in ranked]
        return render_template_string(TEMPLATE, tab="classify", predictions="\n".join(lines))
    except FileNotFoundError as exc:
        return render_template_string(TEMPLATE, tab="classify", error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return render_template_string(TEMPLATE, tab="classify", error=str(exc))
    finally:
        os.unlink(tmp)


@app.route("/food", methods=["POST"])
def food_route():
    f = request.files.get("image")
    if not f or not f.filename:
        return render_template_string(TEMPLATE, tab="food", error="Please select an image.")

    tmp = _save_upload(f)
    try:
        try:
            from visionkit.nutrition import estimate_food
        except ImportError:
            return render_template_string(
                TEMPLATE, tab="food",
                error="TensorFlow is not installed. Run: pip install '.[ml]'",
            )

        est = estimate_food(tmp, ref_width_mm=_FOOD_REF_MM, model_path=_MODEL)
        return render_template_string(TEMPLATE, tab="food", report=format_report(est))
    except FileNotFoundError as exc:
        return render_template_string(TEMPLATE, tab="food", error=str(exc))
    except ValueError as exc:
        return render_template_string(TEMPLATE, tab="food", error=str(exc))
    except KeyError as exc:
        return render_template_string(TEMPLATE, tab="food", error=str(exc.args[0] if exc.args else exc))
    except Exception as exc:  # noqa: BLE001
        return render_template_string(TEMPLATE, tab="food", error=str(exc))
    finally:
        os.unlink(tmp)


def app_main() -> None:
    app.run(debug=False, port=5000, use_reloader=False)


if __name__ == "__main__":
    app_main()
