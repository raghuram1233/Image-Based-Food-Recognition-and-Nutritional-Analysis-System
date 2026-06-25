"""visionkit: a small, professional computer-vision toolkit.

Two independent tools share this package:

* ``visionkit.measure`` -- measure real-world object dimensions from an image
  using a reference object for scale calibration (pure OpenCV/NumPy).
* ``visionkit.classify`` -- train and run a fruit image classifier
  (TensorFlow/Keras, imported lazily so the measurement tool has no ML deps).
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
