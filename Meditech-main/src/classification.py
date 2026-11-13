"""Optional tumor-type / grade classifier.

This is a second, separate model: an Ultralytics *classification* model
(e.g. yolo11n-cls) trained on tumor crops. The class names come from the
training folders, so the same code works for tumor TYPE
(glioma / meningioma / pituitary / ...) or tumor GRADE (I-IV), depending
on how you label your data. See train_classifier.py.

If models/tumor_classifier.pt does not exist, everything here returns
None and the app reports the classification as "Not Available".
"""

import os

import numpy as np

CLASSIFIER_PATH = "models/tumor_classifier.pt"


def load_classifier(path=CLASSIFIER_PATH):
    """Return the classifier, or None if the weights are missing/unusable."""
    if not os.path.exists(path):
        return None
    try:
        from ultralytics import YOLO
        return YOLO(path)
    except Exception:
        return None


def crop_with_padding(image_rgb, box, pad_fraction=0.25):
    """Crop the box plus some context (tumor type often depends on
    location and surrounding tissue, so a tight crop is not ideal)."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    pad_x, pad_y = (x2 - x1) * pad_fraction, (y2 - y1) * pad_fraction
    x1, y1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
    x2, y2 = int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y))
    if x2 <= x1 or y2 <= y1:
        return image_rgb
    return np.ascontiguousarray(image_rgb[y1:y2, x1:x2])


def classify_tumor(classifier, image_rgb, box, top_k=3):
    """Classify the tumor inside `box`.

    Returns (label, confidence_percent, ranked) where `ranked` is a list
    of (label, confidence_percent) for the top_k classes, or None if no
    classifier is available or prediction fails.
    """
    if classifier is None:
        return None
    try:
        crop = crop_with_padding(image_rgb, box)
        result = classifier.predict(crop, verbose=False)[0]
        probs = result.probs
        names = result.names
        order = [int(i) for i in probs.data.argsort(descending=True)[:top_k]]
        ranked = [
            (names[i], float(probs.data[i]) * 100) for i in order
        ]
        return ranked[0][0], ranked[0][1], ranked
    except Exception:
        return None
