"""Area measurements and the experimental severity / priority rules.

NOTE: the severity thresholds below are experimental project rules.
They are NOT clinically validated.
"""

import cv2
import numpy as np
from scipy import ndimage


# ------------------------------------------------------------------
# Brain mask
# ------------------------------------------------------------------

def estimate_brain_mask(image_rgb, min_fraction=0.05):
    """Estimate a boolean mask of the head/brain region of an MRI slice.

    Steps: grayscale -> blur -> Otsu threshold -> small morphological open
    -> keep the largest connected component -> fill internal holes
    (ventricles, dark CSF) so they count as brain area.

    Falls back to a fixed low threshold if Otsu gives an implausibly
    small region (e.g. very low-contrast images).
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    otsu_value, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    mask = _clean_mask(binary > 0)

    if mask.sum() < min_fraction * mask.size:
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask = _clean_mask(binary > 0)

    return mask


def _clean_mask(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
    ).astype(bool)
    mask = keep_largest_component(mask)
    return ndimage.binary_fill_holes(mask)


def keep_largest_component(mask):
    """Keep only the largest connected component of a boolean mask."""
    labeled, count = ndimage.label(mask)
    if count <= 1:
        return mask.astype(bool)
    sizes = ndimage.sum(mask, labeled, range(1, count + 1))
    return labeled == (int(np.argmax(sizes)) + 1)


# ------------------------------------------------------------------
# Areas
# ------------------------------------------------------------------

def calculate_mask_area(mask):
    """Area in pixels of a boolean mask."""
    return int(np.count_nonzero(mask))


def calculate_tumor_area(width, height):
    """Bounding-box area (kept for the notebooks / backwards compatibility)."""
    return width * height


def calculate_tumor_percentage(tumor_area, brain_area):
    if brain_area <= 0:
        raise ValueError("Brain area must be greater than zero.")
    return (tumor_area / brain_area) * 100


# ------------------------------------------------------------------
# Severity (experimental)
# ------------------------------------------------------------------

def experimental_severity_priority(tumor_percentage):
    if tumor_percentage < 1:
        return "Low", "Routine"
    elif tumor_percentage < 3:
        return "Moderate", "Attention"
    elif tumor_percentage < 5:
        return "High", "High"
    else:
        return "Very High", "Urgent"
