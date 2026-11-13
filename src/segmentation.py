"""Tumor segmentation inside a detected bounding box.

Two methods:

1. SAM / MobileSAM (preferred). The YOLO box is used as a prompt and SAM
   returns a pixel mask. Needs a SAM weights file (see README).
2. GrabCut fallback. A classical, training-free method that refines the
   box into a mask. Less accurate than a trained segmentation model.

If both fail (degenerate result), the bounding box itself is returned,
and the returned method name says so.

IMPORTANT: neither method is a trained tumor segmentation model.
For clinically meaningful masks, train a segmentation model (e.g.
YOLO11-seg or nnU-Net) on images with expert-drawn masks.
"""

import cv2
import numpy as np
from scipy import ndimage

from src.assessment import keep_largest_component


def box_to_mask(shape, box):
    """Rectangular mask for an (x1, y1, x2, y2) box."""
    h, w = shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _clip_box(box, w, h):
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, x2 = max(0, min(x1, w - 1)), max(1, min(x2, w))
    y1, y2 = max(0, min(y1, h - 1)), max(1, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


# ------------------------------------------------------------------
# SAM
# ------------------------------------------------------------------

def segment_with_sam(sam_model, image_rgb, box):
    """Box-prompted SAM segmentation via the Ultralytics SAM wrapper."""
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)

    results = sam_model(
        image_rgb, bboxes=[[x1, y1, x2, y2]], verbose=False
    )
    masks = results[0].masks
    if masks is None or len(masks.data) == 0:
        return None

    mask = masks.data[0].cpu().numpy() > 0.5
    if mask.shape != (h, w):
        mask = cv2.resize(
            mask.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


# ------------------------------------------------------------------
# GrabCut fallback
# ------------------------------------------------------------------

def segment_with_grabcut(image_rgb, box, brain_mask=None,
                         pad_fraction=0.5, iterations=5):
    """Refine a bounding box into a mask with GrabCut.

    GrabCut works on a padded window around the box: everything outside
    the box is "probably background", the box interior "probably
    foreground". The result is restricted to the box (and the brain mask
    if given), cleaned and reduced to its largest component.
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None  # too small for GrabCut to be meaningful

    pad_x, pad_y = int(bw * pad_fraction) + 2, int(bh * pad_fraction) + 2
    wx1, wy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    wx2, wy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

    window = np.ascontiguousarray(image_rgb[wy1:wy2, wx1:wx2])
    gc_mask = np.full(window.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[y1 - wy1:y2 - wy1, x1 - wx1:x2 - wx1] = cv2.GC_PR_FGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            window, gc_mask, None, bgd, fgd, iterations,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return None

    fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)

    mask = np.zeros((h, w), dtype=bool)
    mask[wy1:wy2, wx1:wx2] = fg
    mask &= box_to_mask(image_rgb.shape, (x1, y1, x2, y2))
    if brain_mask is not None:
        mask &= brain_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
    ).astype(bool)
    if not mask.any():
        return None
    return ndimage.binary_fill_holes(keep_largest_component(mask))


# ------------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------------

def segment_tumor(image_rgb, box, brain_mask=None, sam_model=None,
                  min_box_fraction=0.10, max_box_fraction=0.95):
    """Return (mask, method_name) for the tumor inside `box`.

    A mask covering less than `min_box_fraction` of the box, or almost
    all of it (more than `max_box_fraction`, i.e. nothing was actually
    separated), is treated as a failed segmentation and replaced by the
    box itself, and the returned method name says so.
    """
    box_area = max(1, (box[2] - box[0]) * (box[3] - box[1]))

    candidates = []
    if sam_model is not None:
        try:
            candidates.append(
                ("SAM", segment_with_sam(sam_model, image_rgb, box))
            )
        except Exception:
            pass
    candidates.append(
        ("GrabCut", segment_with_grabcut(image_rgb, box, brain_mask))
    )

    for name, mask in candidates:
        if mask is None:
            continue
        fraction = mask.sum() / box_area
        if min_box_fraction <= fraction <= max_box_fraction:
            return mask, name

    return box_to_mask(image_rgb.shape, box), "Bounding box (fallback)"


def overlay_mask(image_rgb, mask, color=(255, 0, 0), alpha=0.4):
    """Semi-transparent mask fill plus contour, returned as an RGB image."""
    out = image_rgb.copy()
    fill = np.zeros_like(out)
    fill[mask] = color
    blended = cv2.addWeighted(out, 1 - alpha, fill, alpha, 0)
    out[mask] = blended[mask]

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(out, contours, -1, color, 2)
    return out
