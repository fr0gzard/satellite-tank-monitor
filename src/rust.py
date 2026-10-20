"""Rust flagging: HSV orange/brown filter for tank interiors.

White -> rust -> dark is the emptying signature. This module quantifies the
"rustiness" of a tank interior so time-series code can auto-flag transitions.
"""
import cv2
import numpy as np


# HSV range for rust/oxidized steel (tune per sensor):
# H: orange-brown ~ 5..25 (OpenCV H is 0..180)
RUST_LOWER = np.array([5, 60, 40])
RUST_UPPER = np.array([25, 255, 220])


def rust_mask(img_bgr, lower=RUST_LOWER, upper=RUST_UPPER):
    """Binary mask of rust-colored pixels."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)


def rust_fraction(img_bgr, cx, cy, r_px, r_frac=0.85,
                  lower=RUST_LOWER, upper=RUST_UPPER):
    """Fraction of interior disk pixels classified as rust.

    r_frac: use inner 85% to avoid rim/wall contamination.
    """
    h, w = img_bgr.shape[:2]
    mask = rust_mask(img_bgr, lower, upper)
    yy, xx = np.mgrid[0:h, 0:w]
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r_frac * r_px) ** 2
    n = int(disk.sum())
    if n == 0:
        return 0.0
    return float((mask[disk] > 0).sum() / n)


def brightness(img_bgr, cx, cy, r_px, r_frac=0.85):
    """Mean grayscale interior brightness (white=full, dark=empty)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float)
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r_frac * r_px) ** 2
    return float(gray[disk].mean()) if disk.sum() else float("nan")


def state(rust_frac, bright, rust_thresh=0.15, white_thresh=135):
    """Coarse state label."""
    if rust_frac >= rust_thresh:
        return "rust/transition"
    if bright >= white_thresh:
        return "full/white"
    return "dark/empty"
