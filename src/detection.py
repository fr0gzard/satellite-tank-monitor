"""Automatic tank detection: white segmentation + contour circularity.

Why not plain Otsu? A global Otsu threshold lumps bright tanks together with
bright ground/grass (58% of a typical scene) into one giant blob. White paint
is distinguished by COLOR, not just brightness -> HSV (low saturation, high
value) isolates white surfaces from bright-but-colored ground.

LIMITATION (honest): fully automatic detection on 0.3m imagery is hard.
Tank decks have texture, shadows, and weathering; white regions fragment or
merge with roads/roofs, producing low-circularity blobs. In practice the
semi-automatic workflow (rough center from visual inspection -> refine_rim())
is more reliable than pure auto-detection. This module is a starting point,
not a solved problem.

Pipeline: HSV white mask -> morphological clean -> contours ->
filter by area + circularity + solidity -> min enclosing circle.
"""
import cv2
import numpy as np


def white_mask(img_bgr, sat_max=55, val_min=165):
    """Mask of white/low-saturation bright surfaces (painted tanks, roofs)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, val_min])
    upper = np.array([180, sat_max, 255])
    return cv2.inRange(hsv, lower, upper)


def _clean(mask, ksize=5):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def _circles_from_mask(mask, min_radius_px, max_radius_px,
                       min_circularity, min_solidity=0.80):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-6:
            continue
        circ = 4 * np.pi * area / (peri * peri)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        if not (min_radius_px <= r <= max_radius_px):
            continue
        if circ < min_circularity:
            continue
        hull = cv2.convexHull(cnt)
        solidity = area / (cv2.contourArea(hull) + 1e-6)
        if solidity < min_solidity:
            continue
        out.append((float(cx), float(cy), float(r), float(circ)))
    return out


def detect_tanks(img_bgr, min_radius_px=15, max_radius_px=400,
                 min_circularity=0.60, sat_max=55, val_min=165):
    """Detect white tanks. Returns sorted list of (cx, cy, r, circularity)."""
    mask = _clean(white_mask(img_bgr, sat_max, val_min))
    out = _circles_from_mask(mask, min_radius_px, max_radius_px,
                             min_circularity)
    out.sort(key=lambda t: -t[2])
    return out


def draw_detections(img_bgr, detections, color=(0, 255, 0)):
    out = img_bgr.copy()
    for cx, cy, r, circ in detections:
        cv2.circle(out, (int(cx), int(cy)), int(r), color, 2)
        cv2.circle(out, (int(cx), int(cy)), 3, (0, 0, 255), -1)
        cv2.putText(out, f"{circ:.2f}", (int(cx - r), int(cy - r - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out
