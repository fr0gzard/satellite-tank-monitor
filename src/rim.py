"""Rim refinement v3: gradient-weighted radial search + robust circle fit.

Given a rough (cx, cy, r) from detection, find the true tank wall rim:
  1. Gaussian blur (5x5) + Canny edge map.
  2. For each angle, pick the STRONGEST edge (Sobel gradient magnitude)
     within r +/- search_px, sampled at 0.5 px steps. The wall rim is the
     strongest circular boundary; weaker deck rings lose to it.
  3. Least-squares circle fit (Kasa) on the edge points.
  4. Iteratively reject points with residual > reject_px (default 2 px)
     and refit. This step matters: a naive fit absorbs the tank's OUTER
     GROUND SHADOW, whose edge points sit consistently outside the rim on
     the anti-sun side and drag the circle off-center and oversize.
     Dropping them is what makes the circle hug the wall instead of the
     shadow (validated on Dalian Xingang + Gresik imagery).
"""
import cv2
import numpy as np


def _fit_circle(xs, ys):
    """Kasa least-squares circle fit. Returns (cx, cy, r)."""
    A = np.column_stack([2 * xs, 2 * ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0)))
    return float(cx), float(cy), r


def refine_rim(img_bgr, cx, cy, r, search_px=12, n_angles=360,
               canny_lo=50, canny_hi=130, reject_px=2.0):
    """Refine rim circle. Returns (cx, cy, r, n_points_kept)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_lo, canny_hi)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    H, W = edges.shape
    pts = []
    for a in np.linspace(0, 360, n_angles, endpoint=False):
        th = np.radians(a)
        best_r, best_m = None, -1.0
        for rr in np.arange(r - search_px, r + search_px + 0.01, 0.5):
            x = int(round(cx + rr * np.cos(th)))
            y = int(round(cy + rr * np.sin(th)))
            if 0 <= x < W and 0 <= y < H and edges[y, x] and mag[y, x] > best_m:
                best_m, best_r = mag[y, x], rr
        if best_r is not None:
            pts.append((cx + best_r * np.cos(th), cy + best_r * np.sin(th)))
    if len(pts) < 12:
        return cx, cy, r, len(pts)
    pts = np.array(pts, dtype=float)
    # iterative outlier rejection with a FIXED pixel threshold:
    # shadow points are systematically outside, not random noise, so a
    # sigma-based threshold (v2) lets them survive whenever they dominate.
    for _ in range(3):
        fx, fy, fr = _fit_circle(pts[:, 0], pts[:, 1])
        resid = np.abs(np.hypot(pts[:, 0] - fx, pts[:, 1] - fy) - fr)
        keep = resid <= reject_px
        if keep.sum() < 12 or keep.all():
            break
        pts = pts[keep]
    fx, fy, fr = _fit_circle(pts[:, 0], pts[:, 1])
    return fx, fy, fr, len(pts)
