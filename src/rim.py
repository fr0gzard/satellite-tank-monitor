"""Rim refinement: Canny edges + radial search + least-squares circle fit.

Given a rough (cx, cy, r) from detection, find the true rim:
  1. Canny edge map.
  2. For each angle, pick the strongest edge within r +/- search_px.
  3. Least-squares circle fit (Kasa) on the edge points.
  4. Reject outliers, refit.
"""
import cv2
import numpy as np


def _fit_circle(xs, ys):
    """Kasa least-squares circle fit. Returns (cx, cy, r)."""
    A = np.column_stack([2 * xs, 2 * ys, np.ones_like(xs)])
    b = xs ** 2 + ys ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(c + cx ** 2 + cy ** 2))
    return float(cx), float(cy), r


def refine_rim(img_bgr, cx, cy, r, search_px=12, n_angles=360,
               canny_lo=60, canny_hi=150, outlier_sigma=2.0):
    """Refine rim circle. Returns (cx, cy, r, n_points)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_lo, canny_hi)
    H, W = edges.shape
    pts = []
    for a in np.linspace(0, 360, n_angles, endpoint=False):
        th = np.radians(a)
        best, best_d = None, None
        for d in range(-search_px, search_px + 1):
            rr = r + d
            x = int(round(cx + rr * np.cos(th)))
            y = int(round(cy + rr * np.sin(th)))
            if 0 <= x < W and 0 <= y < H and edges[y, x]:
                if best_d is None or abs(d) < abs(best_d):
                    best, best_d = (x, y), d
        if best is not None:
            pts.append(best)
    if len(pts) < 12:
        return cx, cy, r, len(pts)
    pts = np.array(pts, dtype=float)
    # iterative outlier rejection
    for _ in range(3):
        fx, fy, fr = _fit_circle(pts[:, 0], pts[:, 1])
        resid = np.abs(np.hypot(pts[:, 0] - fx, pts[:, 1] - fy) - fr)
        keep = resid < outlier_sigma * (resid.std() + 1e-6) + 2.0
        if keep.sum() < 12:
            break
        pts = pts[keep]
    fx, fy, fr = _fit_circle(pts[:, 0], pts[:, 1])
    return fx, fy, fr, len(pts)
