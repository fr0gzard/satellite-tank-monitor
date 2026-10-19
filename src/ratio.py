"""Ratio method: fill = 1 - L_inner / L_outer.

Geometry (sun from azimuth S):
- Outer shadow (ground, anti-sun side): L_outer = H_wall / tan(alpha)
- Inner shadow (deck, sun side):        L_inner = h_drop  / tan(alpha)
- Ratio L_inner/L_outer = h_drop/H_wall  (sun angle cancels!)
- Fill fraction = 1 - h_drop/H_wall = 1 - L_inner/L_outer

Conventions: angles in degrees, 0=E, 90=S (image coords, y down).
Inner band is on the SUN side; outer shadow on the ANTI-SUN side.
"""
import numpy as np
from .geometry import wedge_profile


def measure_outer(img_gray, cx, cy, r_px, shadow_deg, gsd,
                  f_max=1.8, thresh=12):
    """Length of ground shadow outward from rim (meters)."""
    rf, p = wedge_profile(img_gray, cx, cy, r_px, shadow_deg, 0.97, f_max)
    bg = np.median(p[(rf > f_max - 0.2) & (rf < f_max)])
    i0 = int(np.argmax(rf > 1.0))
    k = i0
    while k < len(p) and p[k] < bg - thresh:
        k += 1
    if k <= i0:
        return 0.0, float(bg)
    return float((rf[k - 1] - 1.0) * r_px * gsd), float(bg)


def detect_inner_band(img_gray, cx, cy, r_px, sun_deg, gsd,
                      f_lo=0.70, f_hi=1.0, sigma_k=2.5):
    """Width of inner wall-shadow band on deck, sun side (meters).

    Returns (L_inner_m, found_bool). If no band detected, L_inner is an
    upper bound at the detection limit (~2px).
    """
    rf, p = wedge_profile(img_gray, cx, cy, r_px, sun_deg, f_lo, 1.04)
    deck = p[(rf > f_lo) & (rf < 0.85)]
    base = np.median(deck)
    sd = np.median(np.abs(deck - base)) * 1.4826 + 1e-6
    zone = (rf > 0.84) & (rf < 1.0)
    dark = (p < base - sigma_k * sd) & zone
    if not dark.any():
        return 2 * gsd, False  # detection limit
    idx = np.where(dark)[0]
    # longest rim-adjacent run
    k, run = idx[-1], 1
    while k - 1 in idx:
        k -= 1
        run += 1
    step = rf[1] - rf[0]
    return float(run * step * r_px * gsd), True


def fill_fraction(l_inner_m, l_outer_m):
    """fill = 1 - L_inner/L_outer. Returns None if inputs invalid."""
    if l_outer_m <= 0 or l_inner_m < 0:
        return None
    return max(0.0, 1.0 - l_inner_m / l_outer_m)


def fill_lower_bound(img_gray, cx, cy, r_px, sun_deg, shadow_deg, gsd):
    """Conservative lower bound on fill when inner band is below detection."""
    l_out, _ = measure_outer(img_gray, cx, cy, r_px, shadow_deg, gsd)
    l_in, found = detect_inner_band(img_gray, cx, cy, r_px, sun_deg, gsd)
    if found:
        return fill_fraction(l_in, l_out), l_in, l_out
    # not found -> upper bound on emptiness
    return (1 - l_in / l_out) if l_out > 0 else None, l_in, l_out
