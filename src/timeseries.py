"""Time-series differential: track tank interior brightness across Wayback versions.

Core idea: absolute fill % from one image is fragile, but *changes* in interior
brightness over years are robust. White -> rust -> dark = emptied.
"""
import numpy as np
from .geometry import disk_mean


def series(path_fmt, version_ids, cx, cy, r_in_px, ref_cx, ref_cy, ref_r_px):
    """Measure interior and ground-reference brightness for each version.

    path_fmt: e.g. "wb17_{vid}.png" with {vid} placeholder.
    Returns list of dicts: vid, interior, ref, ratio.
    """
    from PIL import Image
    from .geometry import to_gray
    rows = []
    for vid in version_ids:
        img = Image.open(path_fmt.format(vid=vid)).convert("RGB")
        g = to_gray(img)
        interior = disk_mean(g, cx, cy, r_in_px)
        ref = disk_mean(g, ref_cx, ref_cy, ref_r_px)
        rows.append({"vid": vid, "interior": interior, "ref": ref,
                     "ratio": interior / ref if ref else float("nan")})
    return rows


def classify(interior):
    """Coarse state from interior brightness (tune thresholds per sensor)."""
    if interior >= 150:
        return "full/white"
    if interior >= 90:
        return "rust/transition"
    return "dark/empty"


def print_series(rows):
    for r in rows:
        print(f"{r['vid']:>12}: interior={r['interior']:6.1f} "
              f"ref={r['ref']:6.1f} -> {classify(r['interior'])}")
