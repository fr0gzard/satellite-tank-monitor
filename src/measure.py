"""Fill measurement: verified center + triangulated shadow ratio.

Pipeline (each stage exists because the naive version failed in testing):

  1. verify_center -- a blind circle fit is NOT trusted. On real imagery
     a fit absorbs the tank's outer ground shadow (circle ends up too big
     and off-center) or locks onto a deck ring (too small). Instead, scan
     (cx, cy, r) around the seed and keep the circle supported by the most
     actual edge pixels (combined color+Canny edges within +-2.5 px).
     Two independent edge sets agreeing on the same circle is the
     acceptance test; always eyeball the rim-pixel overlay before use.

  2. Sun azimuth from the INNER shadow band: for each direction, walk
     inward from the rim and measure the dark extent. The band is widest
     toward the sun. (Estimating sun direction from the ground shadow at
     an unverified center gives confidently wrong answers.)

  3. L_inner from a median radial profile across a +-20 deg wedge on the
     sun side: lit deck -> dark band is a sharp edge, and the median over
     the wedge suppresses deck mottling/rust. This is the most reliable
     of the three measurements.

  4. L_outer two ways, because a single ray is fragile -- berms and
     aprons truncate or extend the apparent ground shadow by direction:
       a. ray: median wedge profile on the anti-sun axis, threshold
          crossing at the shadow tip.
       b. area (after PantosThn's area-ratio idea): segment the outer
          shadow as a 2D region (Otsu on the annulus, shadow-half
          components) and invert the stadium area numerically to an
          equivalent shadow length. A 2D integral is far less sensitive
          to local contamination than any 1D walk.

  fill = 1 - L_inner / L_outer.  The result is reported as a RANGE
  across estimators: on real tanks they spread by +-10 points, and a
  single number would be false precision.

Validated on the rusty Cilacap giant (Google z20): verified center
(1390, 1369) R=176, L_inner ~= 100 px by three methods, fill ~= 0.3-0.4
across estimators -- a low tank, consistent with its 2022-2026
darkening in the Wayback time series.
"""
import cv2
import numpy as np

from edges import combined_edges


def verify_center(img_bgr, cx, cy, r, search_c=40, search_r=35, band=2.5):
    """Return the (cx, cy, r) near the seed with maximum rim-pixel support.

    Support = number of combined-edge pixels within +-band px of the
    circle. Coarse scan (4 px center / 2 px radius) then a 1 px refine.
    Returns (cx, cy, r, support) in full-image coordinates.
    """
    m = int(r + search_r + search_c + 10)
    x0, y0 = max(int(cx) - m, 0), max(int(cy) - m, 0)
    crop = img_bgr[y0:int(cy) + m, x0:int(cx) + m]
    edges = combined_edges(crop)
    ys, xs = np.nonzero(edges)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    scx, scy = cx - x0, cy - y0

    def support(ccx, ccy, rr):
        d = np.hypot(pts[:, 0] - ccx, pts[:, 1] - ccy)
        return int((np.abs(d - rr) <= band).sum())

    best, best_s = (scx, scy, r), -1
    for ccx in range(int(scx) - search_c, int(scx) + search_c + 1, 4):
        for ccy in range(int(scy) - search_c, int(scy) + search_c + 1, 4):
            for rr in range(int(r) - search_r, int(r) + search_r + 1, 2):
                s = support(ccx, ccy, rr)
                if s > best_s:
                    best, best_s = (ccx, ccy, rr), s
    (bx, by, br), _ = best, best_s
    for ccx in range(bx - 3, bx + 4):
        for ccy in range(by - 3, by + 4):
            for rr in range(br - 2, br + 3):
                s = support(ccx, ccy, rr)
                if s > best_s:
                    best, best_s = (ccx, ccy, rr), s
    bx, by, br = best
    return bx + x0, by + y0, br, best_s


def _sample(gray, cx, cy, r, adeg):
    a = np.radians(adeg)
    x, y = cx + r * np.cos(a), cy + r * np.sin(a)
    h, w = gray.shape
    if 0 <= x < w and 0 <= y < h:
        return float(cv2.getRectSubPix(gray, (1, 1), (float(x), float(y)))[0, 0])
    return np.nan


def _wedge(gray, cx, cy, center_deg, half_wedge, r_from, r_to, step=1.0):
    rs = np.arange(r_from, r_to, step)
    prof = []
    for r in rs:
        vals = [_sample(gray, cx, cy, r, a)
                for a in np.arange(center_deg - half_wedge,
                                    center_deg + half_wedge + 1, 3)]
        prof.append(np.nanmedian(vals))
    return rs, np.array(prof)


def sun_azimuth(gray, cx, cy, r, dark=60):
    """Sun direction (image degrees: 0=E, 90=S) from inner-band widths."""
    widths = np.zeros(360)
    for adeg in range(0, 360, 5):
        w = 0
        for rr in range(r - 4, 20, -4):
            if _sample(gray, cx, cy, rr, adeg) < dark:
                w += 4
            elif w > 0:
                break
        widths[adeg] = w
    peak = widths.max()
    if peak == 0:
        return None
    strong = np.zeros(360)
    strong[widths >= 0.9 * peak] = 1
    # circular mean of the strong directions (handles the 0/360 wrap)
    ang = np.radians(np.arange(360))
    return float((np.degrees(np.arctan2((strong * np.sin(ang)).sum(),
                                        (strong * np.cos(ang)).sum())) + 360) % 360)


def _crossing_down(rs, prof, thr):
    for i in range(len(rs) - 1):
        if prof[i] > thr >= prof[i + 1]:
            return rs[i] + (prof[i] - thr) / (prof[i] - prof[i + 1]) * (rs[i + 1] - rs[i])
    return None


def _crossing_up_from_outside(rs, prof, thr):
    for i in range(len(rs) - 1, 0, -1):
        if prof[i] > thr >= prof[i - 1]:
            return rs[i] - (prof[i] - thr) / (prof[i] - prof[i - 1]) * (rs[i] - rs[i - 1])
    return None


def _invert_area(target, area_fn, lo=1.0, hi=None):
    hi = hi or 2.0 * 176
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        if area_fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def measure_fill(img_bgr, cx, cy, r, verify=True):
    """Measure fill fraction for the tank seeded at (cx, cy, r).

    Returns a dict with the verified center, sun azimuth, all shadow
    length estimates (px) and the fill estimates per method plus a
    consensus range. Lengths are in pixels; convert with the image's
    m/px -- the ratio itself is scale-free.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    support = None
    if verify:
        cx, cy, r, support = verify_center(img_bgr, cx, cy, r)
    sun = sun_azimuth(gray, cx, cy, r)
    if sun is None:
        return {"cx": cx, "cy": cy, "r": r, "support": support,
                "fill": None, "note": "no inner band detected (tank full?)"}
    anti = (sun + 180) % 360

    # --- L_inner: sun-side wedge profile, sharp deck->band edge
    rs_in, p_in = _wedge(gray, cx, cy, sun, 20, 0.15 * r, r - 6)
    deck = np.median(p_in[rs_in < 0.35 * r])
    band_lvl = np.median(p_in[rs_in > 0.85 * r])
    edge = _crossing_down(rs_in, p_in, (deck + band_lvl) / 2)
    l_inner = float(r - edge) if edge is not None else 0.0

    # --- L_outer (ray): anti-sun wedge profile, outside-in tip crossing
    rs_out, p_out = _wedge(gray, cx, cy, anti, 15, r, 2.1 * r)
    shad = np.median(p_out[(rs_out > 1.2 * r) & (rs_out < 1.6 * r)])
    gnd = np.median(p_out[rs_out > 1.85 * r])
    tip = _crossing_up_from_outside(rs_out, p_out, (shad + gnd) / 2)
    l_outer_ray = float(tip - r) if tip is not None else None

    # --- Areas on a crop (for the area estimator + inversion)
    m = int(2.3 * r)
    x0, y0 = max(int(cx) - m, 0), max(int(cy) - m, 0)
    cg = gray[y0:int(cy) + m, x0:int(cx) + m]
    ccx, ccy = cx - x0, cy - y0
    yy, xx = np.mgrid[0:cg.shape[0], 0:cg.shape[1]]
    dist = np.hypot(xx - ccx, yy - ccy)
    inside = dist <= r - 3
    dark_in = inside & (cg < 60)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        dark_in.astype(np.uint8), 8)
    a_in = 0
    for i in range(1, n):
        if ((lab == i) & (dist > r - 25)).sum() > 50:
            a_in = max(a_in, stats[i, cv2.CC_STAT_AREA])
    ann = (dist > r + 2) & (dist <= 2.0 * r)
    thr_o, _ = cv2.threshold(cg[ann].astype(np.uint8), 0, 255,
                             cv2.THRESH_OTSU)
    dark_out = ann & (cg < thr_o)
    n2, lab2, stats2, cents = cv2.connectedComponentsWithStats(
        dark_out.astype(np.uint8), 8)
    au = np.radians(anti)
    avec = np.array([np.cos(au), np.sin(au)])
    a_out = 0
    for i in range(1, n2):
        if stats2[i, cv2.CC_STAT_AREA] < 200:
            continue
        v = np.array([cents[i][0] - ccx, cents[i][1] - ccy])
        v = v / (np.hypot(*v) + 1e-9)
        if np.dot(v, avec) > 0.3:
            a_out += stats2[i, cv2.CC_STAT_AREA]

    su = np.radians(sun)
    svec = np.array([np.cos(su), np.sin(su)])

    def inner_area(L):
        sh = np.hypot(xx - (ccx + L * svec[0]), yy - (ccy + L * svec[1])) <= r
        return int((inside & ~sh).sum())

    def outer_area(L):
        t = np.clip((xx - ccx) * avec[0] + (yy - ccy) * avec[1], 0, L)
        d = np.hypot(xx - (ccx + t * avec[0]), yy - (ccy + t * avec[1]))
        return int(((d <= r) & (dist > r)).sum())

    l_inner_area = _invert_area(a_in, inner_area, hi=2 * r) if a_in else None
    l_outer_area = _invert_area(a_out, outer_area, hi=2 * r) if a_out else None

    fills = {}
    if l_outer_ray:
        fills["ray"] = 1 - l_inner / l_outer_ray
    if l_outer_area:
        fills["area"] = 1 - l_inner / l_outer_area
    if a_in and a_out:
        fills["area_raw"] = 1 - a_in / a_out
    vals = [v for v in fills.values() if v is not None]
    return {
        "cx": cx, "cy": cy, "r": r, "support": support,
        "sun_deg": sun, "anti_deg": anti,
        "l_inner_px": l_inner,
        "l_outer_ray_px": l_outer_ray,
        "l_outer_area_px": l_outer_area,
        "a_inner": int(a_in), "a_outer": int(a_out),
        "fills": fills,
        "fill_range": (min(vals), max(vals)) if vals else None,
        "fill_consensus": float(np.median(vals)) if vals else None,
    }
