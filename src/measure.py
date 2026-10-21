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

    Support metric = ANGULAR COVERAGE: the fraction of 180 angle bins around
    the circle containing at least one edge pixel within +-band px.
    (Raw pixel count was biased: a bigger circle gets more pixels, and edge
    pixels INSIDE the circle -- deck rings, radial spokes, pipes -- counted
    the same as rim pixels. Coverage only rewards circles that trace a
    complete rim; an off-center or shrunken circle leaves an empty arc.)

    Coarse scan (4 px center / 2 px radius) then a 1 px refine.
    Returns (cx, cy, r, coverage 0..1) in full-image coordinates.
    """
    m = int(r + search_r + search_c + 10)
    x0, y0 = max(int(cx) - m, 0), max(int(cy) - m, 0)
    crop = img_bgr[y0:int(cy) + m, x0:int(cx) + m]
    edges = combined_edges(crop)
    ys, xs = np.nonzero(edges)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    scx, scy = cx - x0, cy - y0
    n_bins = 180
    bin_w = 360.0 / n_bins

    def coverage(ccx, ccy, rr):
        if rr <= 0:
            return 0.0
        d = np.hypot(pts[:, 0] - ccx, pts[:, 1] - ccy)
        sel = np.abs(d - rr) <= band
        if not sel.any():
            return 0.0
        px, py = pts[sel, 0], pts[sel, 1]
        a = np.floor(np.degrees(np.arctan2(py - ccy, px - ccx)) % 360 / bin_w)
        return int(len(np.unique(a))) / n_bins

    best, best_c = (scx, scy, r), -1
    for ccx in range(int(scx) - search_c, int(scx) + search_c + 1, 4):
        for ccy in range(int(scy) - search_c, int(scy) + search_c + 1, 4):
            for rr in range(int(r) - search_r, int(r) + search_r + 1, 2):
                c = coverage(ccx, ccy, rr)
                if c > best_c:
                    best, best_c = (ccx, ccy, rr), c
    (bx, by, br), best_c = best, best_c
    for ccx in range(bx - 3, bx + 4):
        for ccy in range(by - 3, by + 4):
            for rr in range(br - 2, br + 3):
                c = coverage(ccx, ccy, rr)
                if c > best_c:
                    best, best_c = (ccx, ccy, rr), c
    bx, by, br = best
    return bx + x0, by + y0, br, best_c


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


def sun_azimuth(gray, cx, cy, r):
    """Sun direction (image degrees: 0=E, 90=S) from inner-band widths.

    Band threshold is RELATIVE to the deep-interior deck level (MAD-based):
    on bright Google imagery the band reads ~150 gray vs deck ~250, so an
    absolute gray<60 gate missed real bands and aborted the measurement.
    """
    widths = np.zeros(360)
    for adeg in range(0, 360, 3):
        rr = np.arange(r - 2, 0.15 * r, -1.0)
        vals = np.array([_sample(gray, cx, cy, x, adeg) for x in rr])
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            continue
        deep = vals[len(vals) // 3:]
        deck = np.median(deep)
        mad = np.median(np.abs(deep - deck)) * 1.4826 + 1e-6
        thr = deck - max(4.0 * mad, 0.12 * deck)
        dark = vals < thr
        w = 0
        for d in dark:
            if d:
                w += 1
            elif w >= 2:
                break
        widths[adeg] = w
    peak = widths.max()
    # ponytail: narrow-band argmax comes from pipe/deck-line shadows, not
    # the band -- gate so it fails safe as "no band" instead of lying.
    if peak < max(4, 0.05 * r):
        return None
    strong = widths >= 0.85 * peak
    # circular mean of the strong directions (handles the 0/360 wrap)
    ang = np.radians(np.arange(360))
    return float((np.degrees(np.arctan2((strong * np.sin(ang)).sum(),
                                        (strong * np.cos(ang)).sum())) + 360) % 360)


def _grad_edge(rs, prof, lo_frac, hi_frac, falling=True):
    """Subpixel radius of the steepest-gradient edge in a radial profile.

    The lit->shadow transition on real imagery is a ramp, not a step: a
    fixed-threshold midpoint crossing drifts with ramp shape, the
    steepest-gradient point tracks the physical edge. Parabolic
    interpolation gives subpixel precision. lo_frac/hi_frac bound the
    search zone as fractions of the covered radius span.
    """
    rs = np.asarray(rs, float)
    prof = np.asarray(prof, float)
    if len(rs) < 6:
        return None
    lo = rs[0] + lo_frac * (rs[-1] - rs[0])
    hi = rs[0] + hi_frac * (rs[-1] - rs[0])
    zone = (rs >= lo) & (rs <= hi)
    g = np.gradient(prof, rs)
    # smooth so deck speckle doesn't outrank the ramp
    if len(g) > 7:
        k = np.ones(5) / 5.0
        gs = np.convolve(g, k, mode="same")
        gs[:2] = gs[-2:] = 0.0
    else:
        gs = g
    sig = -gs if falling else gs
    sig = np.where(zone, sig, -np.inf)
    i = int(np.argmax(sig))
    if not np.isfinite(sig[i]) or sig[i] <= 0:
        return None
    if 0 < i < len(rs) - 1:                              # parabolic subpixel
        denom = sig[i - 1] - 2 * sig[i] + sig[i + 1]
        if denom != 0 and np.isfinite(sig[i - 1]) and np.isfinite(sig[i + 1]):
            i = i + 0.5 * (sig[i - 1] - sig[i + 1]) / denom
    return float(rs[0] + i * (rs[1] - rs[0]))


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
        # Limit the support scan to a sane band around the seed radius:
        # an unconstrained scan climbs onto berm/dyke edges and returns
        # oversize circles on bermed tank farms.
        cx, cy, r, support = verify_center(
            img_bgr, cx, cy, r, search_c=30, search_r=max(8, int(0.2 * r)))
    sun = sun_azimuth(gray, cx, cy, r)
    if support is not None and support < 0.6:
        return {"cx": cx, "cy": cy, "r": r, "support": support,
                "fill": None,
                "note": f"rim coverage {support:.2f} too low, circle unverified"}
    if sun is None:
        return {"cx": cx, "cy": cy, "r": r, "support": support,
                "fill": None, "note": "no inner band detected (tank full?)"}
    anti = (sun + 180) % 360

    # --- L_inner: sun-side wedge profile, deck->band ramp (gradient edge)
    rs_in, p_in = _wedge(gray, cx, cy, sun, 20, 0.15 * r, r - 6)
    deck = np.median(p_in[rs_in < 0.35 * r])
    band_lvl = np.median(p_in[rs_in > 0.85 * r])
    edge = _grad_edge(rs_in, p_in, 0.35, 0.92, falling=True)
    if edge is None:                                     # fallback: threshold
        edge = _crossing_down(rs_in, p_in, (deck + band_lvl) / 2)
    l_inner = float(r - edge) if edge is not None else 0.0

    # --- L_outer (ray): anti-sun wedge profile, ground->shadow ramp (tip)
    rs_out, p_out = _wedge(gray, cx, cy, anti, 15, r, 2.1 * r)
    shad = np.median(p_out[(rs_out > 1.2 * r) & (rs_out < 1.6 * r)])
    gnd = np.median(p_out[rs_out > 1.85 * r])
    tip = _grad_edge(rs_out, p_out, 0.35, 0.9, falling=False)
    if tip is None:                                      # fallback: threshold
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
    # Sanity gates: a ground shadow shorter than the inner band is
    # physically impossible (means that estimator got contaminated),
    # and a fill outside [0, 1] is a failed estimate, not data. Only
    # gated estimates feed the consensus; raw values stay visible.
    gated = {}
    if l_outer_ray and l_outer_ray >= 0.9 * l_inner:
        gated["ray"] = fills["ray"]
    if l_outer_area and l_outer_area >= 0.9 * l_inner:
        gated["area"] = fills["area"]
    if "area_raw" in fills and 0.0 <= fills["area_raw"] <= 1.0:
        gated["area_raw"] = fills["area_raw"]
    gated = {k: v for k, v in gated.items() if -0.02 <= v <= 1.02}
    vals = list(gated.values())
    return {
        "cx": cx, "cy": cy, "r": r, "support": support,
        "sun_deg": sun, "anti_deg": anti,
        "l_inner_px": l_inner,
        "l_outer_ray_px": l_outer_ray,
        "l_outer_area_px": l_outer_area,
        "a_inner": int(a_in), "a_outer": int(a_out),
        "fills": fills,
        "fills_gated": gated,
        "fill_range": (min(vals), max(vals)) if vals else None,
        "fill_consensus": float(np.median(vals)) if vals else None,
        "note": None if vals else "estimators diverged/contaminated",
    }
