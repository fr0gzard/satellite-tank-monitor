"""Tank geometry: radial/angular profiles, polar unwrapping, visualization."""
import numpy as np
from PIL import Image, ImageDraw


def to_gray(img):
    return np.array(img.convert("L")).astype(float)


def sample(img_gray, cx, cy, r_frac, angle_deg, radius_px):
    """Sample grayscale value at fractional radius and angle (0=E, 90=S)."""
    H, W = img_gray.shape
    th = np.radians(angle_deg)
    x = int(np.clip(round(cx + r_frac * radius_px * np.cos(th)), 0, W - 1))
    y = int(np.clip(round(cy + r_frac * radius_px * np.sin(th)), 0, H - 1))
    return float(img_gray[y, x])


def wedge_profile(img_gray, cx, cy, radius_px, center_deg, f0, f1,
                  half_deg=8, step=0.01):
    """Median brightness vs fractional radius, averaged over angular wedge."""
    ths = np.radians(np.linspace(center_deg - half_deg, center_deg + half_deg, 17))
    H, W = img_gray.shape
    rf = np.arange(f0, f1, step)
    out = []
    for f in rf:
        vals = []
        for th in ths:
            x = int(np.clip(round(cx + f * radius_px * np.cos(th)), 0, W - 1))
            y = int(np.clip(round(cy + f * radius_px * np.sin(th)), 0, H - 1))
            vals.append(img_gray[y, x])
        out.append(float(np.median(vals)))
    return rf, np.array(out)


def angular_profile(img_gray, cx, cy, radius_px, f_in=1.05, f_out=1.30, step_deg=5):
    """Median brightness vs angle in an annulus. Useful for shadow direction."""
    angs = np.arange(0, 360, step_deg)
    out = []
    for a in np.radians(angs):
        vals = []
        for f in np.linspace(f_in, f_out, 10):
            H, W = img_gray.shape
            x = int(np.clip(round(cx + f * radius_px * np.cos(a)), 0, W - 1))
            y = int(np.clip(round(cy + f * radius_px * np.sin(a)), 0, H - 1))
            vals.append(img_gray[y, x])
        out.append(float(np.median(vals)))
    return angs, np.array(out)


def disk_mean(img_gray, cx, cy, r_px):
    """Mean brightness inside a disk (for time-series interior measurement)."""
    yy, xx = np.mgrid[0:img_gray.shape[0], 0:img_gray.shape[1]]
    m = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r_px ** 2
    return float(img_gray[m].mean()) if m.sum() else float("nan")


def polar_unwrap(img_gray, cx, cy, radius_px, f0=0.5, f1=1.6, n_ang=720):
    """Unwrap tank to angle-vs-radius image. x=angle, y=radius."""
    angs = np.linspace(0, 360, n_ang, endpoint=False)
    rf = np.linspace(f0, f1, 200)
    H, W = img_gray.shape
    pol = np.zeros((len(rf), n_ang))
    for j, a in enumerate(np.radians(angs)):
        xs = np.clip((cx + rf * radius_px * np.cos(a)).round().astype(int), 0, W - 1)
        ys = np.clip((cy + rf * radius_px * np.sin(a)).round().astype(int), 0, H - 1)
        pol[:, j] = img_gray[ys, xs]
    return pol


def overlay_rim(img, cx, cy, r_px, color=(255, 0, 0), width=3):
    """Draw rim circle on a PIL image (returns copy)."""
    out = img.copy()
    ImageDraw.Draw(out).ellipse([cx - r_px, cy - r_px, cx + r_px, cy + r_px],
                                outline=color, width=width)
    return out


def overlay_ray(img, cx, cy, r_px, angle_deg, f0=1.0, f1=1.6,
                color=(255, 255, 0), width=3, label=None):
    """Draw a radial ray (for shadow direction illustration)."""
    out = img.copy()
    dr = ImageDraw.Draw(out)
    th = np.radians(angle_deg)
    x1, y1 = cx + f0 * r_px * np.cos(th), cy + f0 * r_px * np.sin(th)
    x2, y2 = cx + f1 * r_px * np.cos(th), cy + f1 * r_px * np.sin(th)
    dr.line([(x1, y1), (x2, y2)], fill=color, width=width)
    if label:
        dr.text((x2 + 4, y2 - 8), label, fill=color)
    return out
