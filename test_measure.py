"""Smoke check for measure.py improvements. Run: python test_measure.py"""
import sys
sys.path.insert(0, "src")
import numpy as np
import cv2
from measure import _grad_edge, verify_center, measure_fill


def test_grad_edge_finds_synthetic_ramp():
    # sigmoid ramp from 250 to 60 centered at r=80 (like a real shadow ramp)
    rs = np.arange(20, 140, 1.0)
    prof = 60 + (250 - 60) / (1 + np.exp((rs - 80) / 6.0))
    e = _grad_edge(rs, prof, 0.1, 0.9, falling=True)
    assert e is not None and abs(e - 80) < 3, e


def test_verify_center_prefers_true_circle():
    # bright disk r=120 on dark noisy bg; seed overly small
    rng = np.random.default_rng(0)
    img = np.full((500, 500, 3), 70, np.uint8)
    cv2.circle(img, (250, 250), 120, (240, 240, 240), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img += rng.integers(0, 12, img.shape, dtype=np.uint8)
    cx, cy, r, cov = verify_center(img, 250, 260, 90)
    assert abs(r - 120) <= 4, (cx, cy, r)
    assert abs(cx - 250) <= 4 and abs(cy - 250) <= 4, (cx, cy)
    assert cov > 0.8, cov


def test_full_tank_fails_safe():
    # uniform bright frame: no band -> note says detected nothing, fill None
    img = np.zeros((300, 300, 3), np.uint8)
    cv2.circle(img, (150, 150), 80, (240, 240, 240), -1)
    res = measure_fill(img, 150, 150, 80)
    assert res["fill"] is None
    assert "no inner band" in res["note"] or "unverified" in res["note"], res["note"]


if __name__ == "__main__":
    test_grad_edge_finds_synthetic_ramp()
    test_verify_center_prefers_true_circle()
    test_full_tank_fails_safe()
    print("3 checks OK")
