"""Example: time-series change detection on Cilacap's giant tanks.

Downloads Esri Wayback versions, measures interior brightness, classifies state.
"""
import sys
sys.path.insert(0, "src")
from tiles import grid, gsd, latlon_to_tile
from timeseries import series, print_series

# Cilacap giant tanks (Pertamina RU IV), z17
LAT, LON, Z = -7.69941, 108.99855, 17
# (cx, cy) in a 2x2 z17 grid, radius in px — calibrate visually per tank!
TANKS = {
    "top":    {"cx": 472, "cy": 282, "r": 54},
    "middle": {"cx": 462, "cy": 392, "r": 54},
    "bottom": {"cx": 470, "cy": 486, "r": 54},  # D_white, always full
}
REF = {"cx": 600, "cy": 600, "r": 20}  # bare ground reference
VERSIONS = {  # Wayback version ID -> M number (from the REST endpoint)
    "WB_2018_R08": 8249,
    "WB_2020_R08": 11135,
    "WB_2022_R08": 4905,
    "WB_2024_R06": 12428,
    "WB_2026_R09": 63116,
}

if __name__ == "__main__":
    import os
    os.makedirs("wb_cache", exist_ok=True)
    x0, y0 = latlon_to_tile(LAT, LON, Z)
    for vid, m in VERSIONS.items():
        p = f"wb_cache/cil_{vid}.png"
        try:
            open(p, "rb").close()
            print(vid, "cached")
        except FileNotFoundError:
            print(vid, "downloading...")
            grid("wayback", LAT, LON, Z, 2, 2, wayback_m=m).save(p)
    for name, t in TANKS.items():
        print(f"\n=== {name} ===")
        rows = series("wb_cache/cil_{vid}.png", list(VERSIONS),
                      t["cx"], t["cy"], t["r"],
                      REF["cx"], REF["cy"], REF["r"])
        print_series(rows)
