"""Download satellite imagery tiles from Esri, Google, and Esri Wayback Machine."""
import io
import math
import urllib.request
from PIL import Image


def _fetch(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def latlon_to_tile(lat, lon, z):
    """Convert lat/lon to web-mercator tile x,y at zoom z."""
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    lat_r = math.radians(lat)
    y = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
    return x, y


def gsd(lat, z):
    """Ground sample distance (meters/pixel) at latitude and zoom."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** z)


def esri_tile(z, x, y):
    url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    return Image.open(io.BytesIO(_fetch(url))).convert("RGB")


def google_tile(z, x, y, server="mt1"):
    """Google Earth satellite tiles (free, up to z20 ~0.15m in some areas)."""
    url = f"https://{server}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
    return Image.open(io.BytesIO(_fetch(url))).convert("RGB")


def wayback_tile(m_number, z, x, y):
    url = (f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/"
           f"world_imagery/mapserver/tile/{m_number}/{z}/{y}/{x}")
    return Image.open(io.BytesIO(_fetch(url))).convert("RGB")


def grid(provider, lat, lon, z, nx=3, ny=3, wayback_m=None, google_server="mt1"):
    """Download an nx*ny tile grid centered near lat/lon. Returns PIL Image."""
    x0, y0 = latlon_to_tile(lat, lon, z)
    tiles = []
    for dy in range(-(ny // 2), ny // 2 + 1):
        for dx in range(-(nx // 2), nx // 2 + 1):
            x, y = x0 + dx, y0 + dy
            if provider == "esri":
                tiles.append(esri_tile(z, x, y))
            elif provider == "google":
                tiles.append(google_tile(z, x, y, google_server))
            elif provider == "wayback":
                assert wayback_m is not None
                tiles.append(wayback_tile(wayback_m, z, x, y))
            else:
                raise ValueError(provider)
    w = Image.new("RGB", (nx * 256, ny * 256))
    for i, im in enumerate(tiles):
        w.paste(im, ((i % nx) * 256, (i // nx) * 256))
    return w


def wayback_versions(json_path="wayback_versions.json"):
    """Load Wayback version list (download once from the REST endpoint)."""
    import json
    with open(json_path) as f:
        return json.load(f)
