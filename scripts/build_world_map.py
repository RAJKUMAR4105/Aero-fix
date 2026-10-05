"""Normal world-map geometry for ui/sky.html (land polygons, not dots).

Reads the local Natural Earth 110m country polygons (public domain, bundled
source recorded below), simplifies every ring, splits rings that cross the
antimeridian, rounds to 0.1 deg and emits a compact zigzag/base36 payload
which is embedded directly into ui/sky.html (the component iframe cannot fetch
files at runtime, so the geometry ships inside the HTML - no CDN, no network).

Outputs:
  ui/assets/world-land.json   readable bundled asset (source/license metadata)
  ui/sky.html                 __LAND_DATA__ placeholder replaced by the payload

    python scripts/build_world_map.py
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"C:\Users\VIVEK\AppData\Local\Temp\opencode\world.geo.json")
ASSET = PROJECT / "ui" / "assets" / "world-land.json"
SKY = PROJECT / "ui" / "sky.html"

SIMPLIFY_DEG = 0.3       # Douglas-Peucker tolerance (degrees)
MIN_BBOX_DEG = 1.2       # drop islands smaller than this diagonal
SCALE = 10               # 0.1 degree precision -> integers
DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"


# --- geometry helpers ---------------------------------------------------------

def rdp(points: list[tuple[float, float]], eps: float) -> list[tuple[float, float]]:
    """Iterative Douglas-Peucker (keeps the shape, drops redundant vertices)."""
    if len(points) < 3:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        ax, ay = points[start]
        bx, by = points[end]
        dx, dy = bx - ax, by - ay
        norm = (dx * dx + dy * dy) ** 0.5 or 1e-12
        farthest, best = -1, 0.0
        for i in range(start + 1, end):
            px, py = points[i]
            dist = abs(dy * px - dx * py + bx * ay - by * ax) / norm
            if dist > best:
                farthest, best = i, dist
        if farthest > 0 and best > eps:
            keep[farthest] = True
            stack.append((start, farthest))
            stack.append((farthest, end))
    return [p for p, k in zip(points, keep) if k]


def ring_bbox(points: list[tuple[float, float]]) -> float:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5


def split_dateline(points: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    """Break a ring where consecutive longitudes jump by more than 180 deg, so
    no polygon can ever draw a giant horizontal streak across the map."""
    pieces: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = [points[0]]
    for prev, cur in zip(points, points[1:]):
        if abs(cur[0] - prev[0]) > 180:
            pieces.append(current)
            current = [cur]
        else:
            current.append(cur)
    pieces.append(current)
    return [piece for piece in pieces if len(piece) >= 3]


def process_ring(ring: list[tuple[float, float]]) -> list[list[tuple[float, float]]]:
    pts = [(float(p[0]), float(p[1])) for p in ring]
    if len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) < 1e-9 and abs(pts[0][1] - pts[-1][1]) < 1e-9:
        pts = pts[:-1]                              # drop duplicate closing vertex
    if len(pts) < 3:
        return []
    if ring_bbox(pts) < MIN_BBOX_DEG:
        return []                                    # skip tiny islands
    simplified = rdp(pts, SIMPLIFY_DEG)
    if len(simplified) < 3:
        return []
    rounded = [(round(x * SCALE) / SCALE, round(y * SCALE) / SCALE) for x, y in simplified]
    return split_dateline(rounded)


# --- encoding -----------------------------------------------------------------

def enc(value: int) -> str:
    value = (value << 1) ^ (value >> 31)             # zigzag
    if value == 0:
        return "0"
    out = ""
    while value:
        out = DIGITS[value % 36] + out
        value //= 36
    return out


def encode_features(features: list[list[list[tuple[float, float]]]]) -> str:
    """feature = 'ring;ring'  (delta + zigzag + base36), features = 'f|f|...'"""
    encoded_features = []
    for rings in features:
        encoded_rings = []
        for ring in rings:
            cells: list[str] = []
            px = py = 0
            for x, y in ring:
                xi, yi = int(round(x * SCALE)), int(round(y * SCALE))
                cells.append(enc(xi - px))
                cells.append(enc(yi - py))
                px, py = xi, yi
            encoded_rings.append(f"{len(ring)}," + ",".join(cells))
        if encoded_rings:
            encoded_features.append(";".join(encoded_rings))
    return "|".join(encoded_features)


# --- main ---------------------------------------------------------------------

def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    features: list[list[list[tuple[float, float]]]] = []
    rings_in = rings_out = 0
    for feature in data["features"]:
        geom = feature.get("geometry") or {}
        kind = geom.get("type")
        coords = geom.get("coordinates") or []
        if kind == "Polygon":
            raw_rings = coords
        elif kind == "MultiPolygon":
            raw_rings = [ring for poly in coords for ring in poly]
        else:
            raw_rings = []
        rings: list[list[tuple[float, float]]] = []
        for ring in raw_rings:
            rings_in += 1
            for piece in process_ring(ring):
                rings.append(piece)
                rings_out += 1
        if rings:
            features.append(rings)

    payload = encode_features(features)
    points = sum(len(r) for f in features for r in f)
    print(f"features: {len(features)}   rings: {rings_in} -> {rings_out}   points: {points}")
    print(f"payload chars: {len(payload)}")

    # readable bundled asset (source + license recorded for the README)
    asset = {
        "source": "Natural Earth 1:110m physical vectors (via world.geo.json)",
        "license": "Public domain (Natural Earth) - no attribution required",
        "projection": "equirectangular",
        "precision_deg": 1 / SCALE,
        "simplified_deg": SIMPLIFY_DEG,
        "encoding": "per feature 'ring;ring', per ring 'n,' + zigzag-delta base36 pairs of (lon*10, lat*10)",
        "features": [
            {"rings": [[[round(x, 1), round(y, 1)] for x, y in ring] for ring in rings]}
            for rings in features
        ],
    }
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    ASSET.write_text(json.dumps(asset, separators=(",", ":")), encoding="utf-8")
    print(f"asset: {ASSET}  ({ASSET.stat().st_size} bytes)")

    sky = SKY.read_text(encoding="utf-8")
    if "__LAND_DATA__" not in sky:
        if "const LAND_DATA = '" not in sky:
            sys.exit("sky.html has no LAND_DATA placeholder to inject")
        print("payload already injected (no __LAND_DATA__ placeholder left)")
        return
    SKY.write_text(sky.replace("__LAND_DATA__", payload, 1), encoding="utf-8")
    print(f"injected into {SKY}  (new size {SKY.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
