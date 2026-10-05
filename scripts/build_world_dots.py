"""One-off generator: dot-matrix world map data for ui/sky.html.

Reads a local simplified GeoJSON of country polygons, samples land on a regular
lat/lon grid, and emits a compact base36 string of land cells. The output is
embedded directly in ui/sky.html so the background needs no external API.

    python scripts/build_world_dots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SOURCE = Path(r"C:\Users\VIVEK\AppData\Local\Temp\opencode\world.geo.json")
OUT = Path(r"C:\Users\VIVEK\AppData\Local\Temp\opencode\world_dots.txt")

STEP = 2.0            # degrees between dots
LAT_MIN, LAT_MAX = -56.0, 78.0
POLYS: list[list[tuple[float, float]]] = []


def load_polygons() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    for feature in data["features"]:
        geom = feature.get("geometry") or {}
        kind = geom.get("type")
        coords = geom.get("coordinates") or []
        if kind == "Polygon":
            rings = [coords[0]] if coords else []
        elif kind == "MultiPolygon":
            rings = [poly[0] for poly in coords if poly]
        else:
            rings = []
        for ring in rings:
            if len(ring) < 4:
                continue
            xs = [p[0] for p in ring]
            ys = [p[1] for p in ring]
            if max(xs) - min(xs) < 1.0 and max(ys) - min(ys) < 1.0:
                continue  # skip tiny islands
            POLYS.append([(float(p[0]), float(p[1])) for p in ring])


def contains(ring: list[tuple[float, float]], x: float, y: float) -> bool:
    inside = False
    j = len(ring) - 1
    for i in range(len(ring)):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def bbox(ring: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    return min(xs), min(ys), max(xs), max(ys)


def main() -> None:
    load_polygons()
    boxes = [(ring, bbox(ring)) for ring in POLYS]
    print(f"polygons: {len(boxes)}")

    cells: list[tuple[int, int]] = []
    lat = LAT_MIN
    while lat <= LAT_MAX:
        lon = -180.0
        while lon <= 180.0:
            for ring, (x0, y0, x1, y1) in boxes:
                if x0 <= lon <= x1 and y0 <= lat <= y1 and contains(ring, lon, lat):
                    cells.append((int(round(lon / STEP)), int(round(lat / STEP))))
                    break
            lon += STEP
        lat += STEP
    print(f"land cells: {len(cells)}")

    # zigzag + delta + base36
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"

    def enc(value: int) -> str:
        value = (value << 1) ^ (value >> 31)  # zigzag
        if value == 0:
            return "0"
        out = ""
        while value:
            out = digits[value & 31 if False else value % 36] + out
            value //= 36
        return out

    parts: list[str] = []
    px = py = 0
    for x, y in cells:
        parts.append(enc(x - px))
        parts.append(enc(y - py))
        px, py = x, y
    payload = ",".join(parts)
    OUT.write_text(f"{STEP}|{LAT_MIN}|{LAT_MAX}|{payload}", encoding="utf-8")
    print(f"payload chars: {len(payload)}  file bytes: {OUT.stat().st_size}")


if __name__ == "__main__":
    main()
