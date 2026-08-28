#!/usr/bin/env python3
"""Generate the Tock brand assets in docs/branding/ -- four SVG logos
(icon / icon+wordmark, each in a light and a dark variant) and PNG exports.

The mark is a clock dial whose 12 tick marks double as graph nodes (dodecagon
+ inscribed hexagon edges); hands read 10:10, hour in brand blue, minute in
brand red. The wordmark "Tock" is drawn as monoline strokes (no font
dependency), the `o` a full circle in brand blue.

Palette: #3a7eab blue, #cf4832 red, #d1d3d4 grey; ink #23292b / #eef0f0.

    python scripts/generate_brand_assets.py          # SVGs only
    python scripts/generate_brand_assets.py --png     # + PNG exports (needs cairosvg)
"""
import argparse
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "branding"

R = 42
NODES = [(round(60 + R * math.sin(math.radians(i * 30)), 2),
          round(60 - R * math.cos(math.radians(i * 30)), 2)) for i in range(12)]
CARDINAL = {0, 3, 6, 9}          # 12 / 3 / 6 / 9 -- drawn slightly larger


def _hand(length, degrees):
    x = 60 + length * math.sin(math.radians(degrees))
    y = 60 - length * math.cos(math.radians(degrees))
    return f"M60,60 L{x:.2f},{y:.2f}"


HOUR_HAND = _hand(25, 305)       # ~10 o'clock
MINUTE_HAND = _hand(36, 60)      # 10 minutes


def _mark(theme, scale=1.0):
    if theme == "light":
        edge1, edge2, node, cnode, hub = "#b9bdbf", "#d1d3d4", "#8b9296", "#5f676b", "#23292b"
        e1a = e2a = na = 1.0
    else:
        edge1 = edge2 = node = cnode = "#d1d3d4"
        hub = "#eef0f0"
        e1a, e2a, na = 0.45, 0.28, 0.8

    dodeca = "M" + " L".join(f"{x},{y}" for x, y in NODES) + " Z"
    hexa = "M" + " L".join(f"{NODES[i][0]},{NODES[i][1]}" for i in (0, 2, 4, 6, 8, 10)) + " Z"
    ring = "".join(f'<circle cx="{x}" cy="{y}" r="3.9"/>'
                   for i, (x, y) in enumerate(NODES) if i not in CARDINAL)
    cards = "".join(f'<circle cx="{x}" cy="{y}" r="4.6"/>'
                    for i, (x, y) in enumerate(NODES) if i in CARDINAL)

    return f'''  <g transform="translate(60,60) scale({scale}) translate(-60,-60)">
    <g fill="none" stroke-linejoin="round">
      <path d="{dodeca}" stroke="{edge1}" stroke-width="1.8" stroke-opacity="{e1a}"/>
      <path d="{hexa}" stroke="{edge2}" stroke-width="1.5" stroke-opacity="{e2a}"/>
    </g>
    <g fill="{node}" fill-opacity="{na}">{ring}</g>
    <g fill="{cnode}">{cards}</g>
    <path d="{HOUR_HAND}" fill="none" stroke="#3a7eab" stroke-width="7.5" stroke-linecap="round"/>
    <path d="{MINUTE_HAND}" fill="none" stroke="#cf4832" stroke-width="7.5" stroke-linecap="round"/>
    <circle cx="60" cy="60" r="5.2" fill="{hub}"/>
  </g>'''


def _wordmark(theme):
    ink = "#23292b" if theme == "light" else "#eef0f0"
    o = "#3a7eab" if theme == "light" else "#4f97c4"
    return f'''  <g transform="translate(148,0)" fill="none" stroke-linecap="round" stroke-linejoin="round" stroke-width="11">
    <path d="M6,30 L60,30 M33,30 L33,96" stroke="{ink}"/>
    <circle cx="94" cy="66" r="21.5" stroke="{o}"/>
    <path d="M162.7,49.65 A21.5,21.5 0 1 0 162.7,82.35" stroke="{ink}"/>
    <path d="M186,25 L186,96 M186,66 L211,43 M186,66 L214,96" stroke="{ink}"/>
  </g>'''


def _svg(width, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 120" '
            f'width="{width}" height="120" role="img" aria-label="Tock">\n'
            f'  <title>Tock</title>\n{body}\n</svg>\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true", help="also write PNG exports (needs cairosvg)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    for theme in ("light", "dark"):
        (OUT / f"logo-{theme}.svg").write_text(_svg(120, _mark(theme)))
        (OUT / f"logo-{theme}-text.svg").write_text(
            _svg(392, _mark(theme, scale=0.95) + "\n" + _wordmark(theme)))
    print(f"wrote 4 SVGs to {OUT}")

    if args.png:
        import cairosvg
        exports = [
            ("logo-light", "icon-48.png", 48), ("logo-light", "icon-128.png", 128),
            ("logo-light", "icon-256.png", 256),
            ("logo-light", "logo-light-512.png", 512), ("logo-dark", "logo-dark-512.png", 512),
            ("logo-light-text", "logo-light-text-1200.png", 1200),
            ("logo-dark-text", "logo-dark-text-1200.png", 1200),
        ]
        for src, out, w in exports:
            cairosvg.svg2png(url=str(OUT / f"{src}.svg"), write_to=str(OUT / out),
                             output_width=w)
        print(f"wrote {len(exports)} PNGs")


if __name__ == "__main__":
    main()
