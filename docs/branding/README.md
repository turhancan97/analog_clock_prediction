# Tock — brand assets

**Tock** — reading analog clocks with a CNN. The name is the tick-**tock** of a
clock: short, spoken, unambiguous.

## The mark

A clock dial whose twelve tick marks double as the nodes of a small graph
(dodecagon + inscribed hexagon edges) — the analog face and the network that
reads it, in one shape. Hands are set to **10:10**: hour hand in brand blue,
minute hand in brand red.

## Files

| file | use |
|---|---|
| `logo-light.svg` / `logo-dark.svg` | icon only, for light / dark backgrounds |
| `logo-light-text.svg` / `logo-dark-text.svg` | icon + `Tock` wordmark, light / dark |
| `icon-48.png` `icon-128.png` `icon-256.png` | raster icon (favicons, avatars) |
| `logo-light-512.png` / `logo-dark-512.png` | raster icon, large |
| `logo-light-text-1200.png` / `logo-dark-text-1200.png` | raster lockup |

SVG is the source of truth; the PNGs are generated from it (transparent
background). Regenerate everything with
`python scripts/generate_brand_assets.py --png` (`--png` needs `cairosvg`).

Use the **light** files on backgrounds lighter than ~#888, the **dark** files
on anything darker. Keep clear space of at least half the icon width around the
logo. Don't recolor the hands, rotate the mark, or set the wordmark in a
different typeface.

## Palette

| swatch | hex | role |
|---|---|---|
| ● | `#3a7eab` | brand blue — hour hand, the `o` in the wordmark |
| ● | `#cf4832` | brand red — minute hand, accents |
| ● | `#d1d3d4` | brand grey — dial edges/nodes on dark, neutral surfaces |
| ● | `#23292b` | ink — wordmark & hub on light backgrounds |
| ● | `#eef0f0` | ink — wordmark & hub on dark backgrounds |

On dark backgrounds the wordmark `o` is lightened to `#4f97c4` for contrast.

## Wordmark

`Tock`, drawn as monoline geometric strokes (no font dependency) in the SVG:
cap `T`, lowercase `ock`, the `o` a full circle in brand blue. Round line caps
and joins throughout, stroke weight matched to the icon's hands.
