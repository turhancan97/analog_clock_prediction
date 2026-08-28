#!/usr/bin/env python3
"""Assemble the browser inference demo: inject the exported model blob and the
example images into docs/demo.template.html's placeholders, writing the
self-contained docs/demo.html that gets published as an Artifact.

    python scripts/export_demo_model.py models/clock_simplecnn.keras \\
        --out docs/demo/model.json
    python scripts/build_demo.py            # docs/demo.template.html -> docs/demo.html

Example images: 4 synthetic test renders + 2 real photos, downscaled and
base64'd. Regenerate them with --refresh-examples.
"""
import argparse
import base64
import glob
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import clockmodel as cm

TEMPLATE = cm.REPO_DIR / "docs" / "demo.template.html"
HTML = cm.REPO_DIR / "docs" / "demo.html"
MODEL_JSON = cm.REPO_DIR / "docs" / "demo" / "model.json"
EXAMPLES_JSON = cm.REPO_DIR / "docs" / "demo" / "examples.json"

SYNTHETIC = ["10-10", "3-45", "7-20", "12-00"]


def _encode(path, box=200, q=78):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((box, box))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def build_examples():
    import pandas as pd
    out = []
    for cls in SYNTHETIC:
        p = sorted(glob.glob(str(cm.DATA_DIR / "test" / cls / "*.jpg")))[0]
        out.append({"kind": "synthetic", "label": cls, "uri": _encode(p)})
    df = pd.read_csv(cm.REAL_MANIFEST)
    df = df[(df["split"] == "test") & (df["source"] == "kongaskristjan")].head(2)
    for r in df.itertuples():
        out.append({"kind": "real", "label": r.label,
                    "uri": _encode(cm.REPO_DIR / r.path)})
    EXAMPLES_JSON.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_JSON.write_text(json.dumps(out))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-examples", action="store_true")
    args = ap.parse_args()

    if not MODEL_JSON.exists():
        sys.exit(f"{MODEL_JSON} missing -- run scripts/export_demo_model.py first")
    examples = (build_examples() if args.refresh_examples or not EXAMPLES_JSON.exists()
                else json.loads(EXAMPLES_JSON.read_text()))

    html = TEMPLATE.read_text()
    html = html.replace("__MODEL_JSON__", MODEL_JSON.read_text().strip())
    html = html.replace("__EXAMPLES_JSON__", json.dumps(examples))
    assert "__MODEL_JSON__" not in html and "__EXAMPLES_JSON__" not in html
    HTML.write_text(html)
    kb = HTML.stat().st_size / 1024
    print(f"built {HTML}  ({kb:.0f} KB, {len(examples)} examples)")


if __name__ == "__main__":
    main()
