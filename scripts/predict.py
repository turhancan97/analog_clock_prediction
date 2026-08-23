#!/usr/bin/env python3
"""Read the time off one or more analog clock images."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import clockmodel as cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+", help="paths to clock face images")
    ap.add_argument("--model", default=None)
    ap.add_argument("--top", type=int, default=3, help="how many guesses to show")
    args = ap.parse_args()

    model = cm.load_model(args.model) if args.model else cm.load_model()
    batch = np.stack([cm.load_image(p) for p in args.images])
    probs = model.predict(batch, verbose=0)

    for path, guesses in zip(args.images, cm.decode_predictions(probs, top=args.top)):
        print(f"\n{path}")
        for rank, (_, time, p) in enumerate(guesses, 1):
            bar = "#" * int(round(p * 30))
            print(f"  {rank}. {time:>5}  {p:6.2%}  {bar}")


if __name__ == "__main__":
    main()
