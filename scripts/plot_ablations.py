"""Plot robustness results from diffusion ablation runs."""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt

def _load_csv(path: str):
    xs, ys = [], []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            xs.append(float(row[0]))
            ys.append(float(row[1]))
    return xs, ys

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/ablations_diffusion.csv")
    ap.add_argument("--out", type=str, default="figs/ablations_diffusion.png")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    x, y = _load_csv(args.csv)
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("λ (STL weight)")
    plt.ylabel("Robustness  G(mean_x u ≤ u_max)")
    plt.title("Diffusion 1D: STL Weight Sweep")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
