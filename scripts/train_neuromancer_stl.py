"""Train a tiny Neuromancer model with an STL-style bound constraint.

Example:
    python scripts/train_neuromancer_stl.py --epochs 200 --bound 0.8 --lr 1e-3
"""
import argparse
import json
import os
import sys

from physical_ai_stl.frameworks.neuromancer_stl_demo import DemoConfig, train_demo
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--bound', type=float, default=0.8)
    ap.add_argument('--weight', type=float, default=100.0)
    ap.add_argument('--out', type=str, default='artifacts/neuromancer_stl_results.json')
    args = ap.parse_args()
    cfg = DemoConfig(epochs=args.epochs, lr=args.lr, bound=args.bound, weight=args.weight)
    results = train_demo(cfg)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {args.out}\n{json.dumps(results, indent=2)}")

if __name__ == '__main__':
    main()