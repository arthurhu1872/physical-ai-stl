"""CLI wrapper for the SpaTiaL minimal demo."""
import argparse, json, os, sys
from physical_ai_stl.monitors.spatial_demo import run_demo

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--T', type=int, default=50, help='number of time steps')
    ap.add_argument('--out', type=str, default='artifacts/spatial_demo.json')
    args = ap.parse_args()
    val = run_demo(T=args.T)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'robustness': float(val)}, f, indent=2)
    print(f"SpaTiaL robustness saved to {args.out}: {val:.6f}")

if __name__ == '__main__':
    main()
