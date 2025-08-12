from __future__ import annotations
import glob, os
import numpy as np
from physical_ai_stl.monitoring.moonlight_helper import load_script_from_file, get_monitor, field_to_signal

def main() -> None:
    frames = sorted(glob.glob("results/heat2d_*.npy")) or sorted(glob.glob("results/heat2d_t*.npy"))
    if not frames:
        raise SystemExit("No frames found in results/. Run `python scripts/train_heat2d_strel.py` or the heat2d experiment first.")
    arrs = [np.load(p) for p in frames]
    u = np.stack(arrs, axis=-1)  # (n_x, n_y, n_t)
    threshold = float(u.mean() + 0.5 * u.std())
    mls = load_script_from_file("scripts/specs/contain_hotspot.mls"); mon = get_monitor(mls, "contain")
    signal = field_to_signal(u, threshold=threshold)
    # proxy score: frames with no hotspot
    satisfied = sum(int((np.mean(frame) <= 0.0 + 1e-6)) for frame in signal)
    frac = satisfied / len(signal)
    print(f"Heat2D: fraction of frames with no 'hot' area (proxy): {frac:.3f}")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Skipping MoonLight run: {e}")
