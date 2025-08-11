import torch

from physical_ai_stl.monitoring.rtamt_monitor import (
    stl_always_upper_bound,
    evaluate_series,
)

def main() -> None:
    ckpt = torch.load("results/diffusion_week1.pt", map_location="cpu")
    u = ckpt["u"]  # shape [n_x, n_t]
    u_max = float(ckpt["u_max"])

    spec = stl_always_upper_bound(var="u", u_max=u_max)
    series = u.mean(dim=0).tolist()  # mean over x
    rob = evaluate_series(spec, var="u", series=series, dt=1.0)
    print(f"RTAMT robustness (mean_x u <= {u_max}): {rob:.4f}")

if __name__ == "__main__":
    main()
