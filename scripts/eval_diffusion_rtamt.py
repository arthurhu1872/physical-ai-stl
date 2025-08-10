import torch

from src.physical_ai_stl.monitoring.rtamt_monitor import stl_always_upper_bound, evaluate_series


def main() -> None:
    """Evaluate the trained diffusion model using RTAMT to compute STL robustness."""
    ckpt = torch.load("results/diffusion_week1.pt", map_location="cpu")
    u = ckpt["u"]  # shape [n_x, n_t]
    u_max = float(ckpt["u_max"])

    # Build STL specification: always(u <= u_max)
    spec = stl_always_upper_bound(var="u", u_max=u_max)

    robustness_values = []
    for i in range(u.shape[0]):
        series = {"u": u[i].tolist()}
        rob = evaluate_series(spec, series)
        robustness_values.append(rob)

    import numpy as np

    rob_arr = np.array(robustness_values, dtype=float)
    print(
        f"RTAMT robustness: mean={rob_arr.mean():.4f}, min={rob_arr.min():.4f}, max={rob_arr.max():.4f}"
    )


if __name__ == "__main__":
    main()
