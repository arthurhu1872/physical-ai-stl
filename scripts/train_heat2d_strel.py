import numpy as np
import torch
from tqdm import trange
from pathlib import Path

from src.physical_ai_stl.models.mlp import MLP
from src.physical_ai_stl.training.grids import grid2d
from src.physical_ai_stl.physics.heat2d import heat2d_residual, boundary_loss_2d
from src.physical_ai_stl.monitoring.moonlight_helper import (
    load_script_from_file,
    get_monitor,
    build_grid_graph,
    field_to_signal,
)


def compute_gradmag(u: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> np.ndarray:
    """Compute gradient magnitude of a 2D field u(x, y, t) using central differences."""
    u_np = u.detach().cpu().numpy()
    n_x, n_y, n_t = u_np.shape
    grads = np.zeros_like(u_np)
    # spatial step sizes (assume uniform grid)
    dx = float(X[1, 0, 0] - X[0, 0, 0])
    dy = float(Y[0, 1, 0] - Y[0, 0, 0])
    for t in range(n_t):
        ux = np.zeros((n_x, n_y))
        uy = np.zeros((n_x, n_y))
        # central differences inside domain
        ux[1:-1, :] = (u_np[2:, :, t] - u_np[:-2, :, t]) / (2 * dx)
        uy[:, 1:-1] = (u_np[:, 2:, t] - u_np[:, :-2, t]) / (2 * dy)
        grads[:, :, t] = np.sqrt(ux ** 2 + uy ** 2)
    return grads


def main() -> None:
    """Train a 2D heat PINN and validate spatio-temporal STL properties with MoonLight."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # Grid and model configuration
    n_x, n_y, n_t = 64, 64, 51
    X, Y, T, _ = grid2d(n_x=n_x, n_y=n_y, n_t=n_t, device=device)
    model = MLP(in_dim=3, out_dim=1, hidden=(128, 128, 128)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train PINN without STL loss (physics-based only)
    for epoch in trange(1500, desc="Training Heat2D"):
        optimizer.zero_grad()
        res, _ = heat2d_residual(model, X, Y, T, alpha=0.1)
        loss_pde = (res ** 2).mean()
        loss_bc = boundary_loss_2d(model, device=device)
        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(
                f"epoch {epoch:04d} | total={loss.item():.4e} | pde={loss_pde.item():.4e} | bc={loss_bc.item():.4e}"
            )

    # Compute field after training
    with torch.no_grad():
        _, u_field = heat2d_residual(model, X, Y, T, alpha=0.1)
        u = u_field.detach().cpu()  # shape [n_x, n_y, n_t]

    # Save results
    Path("results").mkdir(exist_ok=True, parents=True)
    torch.save({"u": u, "X": X.cpu(), "Y": Y.cpu(), "T": T.cpu()}, "results/heat2d_week2.pt")

    # Build graph for MoonLight (4-neighbour grid)
    graph, graph_times = build_grid_graph(n_x, n_y, weight=1.0)

    # Prepare signals for STREL monitors
    u_np = u.numpy()
    hot_threshold = 0.4
    hot_signal = field_to_signal(u_np, threshold=hot_threshold)
    grads = compute_gradmag(u, X, Y)
    grad_signal = field_to_signal(grads, threshold=None)

    # Load STREL script and monitors
    script = load_script_from_file("src/physical_ai_stl/specs/scripts/contain_hotspot.mls")
    mon_contain = get_monitor(script, "ContainHotspot")
    mon_smooth = get_monitor(script, "SmoothGrad")

    # Time indices for signals
    times = [float(t) for t in range(u_np.shape[-1])]

    # Evaluate containment and smoothness robustness
    D = 12.0  # distance threshold
    gmax = 8.0  # gradient bound
    contain_results = mon_contain.monitor(graph_times, graph, times, hot_signal, D)
    smooth_results = mon_smooth.monitor(graph_times, graph, times, grad_signal, gmax)

    contain_vals = np.array([res[1] for res in contain_results], dtype=float)
    smooth_vals = np.array([res[1] for res in smooth_results], dtype=float)

    print(
        f"[MoonLight] Containment robustness (min over time): {contain_vals.min():.4f}"
    )
    print(
        f"[MoonLight] Smoothness robustness (min over time):  {smooth_vals.min():.4f}"
    )


if __name__ == "__main__":
    main()
