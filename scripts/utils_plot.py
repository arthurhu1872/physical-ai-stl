import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_u_1d(u: torch.Tensor, X: torch.Tensor, T: torch.Tensor, out: str = "figs/diffusion_heatmap.png") -> None:
    """Plot 1D diffusion field u(x,t) as a heatmap and save to file."""
    u_np = u.detach().cpu().numpy()
    X_np = X.detach().cpu().numpy()
    T_np = T.detach().cpu().numpy()
    plt.figure()
    plt.imshow(
        u_np,
        aspect="auto",
        origin="lower",
        extent=[T_np.min(), T_np.max(), X_np.min(), X_np.max()],
    )
    plt.xlabel("t")
    plt.ylabel("x")
    plt.colorbar(label="u(x,t)")
    plt.title("1D Diffusion PINN (u)")
    plt.savefig(out, dpi=150)
    plt.close()

def plot_u_2d_frame(u_frame: torch.Tensor, out: str = "figs/heat2d_t0.png") -> None:
    """Plot a single frame of a 2D heat field and save to file."""
    u_np = u_frame.detach().cpu().numpy()
    plt.figure()
    plt.imshow(u_np, origin="lower")
    plt.colorbar(label="u(x,y)")
    plt.title("2D Heat (t=0 frame)")
    plt.savefig(out, dpi=150)
    plt.close()
