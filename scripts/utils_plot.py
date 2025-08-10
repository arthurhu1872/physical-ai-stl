import matplotlib.pyplot as plt
import torch


def plot_u_1d(u: torch.Tensor, X: torch.Tensor, T: torch.Tensor, out: str = "results/diffusion_frames.png") -> None:
    """Plot 1D diffusion field u(x,t) as a heatmap and save to file."""
    u_np = u.numpy()
    X_np = X.numpy()
    T_np = T.numpy()
    plt.figure()
    plt.imshow(
        u_np,
        aspect="auto",
        origin="lower",
        extent=[T_np.min(), T_np.max(), X_np.min(), X_np.max()],
    )
    plt.colorbar(label="u(x,t)")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title("1D Diffusion PINN (u)")
    plt.savefig(out, dpi=150)
    plt.close()


def plot_u_2d_frame(u_frame: torch.Tensor, out: str = "results/heat2d_t0.png") -> None:
    """Plot a single frame of a 2D heat field and save to file."""
    u_np = u_frame.numpy()
    plt.figure()
    plt.imshow(u_np, origin="lower")
    plt.colorbar(label="u(x,y)")
    plt.title("2D Heat (t=0 frame)")
    plt.savefig(out, dpi=150)
    plt.close()
