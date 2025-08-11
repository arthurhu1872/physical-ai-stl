import os
import numpy as np
import torch
from torch import optim
from tqdm import trange

from physical_ai_stl.models.mlp import MLP
from physical_ai_stl.training.grids import grid2d
from physical_ai_stl.physics.heat2d import residual_heat2d, bc_ic_heat2d


def compute_gradmag(u: torch.Tensor) -> np.ndarray:
    """Gradient magnitude of a 2D field (central differences)."""
    u_np = u.detach().cpu().numpy()
    gx = np.zeros_like(u_np)
    gy = np.zeros_like(u_np)
    gx[1:-1, :] = 0.5 * (u_np[2:, :] - u_np[:-2, :])
    gy[:, 1:-1] = 0.5 * (u_np[:, 2:] - u_np[:, :-2])
    return np.sqrt(gx * gx + gy * gy)


def main() -> None:
    os.makedirs("results", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    device = "cpu"
    X, Y, T, XYT = grid2d(n_x=64, n_y=64, n_t=16, device=device)

    model = MLP(in_dim=3, out_dim=1, hidden=(64, 64, 64)).to(device)
    opt = optim.Adam(model.parameters(), lr=2e-3)

    alpha = 0.1
    epochs = 200
    batch = 4096

    for _ in trange(epochs, desc="train-2d-heat"):
        idx = torch.randint(0, XYT.shape[0], (batch,), device=device)
        coords = XYT[idx]
        res = residual_heat2d(model, coords, alpha=alpha)
        loss_pde = res.square().mean()
        loss_bcic = bc_ic_heat2d(model, device=device)
        loss = loss_pde + loss_bcic
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Save a couple frames for plotting/audits
    with torch.no_grad():
        for k in [0, T.shape[-1] // 2, T.shape[-1] - 1]:
            inp = torch.stack(
                [X[:, :, k].reshape(-1), Y[:, :, k].reshape(-1), T[:, :, k].reshape(-1)], dim=-1
            )
            u = model(inp).reshape(X.shape[0], X.shape[1])
            np.save(f"results/heat2d_t{k}.npy", u.numpy())

            # Simple figure: gradient magnitude
            import matplotlib.pyplot as plt

            gradmag = compute_gradmag(u)
            plt.figure()
            plt.imshow(gradmag, origin="lower")
            plt.colorbar(label="|∇u|")
            plt.title(f"2D Heat |∇u|, frame {k}")
            plt.savefig(f"figs/heat2d_gradmag_t{k}.png", dpi=150)
            plt.close()

    print("Saved frames to results/ and figs/")


if __name__ == "__main__":
    main()
