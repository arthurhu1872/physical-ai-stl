# scripts/train_burgers_torchphysics.py
"""Train 1D Burgers' equation PINN using TorchPhysics, with optional STL penalty."""
import argparse, math, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000, help="Number of training steps.")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    ap.add_argument("--nu", type=float, default=0.01, help="Viscosity in Burgers' equation.")
    ap.add_argument("--lambda-stl", type=float, default=0.0, help="Weight for STL penalty (0 disables).")
    ap.add_argument("--results", type=str, default="results", help="Output directory for results.")
    ap.add_argument("--tag", type=str, default="run", help="Tag for result files.")
    args = ap.parse_args()
    try:
        import torch, torchphysics as tp, pytorch_lightning as pl
    except ImportError as e:
        print(f"Required package not found: {e}. Please install torch, torchphysics, and pytorch_lightning.")
        sys.exit(1)
    torch.manual_seed(0)
    # Define TorchPhysics spaces and domains (1D space X 1D time)
    X = tp.spaces.R1('x'); T = tp.spaces.R1('t')
    space_domain = tp.domains.Interval(X, 0.0, 1.0)
    time_domain = tp.domains.Interval(T, 0.0, 1.0)
    domain = space_domain * time_domain
    # Define PINN model (FCN with Tanh)
    model = tp.models.FCN(input_space=X*T, output_space=tp.spaces.R1('u'), hidden=(64,64,64,64), activation='tanh')
    nu = float(args.nu)
    # Residual for Burgers: u_t + u u_x - nu u_xx = 0
    def pde_residual(u, x):
        grad_u = tp.utils.grad(u, x)      # shape (N,2): [u_x, u_t]
        u_x = grad_u[:, 0:1]; u_t = grad_u[:, 1:2]
        grad_u_x = tp.utils.grad(u_x, x)   # differentiate u_x again
        u_xx = grad_u_x[:, 0:1]
        return u_t + u * u_x - nu * u_xx
    # Initial condition residual: u(x,0) = sin(pi x)
    def initial_residual(u, xt):
        x_coord = xt[:, 0:1]
        target = torch.sin(math.pi * x_coord)
        return u - target
    # Boundary residuals: u(0,t) = 0 and u(1,t) = 0
    def left_residual(u, xt):  return u  # residual = u since target 0
    def right_residual(u, xt): return u
    # STL constraint: globally (u <= 1.0), approx. by pointwise clamp
    u_max = 1.0
    weight_factor = math.sqrt(args.lambda_stl) if args.lambda_stl > 0.0 else 0.0
    def stl_residual(u, xt):
        violation = (u - u_max).clamp_min(0.0)  # positive excess over 1.0
        return violation * weight_factor if weight_factor > 0.0 else violation
    # Samplers for interior, initial line, and boundaries
    pde_sampler = tp.samplers.GridSampler(domain, n_points=(64, 64)).make_static()
    time0 = tp.domains.Interval(T, 0.0, 0.0)        # t = 0 slice
    initial_line = space_domain * time0
    initial_sampler = tp.samplers.GridSampler(initial_line, n_points=(128, 1)).make_static()
    x0 = tp.domains.Interval(X, 0.0, 0.0); x1 = tp.domains.Interval(X, 1.0, 1.0)
    left_edge = x0 * time_domain; right_edge = x1 * time_domain
    left_sampler = tp.samplers.GridSampler(left_edge, n_points=(1, 64)).make_static()
    right_sampler = tp.samplers.GridSampler(right_edge, n_points=(1, 64)).make_static()
    stl_sampler = None
    if args.lambda_stl > 0.0:
        stl_sampler = tp.samplers.GridSampler(domain, n_points=(32, 32)).make_static()
    # Define TorchPhysics training conditions
    pde_cond = tp.conditions.PINNCondition(module=model, sampler=pde_sampler, residual_fn=pde_residual)
    init_cond = tp.conditions.PINNCondition(module=model, sampler=initial_sampler, residual_fn=initial_residual)
    left_cond = tp.conditions.PINNCondition(module=model, sampler=left_sampler, residual_fn=left_residual)
    right_cond = tp.conditions.PINNCondition(module=model, sampler=right_sampler, residual_fn=right_residual)
    conditions = [pde_cond, init_cond, left_cond, right_cond]
    if stl_sampler:
        stl_cond = tp.conditions.PINNCondition(module=model, sampler=stl_sampler, residual_fn=stl_residual)
        conditions.append(stl_cond)
    # Set up optimizer and train
    optim = tp.OptimizerSetting(optimizer_class=torch.optim.Adam, lr=args.lr)
    solver = tp.solver.Solver(train_conditions=conditions, optimizer_setting=optim)
    trainer = pl.Trainer(max_steps=args.epochs, gpus=(1 if torch.cuda.is_available() else None),
                          logger=False, enable_checkpointing=False)
    trainer.fit(solver)
    # Save results (solution field and metadata)
    n_x, n_t = 128, 128
    x_vals = torch.linspace(0.0, 1.0, n_x)
    t_vals = torch.linspace(0.0, 1.0, n_t)
    Xg, Tg = torch.meshgrid(x_vals, t_vals, indexing="ij")
    coords = torch.stack([Xg.reshape(-1), Tg.reshape(-1)], dim=1)
    model.eval()
    with torch.no_grad():
        u_pred = model(coords).reshape(n_x, n_t)
    import os; os.makedirs(args.results, exist_ok=True)
    out_path = os.path.join(args.results, f"burgers_{args.tag}.pt")
    torch.save({"u": u_pred.cpu(), "X": x_vals.cpu(), "T": t_vals.cpu(), "u_max": float(u_max)}, out_path)
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
