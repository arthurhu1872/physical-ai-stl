# tests/conftest.py — stabilize threading on laptops/CI and make tests deterministic
import os, random
import numpy as np

# Avoid libgomp/OpenBLAS thread explosions on small machines
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Make tests reproducible
random.seed(0)
np.random.seed(0)

try:
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass  # torch is optional for parts of the suite
