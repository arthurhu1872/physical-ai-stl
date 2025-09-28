from __future__ import annotations

import os
import random
import warnings
from typing import Any

__all__ = ["seed_everything", "seed_worker", "torch_generator"]


def _set_if_unset(env: str, value: str) -> None:
    if os.environ.get(env) is None:
        os.environ[env] = value


def seed_everything(
    seed: int = 0,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
    set_pythonhashseed: bool = True,
    configure_cuda_env: bool = True,
    disable_tf32_when_deterministic: bool = True,
    verbose: bool = False,
) -> None:
    # --- Python stdlib RNG ----------------------------------------------------
    if set_pythonhashseed:
        # Must be a string
        os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    # --- NumPy ----------------------------------------------------------------
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        # Keep going even if NumPy is unavailable.
        pass

    # --- PyTorch (optional) ---------------------------------------------------
    try:
        import torch  # type: ignore
    except Exception:
        return

    try:
        torch.manual_seed(seed)
        # If CUDA is built/available, also seed all CUDA devices.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # If something very old/broken, don't hard fail.
        if verbose:
            warnings.warn("Failed to set PyTorch seeds.", RuntimeWarning)

    # Determinism/performance toggles
    try:
        if deterministic:
            # Set environment knobs before CUDA context is created if possible.
            if configure_cuda_env and "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
                _set_if_unset("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                try:
                    # If CUDA context was already created, cuBLAS may ignore this.
                    if torch.cuda.is_available() and torch.cuda.is_initialized() and verbose:
                        warnings.warn(
                            "CUBLAS_WORKSPACE_CONFIG was set after CUDA initialization; "
                            "full determinism may not be guaranteed. Set it early in "
                            "your program for stricter guarantees.",
                            RuntimeWarning,
                        )
                except Exception:
                    pass

            # Prefer official switch when available.
            try:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)  # type: ignore[call-arg]
            except TypeError:
                # For older PyTorch that lacks warn_only.
                torch.use_deterministic_algorithms(True)  # type: ignore[misc]
            except Exception:
                # Ignore if not supported.
                pass

            # cuDNN knobs
            try:
                torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            except Exception:
                pass

            if disable_tf32_when_deterministic:
                # Disable TF32 on Ampere+ for fully consistent FP32 matmuls/convs.
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]
                except Exception:
                    pass
                try:
                    torch.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]
                except Exception:
                    pass
        else:
            # Performance-oriented path.
            try:
                torch.use_deterministic_algorithms(False)  # type: ignore[misc]
            except Exception:
                pass
            try:
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        if verbose:
            warnings.warn("Failed to configure PyTorch determinism/performance flags.", RuntimeWarning)


def seed_worker(worker_id: int) -> None:  # pragma: no cover - utility for DataLoader
    try:
        import numpy as np  # type: ignore
        import torch  # type: ignore
    except Exception:
        # Nothing to do if torch/numpy are unavailable.
        return

    # Derive a 32-bit seed from PyTorch's worker seed.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    os.environ["PYTHONHASHSEED"] = str(worker_seed)


def torch_generator(seed: int, device: Any | None = None):
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover - only used when torch is installed
        raise RuntimeError("torch is required for torch_generator") from e

    gen = torch.Generator(device=device) if device is not None else torch.Generator()
    gen.manual_seed(seed)
    return gen
