from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _import_error = e  # keep around for a helpful message


if torch is not None:

    class MLP(nn.Module):
        """Tiny utility MLP used by some examples."""

        def __init__(self, dims: Sequence[int], act: type[nn.Module] = nn.Tanh) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            for i, (din, dout) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
                layers.append(nn.Linear(din, dout))
                if i < len(dims) - 2:
                    layers.append(act())
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(x)

else:  # pragma: no cover

    class MLP:  # type: ignore[no-redef]
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError(
                "PyTorch is not available; MLP cannot be constructed"
            ) from _import_error
