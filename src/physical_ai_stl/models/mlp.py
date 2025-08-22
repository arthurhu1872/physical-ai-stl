from __future__ import annotations

from typing import Callable, Iterable, Sequence, Union

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["MLP", "Sine"]


# --- small utilities ---------------------------------------------------------------------------

class Sine(nn.Module):
    def __init__(self, w0: float = 1.0) -> None:
        super().__init__()
        self.w0 = float(w0)

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        return torch.sin(self.w0 * x)

    def extra_repr(self) -> str:
        return f"w0={self.w0:g}"


ActivationArg = Union[str, nn.Module, Callable[[], nn.Module]]

def _normalize_activation(act: ActivationArg) -> Callable[[], nn.Module]:
    if isinstance(act, str):
        name = act.lower()
        if name in {"tanh"}:
            return nn.Tanh
        if name in {"relu"}:
            return lambda: nn.ReLU(inplace=False)
        if name in {"silu", "swish"}:
            return nn.SiLU
        if name in {"gelu"}:
            return nn.GELU
        if name in {"sigmoid"}:
            return nn.Sigmoid
        if name in {"sine", "sin", "siren"}:
            return Sine  # default w0=1.0 (see init below for SIREN)
        raise ValueError(f"Unknown activation string: {act!r}")
    if isinstance(act, nn.Module):
        return act.__class__
    if callable(act):
        return act
    raise TypeError("activation must be a string, nn.Module, or a 0-arg callable returning an nn.Module.")


# --- model -------------------------------------------------------------------------------------

class MLP(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (64, 64, 64),
        activation: ActivationArg = "tanh",
        *,
        out_activation: ActivationArg | None = None,
        bias: bool = True,
        init: str = "auto",
        last_layer_scale: float | None = None,
        skip_connections: Sequence[int] = (),
        weight_norm: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()

        if in_dim <= 0 or out_dim <= 0:
            raise ValueError("in_dim and out_dim must be positive integers.")
        if not isinstance(hidden, Sequence) or len(hidden) == 0:
            raise ValueError("hidden must be a non-empty sequence of positive integers.")
        if any(h <= 0 for h in hidden):
            raise ValueError("All hidden layer sizes must be positive.")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.skip_connections = tuple(int(i) for i in skip_connections)

        act_ctor = _normalize_activation(activation)
        out_act_ctor = _normalize_activation(out_activation) if out_activation is not None else None

        # Build layers allowing optional skip concatenations.
        layers: list[nn.Module] = []
        acts: list[nn.Module] = []

        def maybe_weight_norm(linear: nn.Linear) -> nn.Linear:
            return nn.utils.weight_norm(linear) if weight_norm else linear

        # Hidden stack
        last_dim = self.in_dim
        for idx, h in enumerate(self.hidden):
            in_d = last_dim + (self.in_dim if idx in self.skip_connections else 0)
            lin = nn.Linear(in_d, h, bias=bias, dtype=dtype, device=device)
            layers.append(maybe_weight_norm(lin))
            acts.append(act_ctor())
            last_dim = h

        # Output layer
        self.out = maybe_weight_norm(nn.Linear(last_dim, self.out_dim, bias=bias, dtype=dtype, device=device))
        self.layers = nn.ModuleList(layers)
        self.acts = nn.ModuleList(acts)
        self.out_act = out_act_ctor() if out_act_ctor is not None else None

        # Initialization
        self.reset_parameters(init=init, activation=act_ctor, last_layer_scale=last_layer_scale)

    # --- initialization ------------------------------------------------------------------------

    @torch.no_grad()
    def reset_parameters(
        self,
        *,
        init: str = "auto",
        activation: Callable[[], nn.Module] | None = None,
        last_layer_scale: float | None = None,
    ) -> None:
        init = init.lower()
        # Inspect activation type if available for auto mode
        act_name = None
        if activation is not None:
            try:
                act_name = activation().__class__.__name__.lower()
            except Exception:
                act_name = None

        def xavier_(m: nn.Linear, gain: float = 1.0) -> None:
            nn.init.xavier_uniform_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        def kaiming_(m: nn.Linear, nonlinearity: str = "relu") -> None:
            nn.init.kaiming_uniform_(m.weight, a=0.0, mode="fan_in", nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        def siren_(first: nn.Linear | None, rest: Iterable[nn.Linear], w0: float = 30.0) -> None:
            if first is not None:
                # First layer: U(-1/in_dim, 1/in_dim)
                in_d = first.weight.shape[1]
                bound = 1.0 / in_d
                nn.init.uniform_(first.weight, -bound, bound)
                if first.bias is not None:
                    nn.init.uniform_(first.bias, -bound, bound)
            for m in rest:
                in_d = m.weight.shape[1]
                # SIREN paper: U(-sqrt(6/in)/w0, sqrt(6/in)/w0)
                bound = (6.0 / in_d) ** 0.5 / w0
                nn.init.uniform_(m.weight, -bound, bound)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Choose scheme
        if init == "auto":
            if act_name and ("tanh" in act_name or "sigmoid" in act_name):
                chosen = "xavier"
            elif act_name and (("relu" in act_name) or ("silu" in act_name) or ("gelu" in act_name)):
                chosen = "kaiming"
            elif act_name and ("sine" in act_name):
                chosen = "siren"
            else:
                chosen = "xavier"  # safe default
        else:
            chosen = init

        # Apply
        if chosen == "xavier":
            gain = nn.init.calculate_gain("tanh") if (act_name and "tanh" in act_name) else 1.0
            for m in list(self.layers) + [self.out]:
                xavier_(m, gain=gain)
        elif chosen == "kaiming":
            for m in list(self.layers) + [self.out]:
                kaiming_(m, nonlinearity="relu")
        elif chosen == "siren":
            first = self.layers[0] if len(self.layers) > 0 else None
            rest = self.layers[1:]
            w0 = self._infer_siren_w0()
            siren_(first, rest, w0=w0)
            # Output layer: small init (as in SIREN examples)
            nn.init.uniform_(self.out.weight, -1e-4, 1e-4)
            if self.out.bias is not None:
                nn.init.zeros_(self.out.bias)
        else:
            raise ValueError(f"Unknown init scheme: {init!r}")

        if last_layer_scale is not None and last_layer_scale > 0:
            self.out.weight.mul_(float(last_layer_scale))

    def _infer_siren_w0(self) -> float:
        # Look for Sine activations and pick w0 from the first one if present.
        for a in self.acts:
            if isinstance(a, Sine):
                return a.w0 if a.w0 is not None else 30.0
        return 30.0

    # --- forward -------------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"Expected input with last dim {self.in_dim}, got {tuple(x.shape)}.")
        h = x
        for idx, (lin, act) in enumerate(zip(self.layers, self.acts)):
            if idx in self.skip_connections:
                h = torch.cat((h, x), dim=-1)
            h = act(lin(h))
        y = self.out(h)
        if self.out_act is not None:
            y = self.out_act(y)
        return y

    # --- ergonomics ----------------------------------------------------------------------------

    def count_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def extra_repr(self) -> str:
        act_names = [a.__class__.__name__ for a in self.acts[:2]]
        act = act_names[0] + ("..." if len(self.acts) > 1 and len(set(act_names)) == 1 else "")
        skips = f", skip={self.skip_connections}" if self.skip_connections else ""
        return f"in={self.in_dim}, out={self.out_dim}, hidden={self.hidden}, act={act}{skips}"
