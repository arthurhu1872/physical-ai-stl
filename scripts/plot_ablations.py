# ruff: noqa: I001
from __future__ import annotations

import argparse
from collections.abc import Iterable
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
import statistics

import matplotlib
import matplotlib.pyplot as plt


# ------------------------------ Utilities ---------------------------------


@dataclass
class Series:
    label: str
    xs: list[float]
    ys: list[float]
    # Optional uncertainty for shaded band
    y_lo: list[float] | None = None
    y_hi: list[float] | None = None


# Keep Matplotlib's default color‑cycle (color‑blind friendly in modern MPL)
_DEFAULT_PROP_CYCLE = plt.rcParams["axes.prop_cycle"]

# Likely column names in your CSVs
_AUTO_XCANDIDATES = ("lambda", "lam", "stl_weight", "weight", "alpha", "x")
_AUTO_YCANDIDATES = ("robustness", "rho", "rtamt", "moonlight", "value", "y")
_LOWER_CANDIDATES = ("y_lo", "lo", "lower", "ymin", "ci_lo", "lb")
_UPPER_CANDIDATES = ("y_hi", "hi", "upper", "ymax", "ci_hi", "ub")
_STDLIKE_CANDIDATES = ("std", "stdev", "stderr", "sem")


def _as_float(s: str) -> float | None:
    try:
        return float(s)
    except Exception:
        return None


def _sniff_and_open(path: Path) -> tuple[csv.Dialect, list[list[str]]]:
    # Mode "r" is default; avoid unnecessary argument (ruff UP015).
    with open(path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;| ")
        except Exception:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        rows = [row for row in reader]
    return dialect, rows


def _sanitize_label_from_filename(p: Path) -> str:
    name = p.stem
    # Remove common prefixes/suffixes
    name = re.sub(r"^(results?_)?", "", name, flags=re.I)
    name = re.sub(r"(_?ablations?|_?diffusion|_?heat2d|_?burgers?)", "", name, flags=re.I)
    name = re.sub(r"[-_]+", " ", name).strip()
    return name or p.stem


def _infer_col_index(header: list[str], candidates: Iterable[str]) -> int | None:
    low = [h.strip().lower() for h in header]
    for cand in candidates:
        if cand in low:
            return low.index(cand)
    return None


def _rows_to_series(rows: list[list[str]], file_label: str, args) -> Series:
    # Drop comment lines and empties
    clean: list[list[str]] = [r for r in rows if r and not str(r[0]).lstrip().startswith("#")]
    if not clean:
        raise ValueError("CSV appears empty after removing comments.")

    header_present = any(not _as_float(c) for c in clean[0])
    header: list[str]
    data_rows: list[list[str]]

    if header_present:
        header = [c.strip() for c in clean[0]]
        data_rows = clean[1:]
    else:
        header = []
        data_rows = clean

    # Detect columns
    if args.xcol:
        x_idx = _infer_col_index(header, [args.xcol.lower()]) if header else None
        if x_idx is None and header:
            raise ValueError(f"xcol '{args.xcol}' not found in header {header}")
        if x_idx is None and not header:
            x_idx = 0
    else:
        x_idx = _infer_col_index(header, _AUTO_XCANDIDATES) if header else 0

    if args.ycol:
        y_idx = _infer_col_index(header, [args.ycol.lower()]) if header else None
        if y_idx is None and header:
            raise ValueError(f"ycol '{args.ycol}' not found in header {header}")
        if y_idx is None and not header:
            y_idx = 1
    else:
        y_idx = _infer_col_index(header, _AUTO_YCANDIDATES) if header else 1

    # Optional uncertainty
    lo_idx = _infer_col_index(header, _LOWER_CANDIDATES) if header else None
    hi_idx = _infer_col_index(header, _UPPER_CANDIDATES) if header else None
    std_idx = _infer_col_index(header, _STDLIKE_CANDIDATES) if header else None

    xs: list[float] = []
    ys: list[float] = []
    y_lo_vals: list[float] = []
    y_hi_vals: list[float] = []

    for r in data_rows:
        if not r or all(c.strip() == "" for c in r):
            continue
        xv = _as_float(r[x_idx]) if x_idx is not None and x_idx < len(r) else None
        yv = _as_float(r[y_idx]) if y_idx is not None and y_idx < len(r) else None
        if xv is None or yv is None:
            continue
        xs.append(xv)
        ys.append(yv)

        lo_val = None
        hi_val = None
        if lo_idx is not None and lo_idx < len(r):
            lo_val = _as_float(r[lo_idx])
        if hi_idx is not None and hi_idx < len(r):
            hi_val = _as_float(r[hi_idx])

        if lo_val is not None and hi_val is not None:
            y_lo_vals.append(lo_val)
            y_hi_vals.append(hi_val)
        elif std_idx is not None and std_idx < len(r) and _as_float(r[std_idx]) is not None:
            std = float(r[std_idx])
            # Default multiplier: 1.0 for std; 1.96 for sem/stderr if header suggests "se".
            mult = (
                args.err_mult
                if args.err_mult is not None
                else (1.96 if header and "se" in header[std_idx].lower() else 1.0)
            )
            y_lo_vals.append(yv - mult * std)
            y_hi_vals.append(yv + mult * std)

    # Sort by x
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ys = [ys[i] for i in order]

    y_lo_ret: list[float] | None = y_lo_vals if y_lo_vals else None
    y_hi_ret: list[float] | None = y_hi_vals if y_hi_vals else None

    return Series(label=file_label, xs=xs, ys=ys, y_lo=y_lo_ret, y_hi=y_hi_ret)


def _iter_input_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for patt in patterns:
        p = Path(patt).expanduser()
        if p.is_dir():
            for sub in sorted(p.rglob("*.csv")):
                files.append(sub)
        else:
            # Glob expansion ourselves to support quotes
            base = p.parent if p.parent != Path("") else Path(".")
            for cand in base.glob(p.name):
                if cand.is_file():
                    files.append(cand)
    # De‑dupe
    uniq: dict[str, Path] = {}
    for f in files:
        uniq[str(f.resolve())] = f
    return [uniq[k] for k in sorted(uniq.keys())]


def _round_if_requested(xs: list[float], decimals: int | None) -> list[float]:
    if decimals is None:
        return xs
    return [round(x, decimals) for x in xs]


def _aggregate(series_list: list[Series], x_decimals: int | None) -> Series:
    if not series_list:
        raise ValueError("No series to aggregate.")

    # Build map x -> list[ys]
    buckets: dict[float, list[float]] = {}
    for s in series_list:
        xr = _round_if_requested(s.xs, x_decimals)
        if len(xr) != len(s.ys):
            raise ValueError("Mismatched x/y lengths in a series.")
        for x, y in zip(xr, s.ys):
            buckets.setdefault(x, []).append(y)

    xs_sorted = sorted(buckets.keys())
    means: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    for x in xs_sorted:
        vals = buckets[x]
        m = statistics.fmean(vals)
        means.append(m)
        if len(vals) > 1:
            stdev = statistics.pstdev(vals) if len(vals) == 2 else statistics.stdev(vals)
            # 95% normal approx
            err = 1.96 * (stdev / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
            lo.append(m - err)
            hi.append(m + err)
        else:
            lo.append(m)
            hi.append(m)
    return Series(label="mean±95%CI", xs=xs_sorted, ys=means, y_lo=lo, y_hi=hi)


# ------------------------------ Plotting ----------------------------------


def _apply_style(args) -> None:
    matplotlib.rcParams.update(
        {
            "figure.dpi": args.dpi,
            "savefig.dpi": args.dpi,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.alpha": 0.35,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.prop_cycle": _DEFAULT_PROP_CYCLE,
            "mathtext.default": "regular",
        }
    )


def _plot(ax, s: Series, draw_band: bool = True, marker: str = "o") -> None:
    ax.plot(s.xs, s.ys, marker=marker, linewidth=2.0, markersize=4, label=s.label)
    if (
        draw_band
        and s.y_lo is not None
        and s.y_hi is not None
        and len(s.y_lo) == len(s.ys) == len(s.y_hi)
    ):
        ax.fill_between(s.xs, s.y_lo, s.y_hi, alpha=0.15, linewidth=0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot robustness curves from diffusion/heat ablation CSV files."
    )
    ap.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help=(
            "CSV file(s), directory(ies), or glob(s). Each CSV should contain "
            "λ vs robustness."
        ),
    )
    ap.add_argument(
        "--out",
        type=str,
        default="figs/ablations_diffusion.png",
        help="Primary output image path. Parent directory is created.",
    )
    ap.add_argument(
        "--out-formats",
        type=str,
        default="png,pdf",
        help="Comma‑separated list of formats to save (e.g., png,pdf,svg).",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="Diffusion 1D: STL Weight Sweep",
        help="Figure title.",
    )
    ap.add_argument(
        "--xlabel",
        type=str,
        default="λ (STL weight)",
        help="X label.",
    )
    ap.add_argument(
        "--ylabel",
        type=str,
        default="Robustness  G(mean_x u ≤ u_max)",
        help="Y label.",
    )
    ap.add_argument("--xscale", choices=["linear", "log"], default="linear")
    ap.add_argument("--yscale", choices=["linear", "log"], default="linear")
    ap.add_argument(
        "--xcol",
        type=str,
        default=None,
        help="Explicit x column name (if header present).",
    )
    ap.add_argument(
        "--ycol",
        type=str,
        default=None,
        help="Explicit y column name (if header present).",
    )
    ap.add_argument(
        "--aggregate",
        action="store_true",
        help="Aggregate across all inputs (mean ± 95% CI) assuming shared λ grid.",
    )
    ap.add_argument(
        "--x-decimals",
        type=int,
        default=None,
        help=(
            "Round x to this many decimals before aggregation "
            "(helps align near‑duplicates)."
        ),
    )
    ap.add_argument(
        "--err-mult",
        type=float,
        default=None,
        help=(
            "Multiplier for 'std/sem/stderr' columns when present. "
            "Default: 1.0 for std, 1.96 for sem/stderr."
        ),
    )
    ap.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(4.0, 3.0),
        help="Figure size inches (W H).",
    )
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    ap.add_argument("--legend", action="store_true", help="Show legend (auto‑labels by filename).")
    args = ap.parse_args()

    # Expand inputs
    files = _iter_input_files(args.csv)
    if not files:
        raise SystemExit("No CSV files found.")

    # Load series
    series_list: list[Series] = []
    for f in files:
        _, rows = _sniff_and_open(Path(f))
        label = _sanitize_label_from_filename(Path(f))
        s = _rows_to_series(rows, label, args)
        series_list.append(s)

    # Aggregate if requested
    if args.aggregate and len(series_list) > 1:
        series_list = [_aggregate(series_list, args.x_decimals)]

    # Plot
    _apply_style(args)
    fig, ax = plt.subplots(figsize=tuple(args.figsize))
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    ax.set_title(args.title)
    ax.set_xscale(args.xscale)
    ax.set_yscale(args.yscale)

    for s in series_list:
        _plot(ax, s, draw_band=True, marker="o")

    if args.legend and len(series_list) > 1:
        ax.legend(frameon=False)

    ax.margins(x=0.02)
    fig.tight_layout()

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    formats = [fmt.strip() for fmt in args.out_formats.split(",") if fmt.strip()]
    saved: list[Path] = []
    for fmt in formats:
        target = out_path.with_suffix("." + fmt.lower())
        fig.savefig(target, bbox_inches="tight")
        saved.append(target)

    plt.close(fig)

    # Friendly message
    if len(saved) == 1:
        print(f"Wrote {saved[0]}")
    else:
        print("Wrote:")
        for p in saved:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
