#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import seaborn as sns

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

sns.set_theme(
    style="whitegrid",
    context="paper",
    rc={
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    },
)

TAG_RE = re.compile(
    r"dielectric_grid_k0_(?P<k0>[pm]?\d+p\d+e[pm]\d+)_Omega_(?P<omega>[pm]?\d+p\d+e[pm]\d+)\.csv$"
)


def decode_tag(tag: str) -> float:
    m = re.fullmatch(r"([pm]?)(\d+)p(\d+)e([pm])(\d+)", tag)
    if not m:
        raise ValueError(f"Cannot decode tag: {tag}")
    sign_tag, intpart, fracpart, exp_sign_tag, exp_digits = m.groups()
    sign = "-" if sign_tag == "m" else ""
    exp_sign = "-" if exp_sign_tag == "m" else "+"
    return float(f"{sign}{intpart}.{fracpart}e{exp_sign}{exp_digits}")


def scan_files(folder: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(folder.glob("dielectric_grid_k0_*_Omega_*.csv")):
        m = TAG_RE.fullmatch(f.name)
        if not m:
            continue
        rows.append(
            {
                "path": f,
                "k0": decode_tag(m.group("k0")),
                "omega": decode_tag(m.group("omega")),
            }
        )
    return pd.DataFrame(rows)


def find_path(index_df: pd.DataFrame, k0: float, omega: float, tol=1e-14):
    sel = index_df[
        np.isclose(index_df["k0"], k0, atol=tol, rtol=0.0)
        & np.isclose(index_df["omega"], omega, atol=tol, rtol=0.0)
    ]
    if len(sel) == 0:
        return None
    return Path(sel.iloc[0]["path"])


def load_abs_field(path: Path, field: str = "utot") -> pd.DataFrame:
    df = pd.read_csv(path)

    if "is_skipped" in df.columns:
        df = df[df["is_skipped"] == 0].copy()

    abs_col = f"abs_{field}"
    if abs_col in df.columns:
        z = df[abs_col].to_numpy()
    else:
        real_col = f"real_{field}"
        imag_col = f"imag_{field}"
        z = np.sqrt(df[real_col].to_numpy() ** 2 + df[imag_col].to_numpy() ** 2)

    return pd.DataFrame(
        {
            "x": df["x"].to_numpy(),
            "y": df["y"].to_numpy(),
            "z": z,
        }
    )


def compute_max_u0(index_df: pd.DataFrame, k0: float, field: str = "utot") -> float:
    path0 = find_path(index_df, k0, 0.0)
    if path0 is None:
        raise RuntimeError(f"Missing Omega=0 file for k={k0:g}")
    df0 = load_abs_field(path0, field=field)
    max_u0 = np.max(df0["z"].to_numpy())
    if max_u0 <= 0.0:
        max_u0 = 1.0
    return max_u0


def add_cylinder(ax, radius=1.0):
    t = np.linspace(0.0, 2.0 * np.pi, 500)
    ax.plot(radius * np.cos(t), radius * np.sin(t), color="k", linewidth=0.8)


def omega_title(omega: float) -> str:
    if np.isclose(omega, 0.0):
        return r"$\Omega = 0$"
    exp = int(np.log10(abs(omega)))
    sign = "-" if omega < 0 else ""
    return rf"$\Omega = {sign}10^{{{exp}}}$"


def plot_isocontour_matrix(
    folder: Path,
    ks,
    omegas,
    field: str = "utot",
    outdir: Path = Path("plots"),
    outfile: str = "isocontour_matrix.pdf",
    levels=15,
    cmap_name="viridis",
    vmin=None,
    vmax=None,
):
    index_df = scan_files(folder)
    if len(index_df) == 0:
        raise RuntimeError(f"No matching files found in {folder}")

    ks = list(ks)
    omegas = list(omegas)

    nrows = len(ks)
    ncols = len(omegas)
    if nrows == 0 or ncols == 0:
        raise RuntimeError("Please provide at least one k and one omega value.")

    normalized_data = {}
    global_min = np.inf
    global_max = -np.inf

    for k0 in ks:
        max_u0 = compute_max_u0(index_df, k0, field=field)
        for omega in omegas:
            path = find_path(index_df, k0, omega)
            if path is None:
                normalized_data[(k0, omega)] = None
                continue

            df = load_abs_field(path, field=field)
            z = df["z"].to_numpy() / max_u0
            normalized_data[(k0, omega)] = (df["x"].to_numpy(), df["y"].to_numpy(), z)

            if len(z) > 0:
                global_min = min(global_min, float(np.min(z)))
                global_max = max(global_max, float(np.max(z)))

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        raise RuntimeError("No valid data found for the requested k/Omega combinations.")

    if vmin is None:
        vmin = global_min
    if vmax is None:
        vmax = global_max

    contour_levels = np.linspace(vmin, vmax, levels)

    fig = plt.figure(figsize=(3.0 * ncols + 1.9, 3.0 * nrows + 0.4))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[1] * ncols + [0.05],
        wspace=0.0,
        hspace=0.0,
    )

    axes = np.empty((nrows, ncols), dtype=object)
    cax = fig.add_subplot(gs[:, -1])

    mappable = None

    def style_axis(ax, i, j, k0):
        ax.set_aspect("equal")
        ax.grid(False)
        ax.margins(0.0)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)

        if j == ncols - 1:
            ax.set_xticks([-2, -1, 0, 1, 2])
        else:
            ax.set_xticks([-2, -1, 0, 1])

        if i == 0:
            ax.set_yticks([-2, -1, 0, 1, 2])
        else:
            ax.set_yticks([-2, -1, 0, 1])

        if i != nrows - 1:
            ax.tick_params(axis="x", labelbottom=False)
        if j != 0:
            ax.tick_params(axis="y", labelleft=False)

        ax.tick_params(
            top=False,
            right=False,
            bottom=True,
            left=True,
            length=3,
            width=0.8,
            pad=1,
        )

        if i == 0:
            ax.set_title(omega_title(omegas[j]), pad=6)

        if j == 0:
            ax.set_ylabel(rf"$k = {k0:g}$" + "\n" + r"$y$", labelpad=2)
        else:
            ax.set_ylabel("")

        if i == nrows - 1:
            ax.set_xlabel(r"$x$", labelpad=2)
        else:
            ax.set_xlabel("")

    for i, k0 in enumerate(ks):
        for j, omega in enumerate(omegas):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax
            entry = normalized_data[(k0, omega)]

            if entry is None:
                ax.text(
                    0.5, 0.5, "missing file",
                    transform=ax.transAxes,
                    ha="center", va="center"
                )
                ax.set_axis_off()
                continue

            x, y, z = entry
            tri = mtri.Triangulation(x, y)

            cs = ax.tricontourf(
                tri,
                z,
                levels=contour_levels,
                cmap=cmap_name,
                vmin=vmin,
                vmax=vmax,
            )
            mappable = cs

            add_cylinder(ax)
            style_axis(ax, i, j, k0)

    if mappable is not None:
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(r"$\left|u^{\mathrm{tot}}\right|_0$")
        cbar.ax.tick_params(labelsize=10, length=3, width=0.8)

    fig.subplots_adjust(
        left=0.08,
        right=0.92,
        bottom=0.045,
        top=0.965,
        wspace=0.0,
        hspace=0.0,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / outfile
    fig.savefig(outpath, dpi=220, pad_inches=0)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=Path("."))
    parser.add_argument("--k", type=float, nargs="+", required=True,
                        help="Row values for k")
    parser.add_argument("--omega", type=float, nargs="+", required=True,
                        help="Column values for Omega")
    parser.add_argument("--field", type=str, default="utot",
                        choices=["uinc", "usca", "utr", "utot"])
    parser.add_argument("--levels", type=int, default=15,
                        help="Number of contour levels")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Matplotlib colormap name")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("plots"))
    parser.add_argument("--outfile", type=str, default="isocontour_matrix_n1.5.pdf")
    args = parser.parse_args()

    plot_isocontour_matrix(
        folder=args.folder,
        ks=args.k,
        omegas=args.omega,
        field=args.field,
        outdir=args.outdir,
        outfile=args.outfile,
        levels=args.levels,
        cmap_name=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()

#python plotting_results.py --folder n1.5 --k 2 3 4 --omega 0 1e-3 1e-2 1e-1