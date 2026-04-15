#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

sns.set_theme(
    style="whitegrid",
    context="talk",
    rc={
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.25,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "legend.frameon": False,
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    },
)

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


def add_cylinder(ax, radius=1.0):
    t = np.linspace(0.0, 2.0 * np.pi, 500)
    ax.plot(radius * np.cos(t), radius * np.sin(t), color="k", linewidth=0.8)


def normalize_to_one(z: np.ndarray) -> np.ndarray:
    zmax = np.max(z) if len(z) > 0 else 1.0
    if zmax <= 0.0:
        zmax = 1.0
    return z / zmax


def global_extent(data_dict):
    xs, ys = [], []
    for x, y, _ in data_dict.values():
        xs.append(x)
        ys.append(y)
    xmin = min(np.min(x) for x in xs)
    xmax = max(np.max(x) for x in xs)
    ymin = min(np.min(y) for y in ys)
    ymax = max(np.max(y) for y in ys)
    return xmin, xmax, ymin, ymax


def plot_four_panel_matrix(
        tm_fk: Path,
        tm_sk: Path,
        te_fk: Path,
        te_sk: Path,
        field: str = "utot",
        outdir: Path = Path("plots"),
        outfile: str = "tm_te_fk_sk_matrix.pdf",
        cmap_name: str = "viridis",
        marker_size: float = 12.0,
):
    entries = {
        (0, 0): (tm_fk, r"TM", r"First Kind"),
        (0, 1): (tm_sk, r"TM", r"Second Kind"),
        (1, 0): (te_fk, r"TE", r"First Kind"),
        (1, 1): (te_sk, r"TE", r"Second Kind"),
    }

    data = {}
    for key, (path, _, _) in entries.items():
        df = load_abs_field(path, field=field)
        z = normalize_to_one(df["z"].to_numpy())
        data[key] = (df["x"].to_numpy(), df["y"].to_numpy(), z)

    xmin, xmax, ymin, ymax = global_extent(data)

    fig = plt.figure(figsize=(6.6, 6.6), constrained_layout=False)
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 1, 0.05],
        wspace=0.0,
        hspace=0.0,
    )

    axes = np.empty((2, 2), dtype=object)
    cax = fig.add_subplot(gs[:, 2])

    mappable = None

    for i in range(2):
        for j in range(2):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax

            x, y, z = data[(i, j)]

            sc = ax.scatter(
                x,
                y,
                c=z,
                s=marker_size,
                cmap=cmap_name,
                vmin=0.0,
                vmax=1.0,
                linewidths=0,
                rasterized=True,
            )
            mappable = sc

            add_cylinder(ax)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal", adjustable="datalim")

            ax.set_xticks([-2, -1, 0, 1, 2])
            ax.set_yticks([-2, -1, 0, 1, 2])

            if i != 1:
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
                pad=2,
            )

            _, row_name, col_name = entries[(i, j)]

            if i == 0:
                ax.set_title(col_name, pad=8)

            if j == 0:
                ax.set_ylabel(row_name + "\n" + r"$y$")
            else:
                ax.set_ylabel("")

            if i == 1:
                ax.set_xlabel(r"$x$")
            else:
                ax.set_xlabel("")

    if mappable is not None:
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(r"$\left|u^{\mathrm{tot}}\right|_0$", fontsize=15)
        cbar.ax.tick_params(labelsize=12, length=3, width=0.8)

    fig.subplots_adjust(left=0.11, right=0.9, bottom=0.14, top=0.92)

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / outfile
    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tm-fk",
        type=Path,
        default=Path("../cmake-build-release/pec_tm_fk_grid_100x100.csv"),
    )
    parser.add_argument(
        "--tm-sk",
        type=Path,
        default=Path("../cmake-build-release/pec_tm_sk_grid_100x100.csv"),
    )
    parser.add_argument(
        "--te-fk",
        type=Path,
        default=Path("../cmake-build-release/pec_te_fk_grid_100x100.csv"),
    )
    parser.add_argument(
        "--te-sk",
        type=Path,
        default=Path("../cmake-build-release/pec_te_sk_grid_100x100.csv"),
    )
    parser.add_argument("--field", type=str, default="utot", choices=["uinc", "usca", "utr", "utot"])
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--marker-size", type=float, default=12.0)
    parser.add_argument("--outdir", type=Path, default=Path("plots"))
    parser.add_argument("--outfile", type=str, default="pec_tm_te_fk_sk_matrix_scatter_2.pdf")
    args = parser.parse_args()

    plot_four_panel_matrix(
        tm_fk=args.tm_fk,
        tm_sk=args.tm_sk,
        te_fk=args.te_fk,
        te_sk=args.te_sk,
        field=args.field,
        outdir=args.outdir,
        outfile=args.outfile,
        cmap_name=args.cmap,
        marker_size=args.marker_size,
    )


if __name__ == "__main__":
    main()