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
    context="talk",
    rc={
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.35,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "legend.frameon": False,
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
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


def find_path(index_df: pd.DataFrame, k0: float, omega: float = 0.0, tol=1e-14):
    sel = index_df[
        np.isclose(index_df["k0"], k0, atol=tol, rtol=0.0)
        & np.isclose(index_df["omega"], omega, atol=tol, rtol=0.0)
    ]
    if len(sel) == 0:
        return None
    return Path(sel.iloc[0]["path"])


def load_component_field(path: Path, field: str = "utot", component: str = "abs") -> pd.DataFrame:
    df = pd.read_csv(path)

    if "is_skipped" in df.columns:
        df = df[df["is_skipped"] == 0].copy()

    if component == "real":
        z = df[f"real_{field}"].to_numpy()
    elif component == "imag":
        z = df[f"imag_{field}"].to_numpy()
    elif component == "abs":
        abs_col = f"abs_{field}"
        if abs_col in df.columns:
            z = df[abs_col].to_numpy()
        else:
            re = df[f"real_{field}"].to_numpy()
            im = df[f"imag_{field}"].to_numpy()
            z = np.sqrt(re**2 + im**2)
    else:
        raise ValueError(f"Unknown component: {component}")

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


def parse_folder_label(folder: Path) -> str:
    name = folder.name
    if name.startswith("n"):
        val = name[1:]
        return rf"$n = {val}$"
    return name


def component_label(component: str, field: str) -> str:
    if component == "real":
        return rf"$\Re(u^{{\mathrm{{{field}}}}})$"
    if component == "imag":
        return rf"$\Im(u^{{\mathrm{{{field}}}}})$"
    if component == "abs":
        return rf"$\left|u^{{\mathrm{{{field}}}}}\right|_0$"
    return component


def triangulation_to_grid(x, y, z, nx=220, ny=220, xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0):
    tri = mtri.Triangulation(x, y)
    interp = mtri.LinearTriInterpolator(tri, z)

    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    XI, YI = np.meshgrid(xi, yi)

    ZI = interp(XI, YI)
    if np.ma.isMaskedArray(ZI):
        ZI = ZI.filled(np.nan)

    return XI, YI, ZI


def nice_limits(component_data, component):
    vals = []
    for entry in component_data.values():
        if entry is None:
            continue
        z = entry[2]
        vals.append(z[np.isfinite(z)])

    if not vals:
        return -1.0, 1.0

    allz = np.concatenate(vals)
    if component == "abs":
        vmax = np.nanmax(allz)
        if vmax <= 0.0:
            vmax = 1.0
        return 0.0, vmax

    a = np.nanmax(np.abs(allz))
    if a <= 0.0:
        a = 1.0
    return -a, a


def plot_stationary_matrix(
    folders,
    ks,
    field: str = "utot",
    outdir: Path = Path("plots"),
    outfile: str = "stationary_components_matrix.pdf",
    cmap_realimag: str = "RdBu_r",
    cmap_abs: str = "viridis",
):
    folders = [Path(f) for f in folders]
    ks = list(ks)
    components = ["real", "imag", "abs"]

    n_folder = len(folders)
    nrows = 3 * n_folder
    ncols = len(ks)

    if n_folder == 0 or ncols == 0:
        raise RuntimeError("Please provide at least one directory and one k value.")

    data = {}

    for folder in folders:
        index_df = scan_files(folder)
        if len(index_df) == 0:
            print(f"Warning: no matching files found in {folder}")

        for comp in components:
            for k0 in ks:
                path = find_path(index_df, k0, omega=0.0)
                key = (folder, comp, k0)

                if path is None:
                    data[key] = None
                    continue

                df = load_component_field(path, field=field, component=comp)
                x = df["x"].to_numpy()
                y = df["y"].to_numpy()
                z = df["z"].to_numpy()

                zscale = np.nanmax(np.abs(z)) if len(z) > 0 else 1.0
                if not np.isfinite(zscale) or zscale <= 0.0:
                    zscale = 1.0

                data[key] = (x, y, z / zscale)

    fig = plt.figure(figsize=(3.0 * ncols + 0.75, 3.0 * nrows + 0.4))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[1] * ncols + [0.06],
        wspace=0.0,
        hspace=0.0,
    )

    cax = fig.add_subplot(gs[:, -1])

    def style_axis(ax, i, j, row_label):
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_box_aspect(1)
        ax.margins(0)

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
            ax.set_title(rf"$k = {ks[j]:g}$", pad=6)

        if j == 0:
            ax.set_ylabel(row_label, labelpad=4)
        else:
            ax.set_ylabel("")

        if i == nrows - 1:
            ax.set_xlabel(r"$x$", labelpad=2)
        else:
            ax.set_xlabel("")

    vmin = -1.0
    vmax = 1.0
    cmap = "viridis"

    for fidx, folder in enumerate(folders):
        folder_label = parse_folder_label(folder)

        for cidx, comp in enumerate(components):
            i = 3 * fidx + cidx
            row_label = component_label(comp, field)

            for j, k0 in enumerate(ks):
                ax = fig.add_subplot(gs[i, j])
                entry = data[(folder, comp, k0)]

                if entry is None:
                    ax.text(0.5, 0.5, "missing file", transform=ax.transAxes,
                            ha="center", va="center")
                    ax.set_axis_off()
                    continue

                x, y, z = entry
                XI, YI, ZI = triangulation_to_grid(
                    x, y, z,
                    nx=220, ny=220,
                    xmin=-2.0, xmax=2.0,
                    ymin=-2.0, ymax=2.0,
                )

                hm = ax.imshow(
                    ZI,
                    origin="lower",
                    extent=[XI.min(), XI.max(), YI.min(), YI.max()],
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="none",
                    resample=False,
                    aspect="equal",
                )
                mappable = hm

                add_cylinder(ax)
                style_axis(ax, i, j, row_label)

    if mappable is not None:
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(r"Normalized Field Value", fontsize=13)
        cbar.ax.tick_params(labelsize=11, length=3, width=0.8)

    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / outfile
    fig.savefig(outpath, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folders",
        type=str,
        nargs="+",
        required=True,
        help="Directories for different refractive indices, e.g. n1 n1.5",
    )
    parser.add_argument(
        "--k",
        type=float,
        nargs="+",
        required=True,
        help="Column values for k",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="utot",
        choices=["uinc", "usca", "utr", "utot"],
    )
    parser.add_argument("--outdir", type=Path, default=Path("plots"))
    parser.add_argument("--outfile", type=str, default="stationary_components_matrix_n1.5.pdf")
    args = parser.parse_args()

    plot_stationary_matrix(
        folders=args.folders,
        ks=args.k,
        field=args.field,
        outdir=args.outdir,
        outfile=args.outfile,
    )


if __name__ == "__main__":
    main()
    
# python plotting_stationary.py --folders n1.5 --k 1 2 3 4