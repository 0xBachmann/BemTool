#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import seaborn as sns
from matplotlib import cm

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

sns.set_theme(
    style="whitegrid",
    context="paper",
    rc={
        "axes.spines.top": True,
        "axes.spines.right": True,
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
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


def add_cylinder(ax, radius=1.0):
    t = np.linspace(0.0, 2.0 * np.pi, 500)
    ax.plot(radius * np.cos(t), radius * np.sin(t), "k-", linewidth=1.0)


def folder_label(folder: Path) -> str:
    name = folder.name
    if name.startswith("n"):
        return rf"$n_i={name[1:]}$"
    return name


def omega_label(omega: float) -> str:
    if np.isclose(omega, 0.0):
        return r"$\Omega=0$"
    return rf"$\Omega={omega:.2f}$" if not np.isclose(abs(omega), 0.1) else rf"$\Omega={omega:.1f}$"


def style_for_omega(omega: float):
    if np.isclose(omega, 0.0):
        return "-"
    return "-" if omega > 0 else "--"

def masked_triangulation_crossing_circle(x, y, radius=1.0, eps=1e-6):
    tri = mtri.Triangulation(x, y)
    tris = tri.triangles

    r = np.sqrt(x**2 + y**2)
    rtri = r[tris]

    # Only mask triangles that straddle the boundary
    crosses_boundary = (rtri.min(axis=1) < radius - eps) & (rtri.max(axis=1) > radius + eps)

    tri.set_mask(crosses_boundary)
    return tri

def expand_omegas(nonnegative_omegas):
    vals = set()
    for w in nonnegative_omegas:
        if w < 0:
            raise ValueError("Please pass only nonnegative values to --omega.")
        if np.isclose(w, 0.0):
            vals.add(0.0)
        else:
            vals.add(float(w))
            vals.add(float(-w))
    return sorted(vals)

def style_axis(ax, i, j, folder, nrows, ncols, ks):
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
        ax.set_title(rf"$k={ks[j]:g}$", pad=4)

    if j == 0:
        ax.set_ylabel(folder_label(folder) + "\n" + r"$y$", labelpad=2)
    else:
        ax.set_ylabel("")

    if i == nrows - 1:
        ax.set_xlabel(r"$x$", labelpad=2)
    else:
        ax.set_xlabel("")
def build_abs_groups(omegas):
    return sorted({abs(w) for w in omegas})


def color_map_for_abs_omegas(abs_omegas):
    if len(abs_omegas) == 0:
        return {}

    base = cm.get_cmap("viridis")
    vals = np.linspace(0.0, 0.9, max(len(abs_omegas), 2))
    palette = [base(v) for v in vals]

    if len(abs_omegas) == 1:
        return {abs_omegas[0]: palette[0]}

    return {a: palette[i] for i, a in enumerate(abs_omegas)}


def compute_reference_level(index_df: pd.DataFrame, k0: float, level_pct: float, field: str):
    path0 = find_path(index_df, k0, 0.0)
    if path0 is None:
        raise RuntimeError(f"Missing Omega=0 file for k={k0:g}")

    df0 = load_abs_field(path0, field=field)
    max_u0 = np.max(df0["z"].to_numpy())
    level = 0.01 * level_pct * max_u0
    return level, max_u0


def plot_isocontour_matrix_overlay(
    folders,
    ks,
    nonnegative_omegas,
    level_pct: float,
    field: str = "utot",
    outdir: Path = Path("plots"),
    outfile: str = "isocontour_overlay_matrix.pdf",
):
    folders = [Path(f) for f in folders]
    ks = list(ks)
    omegas = expand_omegas(nonnegative_omegas)

    nrows = len(folders)
    ncols = len(ks)

    if nrows == 0 or ncols == 0:
        raise RuntimeError("Please provide at least one folder and one k.")

    abs_omegas = build_abs_groups(omegas)
    colors = color_map_for_abs_omegas(abs_omegas)

    fig = plt.figure(figsize=(3.0 * ncols, 3.0 * nrows + 1.45))
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        wspace=0.0,
        hspace=0.0,
    )

    axes = np.empty((nrows, ncols), dtype=object)

    for i, folder in enumerate(folders):
        index_df = scan_files(folder)
        if len(index_df) == 0:
            raise RuntimeError(f"No matching files found in {folder}")

        for j, k0 in enumerate(ks):
            ax = fig.add_subplot(gs[i, j])
            axes[i, j] = ax

            try:
                level, max_u0 = compute_reference_level(index_df, k0, level_pct, field)
            except RuntimeError as e:
                ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                        ha="center", va="center")
                ax.set_axis_off()
                continue

            drawn_labels = set()
            drawn_any = False

            for omega in omegas:
                path = find_path(index_df, k0, omega)
                if path is None:
                    continue

                df = load_abs_field(path, field=field)
                x = df["x"].to_numpy()
                y = df["y"].to_numpy()
                z = df["z"].to_numpy()

                zmin = np.min(z)
                zmax = np.max(z)
                if not (zmin <= level <= zmax):
                    continue

                color = colors[abs(omega)]
                linestyle = style_for_omega(omega)

                tri = masked_triangulation_crossing_circle(x, y, radius=1.0, eps=0)
                # tri = mtri.Triangulation(x, y)
                cs = ax.tricontour(
                    tri,
                    z,
                    levels=[level],
                    colors=[color],
                    linewidths=2.0,
                    linestyles=[linestyle],
                )

                if len(cs.allsegs[0]) > 0:
                    drawn_any = True
                    label = omega_label(omega)
                    if label not in drawn_labels and i == 0 and j == 0:
                        ax.plot([], [], color=color, linestyle=linestyle,
                                linewidth=2.0, label=label)
                        drawn_labels.add(label)

            add_cylinder(ax, radius=1.0)
            style_axis(ax, i, j, folder, nrows, ncols, ks)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 0.94),
        )

    fig.suptitle(
        rf"Isocontours of $|u^{{\mathrm{{{field}}}}}| = {level_pct:g}\%\max |u_0|$",
        y=0.985,
    )

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        bottom=0.07,
        top=0.82,
        wspace=0.0,
        hspace=0.0,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / outfile
    fig.savefig(outpath, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", type=Path, nargs="+", required=True)
    parser.add_argument("--k", type=float, nargs="+", required=True)
    parser.add_argument(
        "--omega",
        type=float,
        nargs="+",
        required=True,
        help="Nonnegative Omega magnitudes; positive values are automatically mirrored to -Omega.",
    )
    parser.add_argument("--level-pct", type=float, default=50.0,
                        help="Contour level as percentage of max|u_0| for this k")
    parser.add_argument("--field", type=str, default="utot",
                        choices=["uinc", "usca", "utr", "utot"])
    parser.add_argument("--outdir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    plot_isocontour_matrix_overlay(
        folders=args.folders,
        ks=args.k,
        nonnegative_omegas=args.omega,
        level_pct=args.level_pct,
        field=args.field,
        outdir=args.outdir,
        outfile=f"isocontour_overlay_matrix_{args.level_pct}.pdf",
    )


if __name__ == "__main__":
    main()