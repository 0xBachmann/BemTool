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


def omega_label(omega: float) -> str:
    if np.isclose(omega, 0.0):
        return r"$\Omega=0$"
    exp = int(np.log10(abs(omega)))
    sign = "-" if omega < 0.0 else ""
    return rf"$\Omega={sign}10^{{{exp}}}$"


def style_for_omega(omega: float):
    if np.isclose(omega, 0.0):
        return "-"
    return "-" if omega > 0 else "--"


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


def build_abs_groups(omegas):
    return sorted({abs(w) for w in omegas})

def color_map_for_abs_omegas(abs_omegas):
    if len(abs_omegas) == 0:
        return {}

    # truncate viridis to avoid the bright yellow end
    base = cm.get_cmap("viridis")
    vals = np.linspace(0., 0.9, max(len(abs_omegas), 2))
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


def plot_single_isocontour_overlay(
    folder: Path,
    k0: float,
    nonnegative_omegas,
    level_pct: float,
    field: str = "utot",
    outdir: Path = Path("plots"),
    outfile: str | None = None,
):
    index_df = scan_files(folder)
    if len(index_df) == 0:
        raise RuntimeError(f"No matching files found in {folder}")

    omegas = expand_omegas(nonnegative_omegas)

    outdir.mkdir(parents=True, exist_ok=True)

    level, max_u0 = compute_reference_level(index_df, k0, level_pct, field)

    if outfile is None:
        outfile = f"isocontour_overlay_k{k0:g}_pct{level_pct:g}_{field}.pdf"
    outpath = outdir / outfile

    fig, ax = plt.subplots(figsize=(6.8, 6.2), constrained_layout=True)

    abs_omegas = build_abs_groups(omegas)
    colors = color_map_for_abs_omegas(abs_omegas)

    drawn_labels = set()
    drawn_any = False

    for omega in omegas:
        path = find_path(index_df, k0, omega)
        if path is None:
            print(f"Missing file for k={k0:g}, Omega={omega:.3e}")
            continue

        df = load_abs_field(path, field=field)
        x = df["x"].to_numpy()
        y = df["y"].to_numpy()
        z = df["z"].to_numpy()

        zmin = np.min(z)
        zmax = np.max(z)
        if not (zmin <= level <= zmax):
            print(
                f"Skipping Omega={omega:.3e}: contour level {level:.6g} "
                f"outside data range [{zmin:.3g}, {zmax:.3g}]"
            )
            continue

        color = colors[abs(omega)]
        linestyle = style_for_omega(omega)

        tri = mtri.Triangulation(x, y)
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
            if label not in drawn_labels:
                ax.plot([], [], color=color, linestyle=linestyle, linewidth=2.0, label=label)
                drawn_labels.add(label)

    add_cylinder(ax, radius=1.0)

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        rf"$|u_{{{field[1:]}}}| = {level_pct:g}\%\max |u_0|$ for $k={k0:g}$"
    )
    ax.grid(True, linestyle=":", alpha=0.5)

    if drawn_any:
        ax.legend()
    else:
        ax.text(
            0.5, 0.5, "No contours drawn",
            transform=ax.transAxes, ha="center", va="center"
        )

    fig.savefig(outpath, dpi=220)
    plt.close(fig)
    print(f"Wrote {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=Path("."))
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument(
        "--omega",
        type=float,
        nargs="+",
        required=True,
        help="Nonnegative Omega magnitudes; positive values are automatically mirrored to -Omega.",
    )
    parser.add_argument("--level-pct", type=float, default=50.0,
                        help="Contour level as percentage of max|u_0| for this k")
    parser.add_argument("--field", type=str, default="utot", choices=["uinc", "usca", "utr", "utot"])
    parser.add_argument("--outdir", type=Path, default=Path("plots"))
    parser.add_argument("--outfile", type=str, default=None)
    args = parser.parse_args()

    plot_single_isocontour_overlay(
        folder=args.folder,
        k0=args.k,
        nonnegative_omegas=args.omega,
        level_pct=args.level_pct,
        field=args.field,
        outdir=args.outdir,
        outfile=args.outfile,
    )


if __name__ == "__main__":
    main()


#python plotting_isocontour.py --folder n1.5 --k 3 --omega 0 1e-4 1e-3 1e-2 1e-1 --level-pct 50