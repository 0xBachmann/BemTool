#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
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


def load_field(path: Path, field: str = "utot") -> pd.DataFrame:
    df = pd.read_csv(path)

    real_col = f"real_{field}"
    imag_col = f"imag_{field}"

    if "is_skipped" in df.columns:
        df = df[df["is_skipped"] == 0].copy()

    df["u"] = df[real_col].to_numpy() + 1j * df[imag_col].to_numpy()
    return df[["x", "y", "u"]].copy()


def find_path(index_df: pd.DataFrame, k0: float, omega: float, tol=1e-14):
    sel = index_df[
        np.isclose(index_df["k0"], k0, atol=tol, rtol=0.0)
        & np.isclose(index_df["omega"], omega, atol=tol, rtol=0.0)
    ]
    if len(sel) == 0:
        return None
    return Path(sel.iloc[0]["path"])


def common_merge(df_p: pd.DataFrame, df_m: pd.DataFrame, df_0: pd.DataFrame) -> pd.DataFrame:
    a = df_p.rename(columns={"u": "u_p"})
    b = df_m.rename(columns={"u": "u_m"})
    c = df_0.rename(columns={"u": "u_0"})

    merged = pd.merge(a, b, on=["x", "y"], how="inner")
    merged = pd.merge(merged, c, on=["x", "y"], how="inner")
    return merged


def rms(z: np.ndarray) -> float:
    return np.sqrt(np.mean(np.abs(z) ** 2))


def maxabs(z: np.ndarray) -> float:
    return np.max(np.abs(z))


def collect_scalings(folder: Path, ks=None, field="utot") -> pd.DataFrame:
    index_df = scan_files(folder)
    if len(index_df) == 0:
        raise RuntimeError(f"No matching files found in {folder}")

    if ks is None or len(ks) == 0:
        ks = sorted(index_df["k0"].unique())

    rows = []

    for k0 in ks:
        path0 = find_path(index_df, k0, 0.0)
        if path0 is None:
            print(f"Skipping k={k0:g}: missing Omega=0 file")
            continue

        df0 = load_field(path0, field=field)

        omegas_pos = sorted(
            w for w in index_df.loc[np.isclose(index_df["k0"], k0), "omega"].unique() if w > 0
        )

        for omega in omegas_pos:
            pathp = find_path(index_df, k0, +omega)
            pathm = find_path(index_df, k0, -omega)

            if pathp is None or pathm is None:
                print(f"Skipping k={k0:g}, omega={omega:.1e}: missing ±Omega pair")
                continue

            dfp = load_field(pathp, field=field)
            dfm = load_field(pathm, field=field)

            merged = common_merge(dfp, dfm, df0)
            if len(merged) == 0:
                print(f"Skipping k={k0:g}, omega={omega:.1e}: no common grid points")
                continue

            up = merged["u_p"].to_numpy()
            um = merged["u_m"].to_numpy()
            u0 = merged["u_0"].to_numpy()

            uodd = 0.5 * (up - um)
            ueven_corr = 0.5 * (up + um) - u0

            denom = maxabs(u0)
            if denom == 0.0:
                denom = 1.0

            rows.append(
                {
                    "k0": k0,
                    "omega": omega,
                    "npts": len(merged),
                    "odd_rms_norm": rms(uodd) / denom,
                    "odd_max_norm": maxabs(uodd) / denom,
                    "even_rms_norm": rms(ueven_corr) / denom,
                    "even_max_norm": maxabs(ueven_corr) / denom,
                }
            )

    return pd.DataFrame(rows).sort_values(["k0", "omega"])


def make_k_colors(k_values):
    k_values = sorted(k_values)
    if len(k_values) == 0:
        return {}

    # truncated viridis: avoid bright yellow end
    base = cm.get_cmap("viridis")
    vals = np.linspace(0.08, 0.78, max(len(k_values), 2))
    palette = [base(v) for v in vals]

    if len(k_values) == 1:
        return {k_values[0]: palette[0]}

    return {k: palette[i] for i, k in enumerate(k_values)}


def add_reference_slope(ax, x, y_anchor, power, label):
    x = np.asarray(x)
    x = x[x > 0]
    if len(x) == 0:
        return
    x0 = x[0]
    y = y_anchor * (x / x0) ** power
    ax.loglog(x, y, "k:", linewidth=1.5, label=label)


def plot_quantity(df: pd.DataFrame, quantity: str, ylabel: str, slope_power: int, slope_label: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    ks = sorted(df["k0"].unique())
    colors = make_k_colors(ks)

    for k0 in ks:
        sub = df[df["k0"] == k0].sort_values("omega")
        ax.loglog(
            sub["omega"],
            sub[quantity],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            color=colors[k0],
            label=fr"$k={k0:g}$",
        )

    positive = df[df[quantity] > 0].sort_values("omega")
    if len(positive) > 0:
        y_anchor = positive.iloc[0][quantity]
        add_reference_slope(
            ax,
            positive["omega"].to_numpy(),
            y_anchor=y_anchor,
            power=slope_power,
            label=slope_label,
        )

    ax.set_xlabel(r"$\Omega$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":")
    ax.legend()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=Path("."))
    parser.add_argument("--field", type=str, default="utot", choices=["uinc", "usca", "utr", "utot"])
    parser.add_argument("--k", type=float, nargs="*", default=None)
    parser.add_argument("--csv-out", type=Path, default=Path("evenodd_scalings.csv"))
    parser.add_argument("--odd-plot-out", type=Path, default=Path("odd_scaling_all_k.pdf"))
    parser.add_argument("--even-plot-out", type=Path, default=Path("even_scaling_all_k.pdf"))
    args = parser.parse_args()

    df = collect_scalings(args.folder, ks=args.k, field=args.field)
    if len(df) == 0:
        raise RuntimeError("No valid k/omega data collected.")

    print(df.to_string(index=False))
    df.to_csv(args.csv_out, index=False)

    plot_quantity(
        df=df,
        quantity="odd_rms_norm",
        ylabel=r"$\left\lVert\frac{u_{\Omega}-u_{-\Omega}}{2}\right\rVert_{\mathcal{L}^2_0}$",
        slope_power=1,
        slope_label=r"$\mathcal{O}(\Omega)$",
        outpath=Path("plots") / args.odd_plot_out,
    )

    plot_quantity(
        df=df,
        quantity="even_rms_norm",
        ylabel=r"$\left\lVert\frac{u_{\Omega}+u_{-\Omega}}{2}-u_0\right\rVert_{\mathcal{L}^2_0}$",
        slope_power=2,
        slope_label=r"$\mathcal{O}(\Omega^2)$",
        outpath=Path("plots") / args.even_plot_out,
    )


if __name__ == "__main__":
    main()

# python plotting_scaling.py --folder n1.5 --k 1 2 3 4 5 6 7 8 9 10