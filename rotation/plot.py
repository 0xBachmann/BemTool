import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys



choices=["uinc", "usca", "utr", "utot"]

csv_file = sys.argv[1]

field = "utot"

df = pd.read_csv(csv_file)


real_col = f"real_{field}"
imag_col = f"imag_{field}"
abs_col = f"abs_{field}"

# Some fields may not have abs_* stored in the CSV, so compute it if needed
if abs_col not in df.columns:
    df[abs_col] = (df[real_col] ** 2 + df[imag_col] ** 2) ** 0.5

x = df["x"]
y = df["y"]


fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

plots = [
    (real_col, "Real part"),
    (imag_col, "Imaginary part"),
    (abs_col, "Absolute value"),
]

for i in range(3):
    ax = axes[i]
    col = plots[i][0]
    sc = ax.scatter(x, y, c=df[col], s=20)
    ax.set_title(f"{plots[i][1]} of {field}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label=col)



plt.show()
