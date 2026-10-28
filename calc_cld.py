import pathlib
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D


def cld(proba, y, classes):
    own_mask = np.asarray(y).reshape(-1, 1) == np.asarray(classes).reshape(1, -1)
    own = proba[own_mask]  # one value per row
    rival = np.where(own_mask, -1.0, proba).max(axis=1)  # best other class
    return (1 - (own - rival)) / 2

def cld_adj(proba):
    top = np.partition(proba, -2, axis=1)
    return (1 - (top[:, -1] - top[:, -2])) / 2


if __name__ == "__main__":

    # File Paths
    CSV_FILES_PATH = './db/datasets/'
    CLD_PATH = './db/CLD/'
    CLD_IMG_PATH = './db/CLD/img'

    # Matplotlib parameters
    pathlib.Path(CLD_IMG_PATH).mkdir(parents=True, exist_ok=True)
    cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])
    norm = Normalize(vmin=0, vmax=1)
    marker_list = ["o", "x"]
    marker_style = {
        "o": dict(s=120, linewidths=0.8, edgecolors="k"),
        "x": dict(s=120, linewidths=2.5),
    }

    csv_files = sorted(pathlib.Path(CSV_FILES_PATH).glob("*.csv"))
    for file in csv_files:
        # Calculating CLD
        df = pd.read_csv(file)
        X, y = df.drop('target', axis=1), df['target']

        gnb = GaussianNB().fit(X.to_numpy(), y.to_numpy())
        proba = gnb.predict_proba(X.to_numpy())
        hardness = cld(proba, y, gnb.classes_)

        # Saving Raw data.
        with open(f"{CLD_PATH}DB-CLD-{str(file)[-7:-4]}.txt", "w") as f:
            f.write(str(file))
            f.write('\n')
            f.write(str(hardness))
            f.write('\n')
            f.write("Mean IHM: " + str(hardness.mean()))
            f.write('\n')
            f.write("Hardness SD: " + str(hardness.std()))
            f.write('\n')
            f.write("Hardness Maximum:" + str(hardness.max()))

        # Plotting
        Xv, yv, h = X.to_numpy(), y.to_numpy(), np.asarray(hardness)
        markers = dict(zip(gnb.classes_, marker_list))

        fig, ax = plt.subplots(figsize=(12, 9))
        for cls, marker in markers.items():
            m = yv == cls
            ax.scatter(Xv[m, 0], Xv[m, 1], c=h[m], cmap=cmap, norm=norm,
                       marker=marker, **marker_style[marker])

        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="IHM (CLD)")
        ax.legend(handles=[
            Line2D([], [], marker=mk, color="gray", linestyle="None",
                   markersize=8, label=f"Class {c}")
            for c, mk in markers.items()
        ])
        ax.set_title(file.stem)

        # Saving Plot
        fig.savefig(f"{CLD_IMG_PATH}/DB-CLD-{file.stem}.png", dpi=160)
        plt.close(fig)