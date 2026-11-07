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
    PROJECT_PATH = pathlib.Path(__file__).resolve().parent
    CSV_FILES_PATH = PROJECT_PATH / 'db' / 'datasets'
    CLD_PATH = PROJECT_PATH / 'db' / 'CLD'
    CLD_IMG_PATH = CLD_PATH / 'img'
    CSV_IHM_FILES_PATH = CSV_FILES_PATH / 'IHM'

    # Matplotlib parameters
    CLD_PATH.mkdir(parents=True, exist_ok=True)
    CLD_IMG_PATH.mkdir(parents=True, exist_ok=True)
    CSV_IHM_FILES_PATH.mkdir(parents=True, exist_ok=True)
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
        hardness_actual = cld(proba, y, gnb.classes_)
        hardness_predicted = cld_adj(proba)

        # Saving Raw data.
        with open(CLD_PATH / f"CLD-{file.stem}.txt", "w") as f:
            f.write(str(file))
            f.write('\n')
            f.write(str(hardness_actual))
            f.write('\n')
            f.write("Mean IHM: " + str(hardness_actual.mean()))
            f.write('\n')
            f.write("Hardness SD: " + str(hardness_actual.std()))
            f.write('\n')
            f.write("Hardness Maximum:" + str(hardness_actual.max()))

        with open(CLD_PATH / f"CLD_adj-{file.stem}.txt", "w") as f:
            f.write(str(file))
            f.write('\n')
            f.write(str(hardness_actual))
            f.write('\n')
            f.write("Mean IHM: " + str(hardness_predicted.mean()))
            f.write('\n')
            f.write("Hardness SD: " + str(hardness_predicted.std()))
            f.write('\n')
            f.write("Hardness Maximum:" + str(hardness_predicted.max()))

        df["ihm"] = hardness_actual
        df.to_csv(CSV_IHM_FILES_PATH / f"{file.stem}-ihm.csv", index=False)

        # Plotting
        Xv, yv, h = X.to_numpy(), y.to_numpy(), np.asarray(hardness_actual)
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
        fig.savefig(CLD_IMG_PATH / f"DB-CLD-{file.stem}.png", dpi=160)
        plt.close(fig)

        # Plotting predicted hardness
        h_predicted = np.asarray(hardness_predicted)
        predicted_norm = Normalize(vmin=0, vmax=0.5)
        fig, ax = plt.subplots(figsize=(12, 9))
        for cls, marker in markers.items():
            m = yv == cls
            ax.scatter(Xv[m, 0], Xv[m, 1], c=h_predicted[m], cmap=cmap, norm=predicted_norm,
                       marker=marker, **marker_style[marker])

        fig.colorbar(ScalarMappable(norm=predicted_norm, cmap=cmap), ax=ax,
                     label="Predicted hardness (CLD adjusted)")
        ax.legend(handles=[
            Line2D([], [], marker=mk, color="gray", linestyle="None",
                   markersize=8, label=f"Class {c}")
            for c, mk in markers.items()
        ])
        ax.set_title(f"{file.stem} - predicted hardness")

        fig.savefig(CLD_IMG_PATH / f"DB-CLD-adj-{file.stem}.png", dpi=160)
        plt.close(fig)