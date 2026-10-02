import pathlib
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB


def cld(proba, y, classes):
    own_mask = np.asarray(y).reshape(-1, 1) == np.asarray(classes).reshape(1, -1)
    own = proba[own_mask]  # one value per row
    rival = np.where(own_mask, -1.0, proba).max(axis=1)  # best other class
    return (1 - (own - rival)) / 2

def cld_adj(proba):
    top = np.partition(proba, -2, axis=1)
    return (1 - (top[:, -1] - top[:, -2])) / 2


if __name__ == "__main__":
    csv_files = sorted(pathlib.Path('./db/datasets/').glob("*.csv"))
    for file in csv_files:
        df = pd.read_csv(file)
        X, y = df.drop('target', axis=1), df['target']

        gnb = GaussianNB().fit(X.to_numpy(), y.to_numpy())
        proba = gnb.predict_proba(X.to_numpy())
        hardness = cld(proba, y, gnb.classes_)

        with open(f"./db/CLD/DB-CLD-{str(file)[-7:-4]}.txt", "w") as f:
            f.write(str(file))
            f.write('\n')
            f.write(str(hardness))
            f.write('\n')
            f.write("Mean IHM: " + str(hardness.mean()))
            f.write('\n')
            f.write("Hardness SD: " + str(hardness.std()))
            f.write('\n')
            f.write("Sanity: " + str(np.sqrt(gnb.var_)))
