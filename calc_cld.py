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

        print(file)
        print(hardness.mean(), hardness.std())
        print(np.sqrt(gnb.var_)) # Sanity check. Must be close to the SD set to the file.