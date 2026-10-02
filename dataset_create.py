import pandas as pd
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

if __name__ == "__main__":

    NUMBER_OF_FEATURES = 2
    SEED = 5
    DATASET_PATH = './db/datasets/'
    IMG_PATH = './db/datasets/img/'
    CLUSTER_STD = (0.1, 0.15, 0.2, 0.3, 0.5, 0.8)

    for cstd in CLUSTER_STD:
        db_name = f'DB-{cstd}'
        X, y = make_blobs(
            n_samples=(250, 250),
            n_features=NUMBER_OF_FEATURES,
            centers = ((0, 0), (1, 0)),
            cluster_std=cstd,
            random_state=SEED,
        )
        df = pd.DataFrame(X, columns=['x', 'y'])
        df['target'] = y

        fig, ax = plt.subplots()

        ax.scatter(df['x'], df['y'], c=df['target'], cmap='RdBu')
        ax.set_xlabel('feature x')
        ax.set_ylabel('feature y')

        fig.savefig(IMG_PATH + db_name + '.png', dpi=300)
        df.to_csv(DATASET_PATH + db_name + '.csv', index=False)



