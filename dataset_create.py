import pandas as pd
from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

if __name__ == "__main__":

    NUMBER_OF_FEATURES = 2
    NUMBER_OF_CLUSTERS = 2
    NUMBER_OF_SAMPLES = 500 # By each cluster
    CENTER_COORDINATES = ((1, 2), (2, 3))
    SEED = 5
    DATASET_PATH = './db/datasets/'
    IMG_PATH = './db/datasets/img/'
    CLUSTER_STD = (0.2, 0.3, 0.5, 0.8, 1.0)

    for cstd in CLUSTER_STD:
        db_name = f'DB-{cstd}'
        X, y = make_blobs(
            n_samples=[NUMBER_OF_SAMPLES for i in range(NUMBER_OF_CLUSTERS)],
            n_features=NUMBER_OF_FEATURES,
            centers = CENTER_COORDINATES,
            cluster_std=cstd,
            random_state=SEED,
        )
        df = pd.DataFrame(X, columns=['x', 'y'])
        df['target'] = y % 2


        fig, ax = plt.subplots()

        ax.scatter(df['x'], df['y'], c=df['target'], cmap='RdBu')
        ax.set_xlabel('feature x')
        ax.set_ylabel('feature y')

        fig.savefig(IMG_PATH + db_name + '.png', dpi=300)
        plt.close(fig)
        df.to_csv(DATASET_PATH + db_name + '.csv', index=False)



