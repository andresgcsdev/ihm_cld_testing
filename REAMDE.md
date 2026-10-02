# ihm_cld_testing

Testing repository for an undergraduate research project on instance hardness. It validates a from-scratch implementation of the Class Likelihood Difference (CLD) measure on synthetic binary datasets with controlled class overlap. The idea being tested: the mean and standard deviation of the per-instance CLD show whether hardness is concentrated enough in one region to justify a specialized model.

## Usage

Requires Python 3, `numpy`, `pandas` and `scikit-learn`. From the repository root:

1. `python dataset_create.py` generates the CSV files in `db/datasets/`.
2. `python calc_cld.py` prints, for each file, the mean and SD of CLD and CLD_adj, plus the fitted per-class feature stds (`sqrt(gnb.var_)`), which should match the `cluster_std` of the file.

## Datasets

Generated with `make_blobs`: 500 samples (250 per class), 2 features, centers (0, 0) and (1, 0), `random_state=5`, `cluster_std` in {0.1, 0.15, 0.2, 0.3, 0.5, 0.8}. Files are named `DB-<cluster_std>.csv` and the label column is `target`. A larger std means more overlap (gap / std goes from 10 to 1.25).

## CLD

Features are modeled as independent with `GaussianNB`, fitted on the whole dataset (priors come from the class frequencies, which are equal here). With the posteriors `P` from `predict_proba`:

- **CLD** = `(1 - (P_own - max P_other)) / 2`, in [0, 1]. Larger means harder; for two classes it equals `1 - P_own`.
- **CLD_adj** (label-free) = `(1 - (P_top1 - P_top2)) / 2`. For two classes it equals `min(CLD, 1 - CLD)`, so it never exceeds 0.5.

Posteriors are used instead of raw likelihoods so the values do not depend on the feature scale. PyHard was not used because its measures class fits other models (trees, calibrated Naive Bayes, Gower matrix) on instantiation, so CLD cannot be computed on its own.

## Results (CLD)

| `cluster_std` | gap / std | mean | SD |
|---|---|---|---|
| 0.1 | 10 | 1.5e-13 | 1.8e-12 |
| 0.15 | 6.7 | 4.6e-05 | 4.6e-04 |
| 0.2 | 5 | 0.0078 | 0.0521 |
| 0.3 | 3.3 | 0.0697 | 0.1722 |
| 0.5 | 2 | 0.2190 | 0.2513 |
| 0.8 | 1.25 | 0.3470 | 0.2310 |

The mean rises with overlap, while the SD peaks near gap / std = 2 and starts to fall, so the two must be read together.

## Next steps

- Add `cluster_std` of 1.0 and above to confirm the full rise and fall of the SD.
- Datasets with several overlap regions, clustering of the hard instances and one model per cluster (K-fold, fitting on the training folds only).

## References

- Lorena, A. C., Paiva, P. Y. A., Prudêncio, R. B. C. (2024). Trusting My Predictions: On the Value of Instance-Level Analysis. *ACM Computing Surveys*, 56(7), Article 167, 28 pages.
- Ueda, P. S. M., Rivolli, A., Lorena, A. C. (2024). An instance level analysis of classification difficulty for unlabeled data. BRACIS 2024. (CLD and CLD_adj definitions.)