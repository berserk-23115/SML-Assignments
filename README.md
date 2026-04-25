# SML Assignments

This repository contains the completed assignments for Statistical Machine Learning. Each assignment lives in its own folder and includes the corresponding Python scripts, notebooks, datasets, and generated figures or reports.

## Repository Layout

- [A1](A1/) - Gaussian classification on MNIST digits `0`, `1`, and `2`, with LDA/QDA and t-SNE visualization.
- [A2](A2/) - PCA, Fisher Discriminant Analysis, and classification experiments on MNIST digits `0`, `1`, and `2`.
- [A3](A3/) - Ridge regression, Lasso, PCA, and AdaBoost / boosting-style experiments on MNIST-derived tasks.
- [A4](A4/) - Additional MNIST classification experiments, including PCA-based preprocessing and decision-boundary plots.

## Datasets

The repository already includes the datasets required by the assignments:

- MNIST IDX files for [A1](A1/) and [A2](A2/)
- `mnist.npz` for [A3](A3/) and [A4](A4/)

## Requirements

The assignments use Python with common scientific libraries such as `numpy`, `matplotlib`, and `scikit-learn`. If you want to reproduce the experiments, activate the existing virtual environment in the workspace or install the dependencies listed in the assignment folders as needed.

## Running the Code

Run the scripts from inside the relevant assignment directory so the relative dataset paths resolve correctly.

```bash
cd A1
python 2023115_A1.py

cd ../A2
python 2023115_A2.py

cd ../A3
python 2023115_A3_Q1.py
python 2023115_A3_Q2.py
python 2023115_A3_Q3.py

cd ../A4
python 2023115_A4_Q1.py
python 2023115_A4_Q2.py
python 2023115_A4_Q3.py
```

## Notes

- Some scripts generate plots and save them back into the assignment folder.
- The notebook files in `A1/` and `A2/` appear to be alternate or supporting versions of the same work.
