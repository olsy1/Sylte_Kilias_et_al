# Representational drift in hippocampal CA1 is geometrically preserved
https://doi.org/10.5281/zenodo.23107612

Code to reproduce: Sylte, O. C., Kilias, A., Bartos, M., & Sauer, J. F. (2026). Representational drift in hippocampal CA1 is geometrically preserved

## Data
The two-photon calcium imaging dataset are available on Zenodo: [10.5281/zenodo.23107612](https://doi.org/10.5281/zenodo.23107612).

Download the six `.h5` files into one folder.

## Requirements
Python 3.8 with numpy, scipy, pandas, h5py, matplotlib, seaborn, scikit-learn, statsmodels, pingouin, ripser, persim, dPCA (https://github.com/machenslab/dPCA) and Jupyter.

## Usage
1. Clone this repository.
2. In each notebook, set `DATA_ROOT` (currently `'DATAPATH'`) to the folder containing the `.h5` files.
3. Run the notebooks from the repository's top-level folder (they import `analysis_code`).
