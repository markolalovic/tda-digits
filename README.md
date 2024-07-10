# Topological Features Applied to the MNIST Dataset

**Extracting topological features to enhance standard algorithms with qualitative geometric information**

## About

This repository contains source code for a tutorial on applying computational topology in machine learning. To demonstrate the use of persistent homology, we apply it to the MNIST dataset of handwritten digits.

For more information, you can read the [blog post](https://blog.lalovic.io/tda-digits) or check out the [interactive example](https://tdadigits.pages.dev/).

## How-to

### Dependencies

Ensure you have the following dependencies installed:

- Python 3
- Dionysus 2 for computing persistent homology
- Boost version 1.55 or higher for Dionysus 2
- NumPy for loading data and computing
- Scikit-learn for machine learning algorithms
- Scikit-image for image pre-processing
- Matplotlib for plotting
- Networkx for drawing graphs

### Data Preparation

To get the data, run the `prepare_data.py` script:

```bash
cd scripts
python3 prepare_data.py
```
This script downloads and saves 10,000 images of digits as NumPy arrays `X_10000.npy` and `y_10000.npy` in the data directory.

### Feature Extraction
To extract the features, run the `tda_digits.py` script:

```python
$ cd src
$ python3 tda_digits.py
```

This generates figures for digit 8, which can be found in the example directory.

### Usage Details

For detailed instructions on using the functions and classes, refer to the Jupyter notebooks: `Example.ipynb` and `Classification.ipynb` located in the scripts directory.

## References

* Aaron Adcock, Erik Carlsson, Gunnar Carlsson, "The Ring of Algebraic Functions on Persistence Bar Codes", Apr 2013.
[https://arxiv.org/abs/1304.0530](https://arxiv.org/abs/1304.0530)

* Dmitriy Morozov, "Dionysus 2 documentation".
[https://mrzv.org/software/dionysus2/](https://mrzv.org/software/dionysus2/)
