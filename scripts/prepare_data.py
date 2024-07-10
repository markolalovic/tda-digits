#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' prepare_data.py: Saves n_samples = 10000 images of digits to numpy array. '''

from __future__ import print_function # if you are using Python 2
import numpy as np
from sklearn.datasets import fetch_openml
import sys

n_samples = 10000
if len(sys.argv) == 2:
    print('Setting n_samples to: %i' % (n_samples))
    n_samples = int(sys.argv[1])

# load data from https://www.openml.org/d/554
print('Loading digits...')
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

np.save('../data/' + 'X_' + str(n_samples) + '.npy', X[:n_samples])
np.save('../data/' + 'y_' + str(n_samples) + '.npy', y[:n_samples])
