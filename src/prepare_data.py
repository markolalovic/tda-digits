#!/usr/bin/env python
# -*- coding: utf-8 -*-
# prepare_data.py: Saves only the first n images of digits

import numpy as np
from sklearn.datasets import fetch_openml


N = 100

# load data from https://www.openml.org/d/554
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

np.save('../data/' + 'X_' + str(N) + '.npy', X[:N])
np.save('../data/' + 'y_' + str(N) + '.npy', y[:N])
