#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' draw_mnist.py: Draws MNIST images and their graph structure.
'''

from __future__ import print_function # if you are using Python 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import ImageGrid

def draw_mnist(what='digits', show=False, save=False):
    if what == 'digits':
        images = [mpimg.imread('../mnist-images/original-'
                                + str(i) + '.png') for i in range(10)]
    elif what == 'features':
        images = [mpimg.imread('../mnist-images/graph-'
                                + str(i) + '.png') for i in range(10)]

    fig = plt.figure(figsize=(40, 20))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(2, 5),
                     axes_pad=0.1,  # pad size between axes in inches
                     )

    for ax, image in zip(grid, images):
        ax.axis('off')  # no borders
        ax.get_xaxis().set_visible(False) # no x-axis
        ax.get_yaxis().set_visible(False) # no y-axis

        ax.imshow(image, cmap='gray')

    if show:
        plt.show()

    if save:
        if what == 'digits':
            plt.savefig('../figures/mnist-digits.png')
        elif what == 'features':
            plt.savefig('../figures/mnist-features.png')

if __name__ == '__main__':
    # draw_mnist(what='digits', show=True)
    draw_mnist(what='features', show=True)

    # draw_mnist(what='digits', save=True)
    draw_mnist(what='features', save=True)
