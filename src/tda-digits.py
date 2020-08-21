#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" tda-digits.py: Topological features applied to the digits data set.
author: Marko Lalovic <marko.lalovic@yahoo.com>
license: MIT License
"""

from __future__ import print_function   # if you are using Python 2
import dionysus as ds
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import pylab as pl
from matplotlib import collections  as mc
from sklearn.datasets import load_digits


def get_simplices(B, sweep_direction, show=False):
    ''' Constructs a simplex stream for computing the persistent homology
    using the filtration on the vertices of the graph G corresponding to
    the pixels of the image B.

    Filtration is the following. We are adding the vertices and edges to
    the graph G as we sweep across the image B in sweep_direction.
    In this way we get spatial information from the image B.

    Args:
        B::numpy.ndarray
            Assumed to be a binary image of a handwritten digit.
        sweep_direction::str
            Assumed to be 'right', 'left', 'top' or 'bottom'.
        show::bool
            Set to True, if you want to see the sweeping filtration process

    Returns:
        simplices::list
            In the form of (simplex, time) for example:
                [([0], 1), ([1], 2), ([2], 2), ([0, 1], 2), ... ([5, 6], 5)]
    '''
    m = B.shape[0]

    if sweep_direction in ['right', 'top']:
        sweep_range = range(1, m + 1)
    else: # left, top
        sweep_range = range(m - 1, -1, -1)

    simplices = []
    time = 1
    for sweep_line in sweep_range:
        C = np.zeros((m, m))

        if sweep_direction == 'right':
            C[:, 0:sweep_line] = B[:, 0:sweep_line]
        elif sweep_direction == 'left':
            C[:, sweep_line:m] = B[:, sweep_line:m]
        elif sweep_direction == 'bottom':
            C[sweep_line:m, :] = B[sweep_line:m, :]
        else:
            C[0:sweep_line, :] = B[0:sweep_line, :]

        if np.nonzero(C): # nothing to add if C is empty
            adj_mat = get_adj_mat_from(C, sweep_direction)

            if show:
                plt.imshow(C)
                plt.title('Image at sweep line: ' + str(sweep_line))
                plt.show()
                G = get_graph(adj_mat, show=True)

            # add vertices
            for i in range(adj_mat.shape[0]):
                if [i] not in [vertices[0] for vertices in simplices]:
                    simplices.append( ([i], time) )

            # add edges
            for i in range(adj_mat.shape[0]):
                for j in range(i+1, adj_mat.shape[1]):
                        if adj_mat[i, j] > 0 and \
                            [i, j] not in [vertices[0] for vertices in simplices]:
                            simplices.append( ([i, j], time) )
        time += 1 # increase the time of filtration

    return simplices

def get_betti_barcodes(simplices):
    ''' Computes the persistent homology given the simplex stream and
    persistence diagram.

    Args:
        simplices::list
            The simplex stream.
    Returns:
        intervals::dictionary
            Betti barcode intervals for dimensions 0 and 1, for example:
                {0: [[1.0, inf]], 1: [[3.0, inf], [5.0, inf]]}
    '''
    flt = ds.Filtration()
    for vertices, time in simplices:
        flt.append(ds.Simplex(vertices, time))

    flt.sort()
    homp = ds.homology_persistence(flt)
    dgms = ds.init_diagrams(homp, flt)

    intervals = {}
    intervals[0] = [] # Betti 0 barcodes
    intervals[1] = [] # Betti 1 barcodes
    for i, dgm in enumerate(dgms):
        print('Betti', i)
        for pt in dgm:
            intervals[i].append([pt.birth, pt.death])
            print('(', pt.birth, ', ', pt.death, ')', sep='')
        print()

    return intervals

def get_adj_mat_from(B, sweep_direction):
    ''' Constructs a graph G in the form of adjacency matrix given the image B in the
    following way. Graph construction is the following. We treat the pixels as
    vertices and add edges between adjacent pixels (including diagonal neighbours).

    Args:
        B::numpy.ndarray
            Assumed to be a binary image of a handwritten digit.
        sweep_direction::str
            Assumed to be 'right', 'left', 'top' or 'bottom'.

    Returns:
        adj_mat::numpy.ndarray
            Adjacency matrix of the graph G.
    '''
    ics, jcs = np.nonzero(B)
    n = len(ics)
    adj_mat = np.zeros((n, n))
    # TODO: this is for sweep_direction='top', add other 3 sweep directions
    for i in range(n):
        for j in range(n):
            if neighbours( (ics[i], jcs[i]), (ics[j], jcs[j]) ):
                adj_mat[i, j] = 1
    return adj_mat

def get_graph(adj_mat, show=False):
    ''' Transforms the given adjacency matrix representation of a graph to
    the (V, E) graph representation.

    Args:
        adj_mat::numpy.ndarray
            Graphs adjacency matrix.
        show::bool
            Set to true, if you want to see the drawing of the graph G.

    Returns:
        G::networkx.Graph()
            Graph object.
    '''
    G = nx.Graph()

    # add vertices
    G.add_nodes_from(list(range(adj_mat.shape[0])))

    # add edges
    for i in range(adj_mat.shape[0]):
        for j in range(i+1, adj_mat.shape[1]):
                if adj_mat[i, j] > 0:
                    G.add_edge(i, j)

    if show:
        plt.plot()
        pos = nx.fruchterman_reingold_layout(G, scale=2)
        nx.draw(G,
                pos,
                font_size=10,
                node_color='steelblue',
                with_labels=True,
                font_weight='bold')
        plt.show()

    return G

def neighbours(p, q):
    ''' Tests if pixels from some image are neighbours including diagonally.

    Args:
        p::tuple
            Coordinate of a pixel for which we calculate the neighbourhood.
        q::tuple
            Coordinate of a pixel for which we test if it is in the neighbourhood.

    Returns:
        bool
            If pixel q is in the neighbourhood of pixel p.
    '''
    list1 = list(range(p[0] - 1, p[0] + 2))
    list2 = list(range(p[1] - 1, p[1] + 2))
    nbd = [(i, j) for i in list1 for j in list2]
    return q in nbd

def get_example(show=False):
    ''' Example image of handwritten image of number 8. Eroded such that the
    width of the curve is 1 pixel.

    Args:
        show::bool
            Set to True, if you want to see the image.

    Returns:
        A::numpy.ndarray
            The image.
    '''
    coords = [[0, 2], [1, 1], [1, 3], [2, 2], [3, 1], [3, 3], [4, 2]]
    m = 5
    A = np.zeros((m, m))
    for coord in coords:
        A[coord[0], coord[1]] = 1

    if show:
        plt.imshow(A)
        plt.title('Example image of number 8')
        plt.show()

    return A

def get_number(n=8, show=False):
    ''' To load the image of a handwritten digit from Scikit-learn dataset of
    1797 8x8 images.

    Args:
        n::int
            The number of the handwritten digit image we want.
        show::bool
            Set True, if you want to see the image.

    Returns:
        A::numpy.ndarray
            The image of the handwritten digit.
    '''

    digits = load_digits()
    A = digits.images[n]

    if show:
        plt.imshow(A)
        plt.title('Handwritten digit of number ' + str(n%10))
        plt.show()

    return A

def binarize(A, threshold=0, show=False):
    ''' Produce the binary image B by thresholding the input image A.

    Args:
        A::numpy.ndarray
            The image of the handwritten digit.
        threshold::float
            The threshold for binarization. Default is mean(A)/2.
        show::bool
            Set True, if you want to see the result.

    Returns:
        B::numpy.ndarray
            The binary image of the handwritten digit.
    '''

    B = np.copy(A)
    if threshold == 0:
        threshold = np.mean(A)/2
    B[B <= threshold] = 0
    B[B > threshold] = 1

    if show:
        plt.imshow(B)
        plt.title('Binary image')
        plt.show()

    return B

def draw_betti_barcodes(intervals, xlim):
    ''' Draws the Betti barcodes for dimensions 0 and 1.

    Args:
        intervals::dictionary
            Betti barcode intervals for dimensions 0 and 1, for example:
                {0: [[1.0, inf]], 1: [[3.0, inf], [5.0, inf]]}
        xlim::float
            The plots limit for x axis.
    '''
    lines = {}
    lines[0] = []
    lines[1] = []
    for dim in [0, 1]:
        for i in range(len(intervals[dim])):
            a, b = intervals[dim][i]
            if str(b) == 'inf':
                b = xlim
            lines[dim].append([(a, i), (b, i)]) # draw it at height i

    fig, axs = pl.subplots(2, 1, constrained_layout=True)
    fig.canvas.draw()


    xticks = list(range(xlim + 1))
    xlabels = [str(i) for i in range(xlim)] + ['inf']

    for dim in [0, 1]:
        lc = mc.LineCollection(lines[dim], linewidths=20)

        axs[dim].set_aspect('auto')
        axs[dim].set_xlim([-.25, xlim + .25])
        axs[dim].set_xbound(lower=-.25, upper=xlim + .25)

        axs[dim].add_collection(lc)

        axs[dim].set_xticks(xticks)
        axs[dim].set_xticklabels(xlabels)

        yticks = list(range(len(lines[dim])))
        ylabels = [str(i) for i in yticks]
        axs[dim].set_yticks(yticks)
        axs[dim].set_yticklabels(ylabels)

        axs[dim].xaxis.grid(True)
        axs[dim].margins(0.1)
        axs[dim].set_title('Betti ' + str(dim) + ' Barcode');

    pl.show()


if __name__ == "__main__":
    use_example = True
    sweep_direction = 'top'

    if use_example:
        B = get_example(show=True)
    else:
        A = get_number(n=8, show=True)
        B = binarize(A, show=True)

    m = B.shape[0]
    adj_mat = get_adj_mat_from(B, sweep_direction)
    G = get_graph(adj_mat, show=True)

    simplices = get_simplices(B, sweep_direction) # show=True, to see the filtration
    intervals = get_betti_barcodes(simplices)
    draw_betti_barcodes(intervals, 2*m)

    # TODO:
    # for sweep_direction in ['right', 'left', 'top', 'bottom']:
    #   extract features:
    #   4 sweeps and Betti 0, Betti 1 barcodes = 8 features vectors per image
