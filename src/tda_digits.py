#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" tda_digits.py: Topological features applied to the digits data set.

Author: Marko Lalovic <marko.lalovic@yahoo.com>
License: MIT License
"""

from __future__ import print_function   # if you are using Python 2
import dionysus as ds
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib import collections  as mc
from sklearn.datasets import load_digits
from skimage.morphology import skeletonize

from drawing_module import *

CANVAS_WIDTH = 10
CANVAS_HEIGHT = 10
M = 28 # MNIST handwritten digits images are 28x28

def get_simplices(emb_graph, show=False):
    ''' Constructs a simplex stream for computing the persistent homology
    using the filtration on the vertices of the graph G corresponding to
    the pixels of the image B.

    Filtration is the following. We are adding the vertices and edges to
    the graph G as we sweep across the image B in sweep_direction.
    In this way we get spatial information from the image B.'''
    simplices = []

    number_of = {}
    for i in range(emb_graph.n):
        node = emb_graph.nodes[i]
        simplices.append( ([i], node.time) )
        number_of[node] = i

    for edge in emb_graph.edges:
        u = number_of[edge.p1]
        v = number_of[edge.p2]
        simplices.append( ([u, v], edge.time) )

    if show:
        graph_nx = nx.Graph()

        for simplex, time in simplices:
            if len(simplex) == 1:
                graph_nx.add_node(simplex[0])
            else:
                edge = (simplex[0], simplex[1])
                graph_nx.add_edge(*edge)

        pos = nx.spring_layout(graph_nx)
        # pos = nx.fruchterman_reingold_layout(graph_nx)
        nx.draw(graph_nx,
                pos,
                font_size=10,
                node_color='steelblue',
                with_labels=True)
        plt.show()

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
    for simplex, time in simplices:
        flt.append(ds.Simplex(simplex, time))

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
    ''' Constructs a graph G in the form of adjacency matrix given the points.
    Graph construction is the following. We treat the pixels as vertices and
    add edges between adjacent pixels (including diagonal neighbours).

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

def get_image(n=17, show=False):
    ''' Loads the image of a handwritten digit from MNIST dataset.

    Args:
        n::int
            The number of the handwritten digit image we want,
            e.g. n = 17 for image of number 8.
        show::bool
            Set True, if you want to see the loaded image.

    Returns:
        image::numpy.ndarray
            Array of size MxM.
    '''
    X = np.load('../data/X_100.npy', allow_pickle=True)
    y = np.load('../data/y_100.npy', allow_pickle=True)
    image = X[n]
    image = image.reshape((M, M))

    if show:
        plt.imshow(image, cmap='gray')
        plt.title('Original image of number ' + y[n])
        plt.show()

    return image

def get_binary_image(image, threshold=0, show=False):
    ''' Produce the binary image B by thresholding the input image A.

    Args:
        A::numpy.ndarray
            The image of the handwritten digit.
        threshold::float
            The threshold for binarization. Default is mean(A)/2.
        show::bool
            Set True, if you want to see the binary image.

    Returns:
        B::numpy.ndarray
            The binary image of the handwritten digit.
    '''

    if threshold == 0:
        threshold = np.mean(image)
    image[image <= threshold] = 0
    image[image > threshold] = 1

    if show:
        plt.imshow(image)
        plt.title('Binary image')
        plt.show()

    return image

def get_skeleton(binary_image, show=False):
    ''' Reduces the binary image to 1 pixel width to expose its topology
    using the Zhang-Suen Thinning algorithm.

    Args:
        image::numpy.ndarray
            Array of binary image.
        show::bool
            Set True, if you want to see the skeleton of the image.

    Returns:
        skeleton::numpy.ndarray
            Array of the skeleton of the input image.
    '''
    binary_image = binary_image.astype(bool)
    skeleton = skeletonize(binary_image)
    skeleton = skeleton.astype(int)

    if show:
        plt.imshow(skeleton)
        plt.title('Skeleton of the image')
        plt.show()

    return skeleton

def get_points(skeleton, sweep_direction='top', show=False):
    ''' Transforms the pixels of skeleton to points.

    Args:
        skeleton::numpy.ndarray
            Array of skeleton.
        sweep_direction::str
            Assumed to be 'right', 'left', 'top' or 'bottom'.
        show::bool
            Set True, if you want to see the points.

    Returns:
        points::list
            The points of the skeleton of the handwritten digit of a number
            as a list of Point objects.
    '''
    # transpose and flip the image matrix according to the sweep directions
    if sweep_direction == 'top':
        skeleton = np.flipud(skeleton)
        skeleton = skeleton.transpose()
    elif sweep_direction == 'bottom':
        skeleton = skeleton.transpose()
    elif sweep_direction == 'right':
        skeleton = np.flipud(skeleton)

    coords = skeleton.nonzero()
    coords = list(zip(coords[0], coords[1]))
    points = [Point(coords[i][0], coords[i][1]) for i in range(len(coords))]

    if show:
        canvas = Canvas('Points of the skeleton')
        draw_points(canvas, points)
        canvas.show()

    return points

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
    sweep_direction='top'

    image = get_image(n=17, show=True)
    binary_image = get_binary_image(image, show=True)
    skeleton = get_skeleton(binary_image, show=True)
    points = get_points(skeleton, sweep_direction, show=True)

    point_list = PointList(points)
    emb_graph = point_list.get_emb_graph()
    canvas = Canvas('Embedded graph')
    draw_graph(canvas, emb_graph)
    canvas.show()

    simplices = get_simplices(emb_graph, show=True)
    intervals = get_betti_barcodes(simplices)
    draw_betti_barcodes(intervals, M)

    # TODO:
    # for sweep_direction in ['right', 'left', 'top', 'bottom']:
    #   extract features:
    #    from Betti 0 and Betti 1 barcodes
    # 4 sweeps * 2 dimensions = 8 features vectors per image
