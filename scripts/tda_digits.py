#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' tda_digits.py: Topological features applied to the digits data set.

Author: Marko Lalovic <marko.lalovic@yahoo.com>
License: MIT License
'''

from __future__ import print_function # if you are using Python 2
import dionysus as ds
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from matplotlib import collections  as mc
from sklearn.datasets import load_digits
from skimage.morphology import skeletonize
import math
import sys

from drawing_module import *

n_samples = 10000 # number of loaded MNIST handwritten digits
if len(sys.argv) == 2:
    print('Setting n_samples to: %i' % (n_samples))
    n_samples = int(sys.argv[1])

n_features = 32 # number of features per image
image_size = 28 # MNIST handwritten digits images are 28x28
data_X = np.load('../data/X_' + str(n_samples) + '.npy', allow_pickle=True)
data_y = np.load('../data/y_'  + str(n_samples) + '.npy', allow_pickle=True)

def get_simplices(emb_graph, show=False, save=False):
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

        nx.draw_kamada_kawai(graph_nx,
                font_size=10,
                node_color='steelblue',
                with_labels=True)
        plt.show()

    if save:
        fig = plt.figure()
        graph_nx = nx.Graph()

        for simplex, time in simplices:
            if len(simplex) == 1:
                graph_nx.add_node(simplex[0])
            else:
                edge = (simplex[0], simplex[1])
                graph_nx.add_edge(*edge)

        nx.draw_kamada_kawai(graph_nx,
                font_size=10,
                node_color='steelblue',
                with_labels=True)

        print('Saving to ../figures: 5_simplices.png')
        plt.savefig('../figures/5_simplices.png')

    return simplices

def get_betti_barcodes(simplices, show=False):
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
        if show:
            print('Betti', i)
        for pt in dgm:
            intervals[i].append([pt.birth, pt.death])
            if show:
                print('(', pt.birth, ', ', pt.death, ')', sep='')
        if show:
            print()

    return intervals

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

def get_image(n=17, show=False, save=False):
    ''' Loads the image of a handwritten digit from MNIST dataset.

    Args:
        n::int
            The number of the handwritten digit image we want,
            e.g. n = 17 for image of number 8.
        show::bool
            Set True, if you want to see the loaded image.

    Returns:
        image::numpy.ndarray
            Array of size image_size x image_size.
    '''
    image = data_X[n]
    image = image.reshape((image_size, image_size))

    if show:
        inverted = np.array(list(map(lambda x: 255.0 - x , image)))
        plt.imshow(inverted, cmap='gray')
        plt.title('Original image of number ' + data_y[n])
        plt.show()

    if save:
        inverted = np.array(list(map(lambda x: 255.0 - x , image)))
        plt.imshow(inverted, cmap='gray')
        plt.title('Original image of number ' + data_y[n])
        print('Saving to ../figures: 0_original-image.png')
        plt.savefig('../figures/0_original-image.png')

    return image

def get_binary_image(image, threshold=0, show=False, save=False):
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
        plt.imshow(~image.astype(bool), cmap='gray')
        plt.title('Binary image')
        plt.show()

    if save:
        plt.imshow(~image.astype(bool), cmap='gray')
        plt.title('Binary image')
        print('Saving to ../figures: 1_binary-image.png')
        plt.savefig('../figures/1_binary-image.png')

    return image

def get_skeleton(binary_image, show=False, save=False):
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
        plt.imshow(~skeleton.astype(bool), cmap='gray')
        plt.title('Skeleton of the image')
        plt.show()

    if save:
        plt.imshow(~skeleton.astype(bool), cmap='gray')
        plt.title('Skeleton of the image')
        print('Saving to ../figures: 2_skeleton.png')
        plt.savefig('../figures/2_skeleton.png')

    return skeleton

def get_points(skeleton, sweep_direction='top', show=False, save=False):
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

    if save:
        canvas = Canvas('Points of the skeleton')
        draw_points(canvas, points)
        print('Saving to ../figures: 3_points.png')
        plt.savefig('../figures/3_points.png')

    return points

def draw_betti_barcodes(intervals, xlim, show=False, save=False):
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

    if show:
        pl.show()
    if save:
        print('Saving to ../figures: 6_betti-barcodes.png')
        pl.savefig('../figures/6_betti-barcodes.png')

def example(n=17, show=False, save=False):
    ''' Shows example of feature extraction for image of number 8.'''

    sweep_direction='top'

    image = get_image(n, show, save)
    binary_image = get_binary_image(image, 0, show, save)
    skeleton = get_skeleton(binary_image, show, save)

    points = get_points(skeleton, sweep_direction, show, save)
    point_list = PointList(points)
    emb_graph = point_list.get_emb_graph(show, save)
    simplices = get_simplices(emb_graph, show, save)

    intervals = get_betti_barcodes(simplices)
    draw_betti_barcodes(intervals, image_size, show, save)

    f0 = extract_features(intervals[0])
    f1 = extract_features(intervals[1])
    features = f0 + f1

    print('Extracted features: ')
    print(features)

def extract_features(intervals):
    ''' Extracts 4 features:
        	sum_i { x_i * (y_i - x_i) }
        	sum_i { (y_max - y_i) * (y_i - x_i) }
        	sum_i { x_i^2 * (y_i - x_i)^4 }
        	sum_i { (y_max - y_i)^2 * (y_i - x_i)^4 }
    From the barcode intervals:
        (x1, y1), (x2, y2), ..., (x_n, y_n);

    Args:
        intervals::list
            Betti barcode intervals, for example: [[3.0, inf], [5.0, inf]]
    Returns:
        features::list
            The 4 computed features.
    '''
    xs = []
    ys = []
    for interval in intervals:
        x = interval[0]
        y = interval[1]
        if str(y) == 'inf': # replace the inf with image_size
            y = image_size
        xs.append(x)
        ys.append(y)

    f1, f2, f3, f4 = 0., 0., 0., 0.
    for i in range(len(xs)):
        f1 += xs[i] * (ys[i] - xs[i])
        f2 += (max(ys) - ys[i]) * (ys[i] - xs[i])
        f3 += math.pow(xs[i], 2) * math.pow(ys[i] - xs[i], 4)
        f4 += math.pow(max(ys) - ys[i], 2) * math.pow(ys[i] - xs[i], 4)

    return [f1, f2, f3, f4]

def extract_all_features(n):
    ''' Extracts features of nth image, all together:
            4 sweeps * (2 barcodes * 4 features) = 32 features

        Args:
            n::int
                The number of the handwritten digit image, for example:
                    n = 17 for image of number 8.

        Returns:
            all_features::list
                The 32 computed features.
    '''
    image = get_image(n)
    binary_image = get_binary_image(image)
    skeleton = get_skeleton(binary_image)

    all_features = []
    for sweep_direction in ['right', 'left', 'top', 'bottom']:
        points = get_points(skeleton, sweep_direction)
        point_list = PointList(points)
        emb_graph = point_list.get_emb_graph()
        simplices = get_simplices(emb_graph)
        intervals = get_betti_barcodes(simplices)

        f0 = extract_features(intervals[0])
        f1 = extract_features(intervals[1])
        features = f0 + f1
        all_features += features

    return all_features

def save_features_matrix(n_samples=1000):
    ''' Saves the feature matrix of shape (n_samples, n_features) to ../data
    directory.

    Args:
        n_samples::int
            Number of samples of handwritten digit images.
    '''
    df = np.zeros((n_samples, n_features))

    # extract all features of each image and save it to an array
    print('Extracting all features...')
    for n in range(n_samples):
        df[n] = extract_all_features(n)

    print('Features extracted.')
    np.save('../data/' + 'features_' + str(n_samples) + '.npy', df)

if __name__ == '__main__':
    example(save=True)
    save_features_matrix()
