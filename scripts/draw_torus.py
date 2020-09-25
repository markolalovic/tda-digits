#!/usr/bin/env python
# -*- coding: utf-8 -*-
''' draw_torus.py: Draws torus and it's graph structure.
'''

from __future__ import print_function # if you are using Python 2
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

width = 110
height = 81

ellipses = np.load('../data/ellipses.npy')
rngs = np.load('../data/rngs.npy')
vpx = np.load('../data/vpx.npy')
vpy = np.load('../data/vpy.npy')


def draw_torus_surface(fig_size=20, n=30, show=False, save=False):
    t = np.linspace(0, 2 * np.pi, n)
    theta, phi = np.meshgrid(t, t)

    r, R = 0.2, 0.4
    X = (R + r * np.cos(phi)) * np.cos(theta)
    Y = (R + r * np.cos(phi)) * np.sin(theta)
    Z = r * np.sin(phi)

    fig = mlab.figure(figure=None,
        bgcolor=(1, 1, 1),
        fgcolor=None,
        engine=None,
        size=(1600, 1000))

    torus = mlab.mesh(X, Y, Z,
            colormap='Vega20b',
            resolution=10,
            )
    mlab.view(azimuth=-45,
              elevation=30,
              focalpoint='auto',
              distance=2.5,
              figure=fig)

    if show:
        mlab.show()

    if save:
        mlab.savefig('../figures/torus-surface.png')

def point_cloud(ax, ellipses, rngs):
    # set parameter range
    n = 60
    t = np.linspace(0, 2 * np.pi, n)
    t = np.array( list(t) + list(t) )

    # point cloud
    for i in range(len(ellipses)):
        ellipse, rng = ellipses[i], rngs[i]

        center, a_axis, b_axis = ellipse # ellipse parameters

        px = a_axis * np.cos(t[rng[0]:rng[1]])
        py = center + b_axis * np.sin(t[rng[0]:rng[1]])

        # set marker size for projection perspective
        marker_sizes = [height - y for y in py]
        marker_sizes = np.log( np.array(marker_sizes) ) * 100
        marker_sizes = list(marker_sizes)

        # draw the point cloud
        if i == 14:
            ax.scatter(px, py, c='steelblue', s=marker_sizes)
        else:
            ax.scatter(px, py, c='black', s=marker_sizes)

    marker_sizes = [height - y for y in vpy]
    marker_sizes = np.log( np.array(marker_sizes) ) * 100
    marker_sizes = list(marker_sizes)
    ax.scatter(vpx, vpy, c='steelblue', s=marker_sizes)

def horizontal_feature(ellipses, rngs):
    # set parameter range
    n = 60
    t = np.linspace(0, 2 * np.pi, n)
    t = np.array( list(t) + list(t) )

    i = 14 # on top of torus
    ellipse, rng = ellipses[i], rngs[i]
    center, a_axis, b_axis = ellipse # ellipse parameters
    px = a_axis * np.cos(t[rng[0]: (rng[1] - 1) ])
    py = center + b_axis * np.sin(t[rng[0]: (rng[1] - 1) ])

    # set line width for projection perspective
    marker_sizes = [height - y for y in py]
    marker_sizes = np.log( np.array(marker_sizes) ) * 2
    line_widths = np.array(marker_sizes)
    line_widths = list(line_widths)
    for i in range(len(px) - 1):
        plt.plot([px[i], px[i+1]], [py[i], py[i+1]],
                 color='steelblue',
                 solid_capstyle='round',
                 linewidth=line_widths[i])

def vertical_feature(vpx, vpy):
    # front
    marker_sizes = [height - y for y in vpy]
    marker_sizes = np.log( np.array(marker_sizes) )
    line_widths = np.array(marker_sizes) * 2
    line_widths = list(line_widths)
    for i in range(len(vpx) - 1):
        plt.plot([vpx[i], vpx[i+1]], [vpy[i], vpy[i+1]],
                 color='steelblue',
                 solid_capstyle='round',
                 linewidth=line_widths[i])

def draw_torus_features(fig_size=20, show=False, save=False):
    fig, ax = plt.subplots(
        figsize=(fig_size, 16))

    point_cloud(ax, ellipses, rngs)

    # horizontal ellipse
    horizontal_feature(ellipses, rngs)

    # vertical ellipse
    vertical_feature(vpx, vpy)

    # flip it, it looks better
    ax.set_ylim(height, 0)  # decreasing y

    # and remove axis
    ax.get_xaxis().set_visible(False) # no x-axis
    ax.get_yaxis().set_visible(False) # no y-axis

    # adjust the margins
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0, x1, y0 + 5, y1 - 5))

    ax.axis('off')  # no borders

    if show:
        plt.show()

    if save:
        plt.savefig('../figures/torus-features.png')

if __name__ == '__main__':
    draw_torus_surface(fig_size=20, n=100, show=True)
    # draw_torus_surface(fig_size=20, n=100, save=True)

    # draw_torus_features(fig_size=20, show=True)
    # draw_torus_features(fig_size=20, save=True)
