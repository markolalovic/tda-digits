#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function   # if you are using Python 2
import dionysus as ds
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math


CANVAS_WIDTH = 10
CANVAS_HEIGHT = 10
M = 28 # MNIST handwritten digits images are 28x28

class Point:
    ''' Class Point for storing coordinates and label of a point.

    Args:
        x::float
            The x coordinate of the point.
        y::float
            The y coordinate of the point.
        label::str
            Should be: 'E' for edge point and 'V' for vertex point.
    '''
    def __init__(self, x=0, y=0, label=''):
        self.__x = x
        self.__y = y
        if label not in ('E', 'V', ''):
            raise ValueError ("Label must be 'E' or 'V'")
        self.__label = label

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def label(self):
        return self.__label

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y, self.label)

class PointList:
    def __init__(self, points):
        ''' PointList Class to hold a list of Point objects.'''
        if isinstance(points[0], Point):
            self.points = points
        else:
            raise ValueError("Args must be a list of Points.")

    def __str__(self):
        return '[' + ','.join(['{!s}'.format(p) for p in self.points]) + ']'

class Canvas:
    """ Class Canvas on which we draw the graphics."""
    def __init__(self, title, xlabel='X', ylabel='Y',
                 p1=Point(-2, -2), p2=Point(M, M)):
        self.fig = plt.figure()
        self.fig.set_size_inches(CANVAS_WIDTH, CANVAS_HEIGHT)
        self.ax = self.fig.add_subplot(111, aspect='equal')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(range(p1.x, p2.x))
        plt.yticks(range(p1.y, p2.y))
        self.ax.grid(True)
        self.ax.set_xlim([p1.x, p2.x])
        self.ax.set_ylim([p1.y, p2.y])

    def show(self):
        """ Show the canvas, displaying any graphics drawn on it."""
        plt.show()
        plt.clf()
        plt.cla()
        plt.close()


def draw_point(canvas, pt, radius=0.4, **kwargs):
    ''' Draws a point.'''
    point = patches.Circle((pt.x, pt.y), radius, **kwargs)
    canvas.ax.add_patch(point)

def get_distance(p1, p2):
    ''' Euclidean distance between p1, p2.'''
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def draw_ball(canvas, x, y, radius=5, color='blue', **kwargs):
    """ Draws a ball."""
    circle = patches.Circle((x, y),
                        radius,
                        fill=False,
                        edgecolor=color,
                        linestyle='dotted',
                        linewidth='2.2',
                        **kwargs)
    canvas.ax.add_patch(circle)

def draw_edge(canvas, p1, p2, color='blue', **kwargs):
    """ Draws a line segment between points p1 and p2."""
    line = patches.FancyArrow(p1.x, p1.y,
                              p2.x - p1.x,
                              p2.y - p1.y,
                              color=color,
                              **kwargs)
    canvas.ax.add_patch(line)


if __name__ == "__main__":
    canvas = Canvas('Number 7')

    diagonal = [Point(i, i) for i in range(1, M, 1)]
    top = [Point(i, M - 1) for i in range(1, M, 1)]
    middle = [Point(i, 14) for i in range(7, 20, 1)]
    pl = PointList(diagonal + top + middle)

    for point in pl.points:
        draw_point(canvas, point)

    p1 = Point(15, 15)
    p2 = Point(20, 20)
    draw_edge(canvas, p1, p2, 'red')
    draw_ball(canvas, 15, 15, 5, 'green')

    plt.show()

    # TODO: add noise
    # TODO: implement Aanjaneya's reconstruction algorithm
    r = 1
    delta = 0.5
