#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" drawing_module.py: A simple 2D drawing module to draw points, edges
and graphs embedded in the Euclidean plane.

Author: Marko Lalovic <marko.lalovic@yahoo.com>
License: MIT License
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches


canvas_width = 7
canvas_height = 7
image_size = 28

class Point:
    def __init__(self, x=0, y=0, time=-1):
        ''' Class Point for storing coordinates and time of a point creation.

        Args:
            x::float
                The x coordinate of the point.
            y::float
                The y coordinate of the point.
            time::float
                Time of the vertex creation.
        '''
        self.x = x
        self.y = y
        self.time = time

    def __str__(self):
        return "({}, {}, {})".format(self.x, self.y, self.time)

    def equal(self, p):
        return (self.x == p.x) and (self.y == p.y)

class PointList:
    def __init__(self, points):
        ''' PointList Class to hold a list of Point objects.'''
        self.points = points

    def __str__(self):
        return '[' + ','.join(['{!s}'.format(p) for p in self.points]) + ']'

    def __len__(self):
        return len(self.points)

    def get_points_with(self, y):
        row = []
        for point in self.points:
            if point.y == y:
                row.append(point)

        return row

    def get_nhbs(self, p1):
        nhbs = []
        left = Point(p1.x - 1, p1.y)
        diag_left = Point(p1.x - 1, p1.y - 1)
        below = Point(p1.x, p1.y - 1)
        diag_right = Point(p1.x + 1, p1.y - 1)
        for p2 in self.points:
            if p2.equal(left) or p2.equal(diag_left) or \
               p2.equal(below) or p2.equal(diag_right):
                nhbs.append(p2)

        return nhbs

    def get_emb_graph(self, show=False, save=False):
        nodes = []
        edges = []
        for y in range(image_size):
            row = self.get_points_with(y)
            for point in row:
                # add points
                point.time = y
                nodes.append(point)

                # add edges
                nhbs = self.get_nhbs(point)
                for nhb in nhbs:
                    edges.append(Edge(point, nhb, y))

        emb_graph = EmbeddedGraph(nodes, edges)
        remove_c3s(emb_graph) # remove small cycles of length 3

        if show:
            canvas = Canvas('Embedded graph')
            draw_graph(canvas, emb_graph)
            canvas.show()

        if save:
            canvas = Canvas('Embedded graph')
            draw_graph(canvas, emb_graph)
            print('Saving to ../figures: 4_embedded-graph.png')
            plt.savefig('../figures/4_embedded-graph.png')                        


        return emb_graph

    def append(self, p):
        self.points.append(p)


def remove_c3s(emb_graph):
    '''
    Removes small cycles of length 3 that are the side effect
    of graph construction method used.
    '''
    edge_of_c3 = get_edge_of_c3(emb_graph)
    while edge_of_c3:
        emb_graph.edges.remove(edge_of_c3)
        edge_of_c3 = get_edge_of_c3(emb_graph)

def get_edge_of_c3(emb_graph):
    edge_of_c3 = None
    for edge in emb_graph.edges:
        u, v = edge.p1, edge.p2
        nhbs_u = emb_graph.get_nhbs(u)
        nhbs_v = emb_graph.get_nhbs(v)
        if set(nhbs_u).intersection(nhbs_v): # common neighbour
            edge_of_c3 = edge
            break

    return edge_of_c3

class Edge:
    def __init__(self, p1, p2, time=-1):
        ''' Class Edge for storing edge points and time of the edge creation.

        Args:
            p1::Point
                Edge point.
            p2::Point
                Edge point.
            time::float
                Time of the edge creation.
        '''
        self.p1 = p1
        self.p2 = p2
        self.time = time

    def __str__(self):
        return "[{}, {}, {}]".format(str(self.p1), str(self.p2), self.time)

class EmbeddedGraph:
    def __init__(self, nodes, edges):
        ''' Graph with points embedded in the plane.'''
        self.nodes = nodes
        self.edges = edges # assumed to be a list of edges: [edge1, edge2, ...]

    def __str__(self):
        nodes = [str(point) for point in self.nodes]
        edges = [str(edge) for edge in self.edges]
        return "nodes = [{}],\n edges=[{}]".format(str(nodes), str(edges))

    @property
    def n(self):
        ''' Number of nodes in EmbeddedGraph.'''
        return len(self.nodes)

    def get_nhbs(self, u):
        nhbs = []
        for edge in self.edges:
            if u.equal(edge.p1):
                nhbs.append(edge.p2)
            if u.equal(edge.p2):
                nhbs.append(edge.p1)

        return nhbs

class Canvas:
    """ Class Canvas on which we draw the graphics."""
    def __init__(self, title, xlabel='X', ylabel='Y',
                 p1=Point(4, 2), p2=Point(image_size - 4, image_size - 4)):
        self.fig = plt.figure()
        self.fig.set_size_inches(canvas_width, canvas_height)
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

def draw_point(canvas, pt, radius=0.25, color='blue', **kwargs):
    ''' Draws a point.'''
    point = patches.Circle((pt.x, pt.y),
                        radius=radius,
                        fill=True,
                        facecolor=color,
                        **kwargs)
    canvas.ax.add_patch(point)

def draw_points(canvas, points, color='blue'):
    for point in points:
        draw_point(canvas, point, color=color)

def draw_edge(canvas, p1, p2, color='blue', **kwargs):
    """ Draws a line segment between points p1 and p2."""
    line = patches.FancyArrow(p1.x, p1.y,
                              p2.x - p1.x,
                              p2.y - p1.y,
                              color=color,
                              linewidth='3.3',
                              **kwargs)
    canvas.ax.add_patch(line)

def draw_graph(canvas, emb_graph, color='blue'):
    for pt in emb_graph.nodes:
        draw_point(canvas, pt, color=color)

    for edge in emb_graph.edges:
        draw_edge(canvas, edge.p1, edge.p2, color=color)
