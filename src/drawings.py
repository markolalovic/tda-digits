#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function   # if you are using Python 2
import dionysus as ds
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import copy
import networkx as nx


CANVAS_WIDTH = 10
CANVAS_HEIGHT = 10
M = 28 # MNIST handwritten digits images are 28x28

class EmbeddedGraph:
    def __init__(self, nodes, edges):
        ''' Graph with points embedded in the plane.'''
        self.nodes = PointList(nodes)
        self.edges = [PointList(edge) for edge in edges]

    def __str__(self):
        points = [str(point) for point in self.nodes.points]
        edges = [str(edge) for edge in self.edges]
        components = [str(cmpt_emb_G) for cmpt_emb_G in self.components.values()]

        return "nodes: {}\nedges: {}\ncomponents: {}".format(
            str(points), str(edges),  str(components))

    @property
    def n(self):
        ''' Number of nodes in EmbeddedGraph.'''
        return len(self.nodes.points)

    @property
    def m(self):
        ''' Number of edges in EmbeddedGraph.'''
        return len(self.edges)

    @property
    def k(self):
        ''' Number of connected components of EmbeddedGraph.'''
        return len(self.components)

    # TODO: compute the components
    # TODO: compute centers of components when defining graph on components
    @property
    def components(self):
        ''' Computes connected components of EmbeddedGraph'''
        graph_G = graph(self)
        cmpts_G = graph_G.components

        cmpts_emb_G = {}
        point_of = {}
        for i in range(self.n):
            point_of[i] = self.nodes.points[i]

        for i, cmpt_G in cmpts_G.items():
            cmpts_emb_G[i] = PointList( [point_of[j] for j in cmpt_G] )

        return cmpts_emb_G

class Graph:
    def __init__(self, nodes, edges):
        ''' Graph represented with nodes and edges.'''
        self.nodes = nodes
        self.edges = edges

    @property
    def n(self):
        ''' Number of nodes in Graph.'''
        return len(self.nodes)

    @property
    def m(self):
        ''' Number of edges in Graph.'''
        return len(self.edges)

    @property
    def k(self):
        ''' Number of connected components of Graph.'''
        return len(self.components)

    @property
    def components(self):
        cmpts = {}
        k = 0
        unvisited = copy.copy(self.nodes)
        for v in self.nodes:
            if v in unvisited:
                comp_of_v = component(v, self.nodes, self.edges)
                # remove visited nodes in component from unvisited
                unvisited = list(set(unvisited) - set(comp_of_v))
                cmpts[k] = comp_of_v
                k += 1

        return cmpts

    def __str__(self):
        return "nodes: {}\nedges: {}".format(str(self.nodes), str(self.edges))

    def draw(self):
        graph_G = nx.Graph()
        graph_G.add_nodes_from(self.nodes)
        graph_G.add_edges_from(self.edges)

        pos = nx.spring_layout(graph_G)
        nx.draw(graph_G, pos, font_size=10,
                node_color='red', with_labels=True)
        plt.show()

def nhbs(v, graph_G):
    N = []
    for edge in graph_G.edges:
        u1, u2 = edge
        if u1 == v:
            N.append(u2)
        elif u2 == v:
            N.append(u1)
    return N

def component(v, nodes, edges):
    ''' Wrapper of comp.'''
    G = Graph(nodes, edges)
    return comp(v, G, [v]) # T=[v] at the start

def comp(v, graph_G, T):
    N = list(set(nhbs(v, graph_G)) - set(T))
    if N == []:
        return [v]
    else:
        T += N # expand the tree
        for n in N:
            T += comp(n, graph_G, T) # expand the tree (BFS)
    return list(set(T))
# tests
# graph_G = nx.petersen_graph()
# graph_G = Graph(list(graph_G.nodes()), list(graph_G.edges()))
# graph_G.components[0] == graph_G.nodes # True
# graph_G = Graph(list(range(10)), [])
# len(graph_G.components) == 10 # True

def graph(emb_G):
    ''' Translate from EmbeddedGraph to Graph.'''

    point_of = {}
    for i in range(emb_G.n):
        point_of[i] = emb_G.nodes.points[i]

    number_of = {}
    for i in range(emb_G.n):
        number_of[emb_G.nodes.points[i]] = i

    nodes = point_of.keys()
    edges = []
    for i in range(emb_G.n):
        for j in range(i + 1, emb_G.n):
            # test if there is an edge between Points v1 and v2
            v1 = emb_G.nodes.points[i]
            v2 = emb_G.nodes.points[j]

            for edge in emb_G.edges:
                u1 = edge.points[0]
                u2 = edge.points[1]
                if v1.equal(u1) and v2.equal(u2) or \
                   v1.equal(u2) and v2.equal(u1):
                    edges.append( (number_of[v1], number_of[v2]) )

    return Graph(nodes, edges)

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

    def equal(self, p):
        return (self.x == p.x) and (self.y == p.y)

class PointList:
    def __init__(self, points):
        ''' PointList Class to hold a list of Point objects.'''
        if points == [] or isinstance(points[0], Point):
            self.points = points
        else:
            raise ValueError("Args must be a list of Points.")

    @property
    def vertex_points(self):
        vertex_points = []
        for point in self.points:
            if point.label == 'V':
                vertex_points.append(point)

        return vertex_points

    @property
    def edge_points(self):
        edge_points = []
        for point in self.points:
            if point.label == 'E':
                edge_points.append(point)

        return edge_points

    @property
    def center(self):
        ''' Center of mass of the point cloud.'''
        x = np.mean(np.array( [point.x for point in self.points] ))
        y = np.mean(np.array( [point.y for point in self.points] ))

        return Point(x, y)

    def __str__(self):
        return '[' + ','.join(['{!s}'.format(p) for p in self.points]) + ']'

    def __len__(self):
        return len(self.points)

    def contains(self, p):
        for pt in self.points:
            if pt.x == p.x and pt.y == p.y:
                return True
        return False

    def append(self, p):
        self.points.append(p)

    def difference(self, pl):
        difference = PointList([])
        for pt in self.points:
            if not pl.contains(pt):
                difference.append(pt)
        return difference

    def distance(self, point_list):
        ''' Computes minimum distance from self to another point list.'''
        distances = []
        for p1 in self.points:
            for p2 in point_list.points:
                distances.append(distance(p1, p2))

        return np.min(np.array(distances))

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


def draw_point(canvas, pt, radius=0.25, color='blue', **kwargs):
    ''' Draws a point.'''
    point = patches.Circle((pt.x, pt.y),
                        radius=radius,
                        fill=True,
                        facecolor=color,
                        **kwargs)
    canvas.ax.add_patch(point)

def draw_points(canvas, points):
    for point in points:
        if point.label == 'V':
            color = 'red'
        elif point.label == 'E':
            color = 'blue'
        else:
            color = 'green'
        draw_point(canvas, point, color=color)

def distance(p1, p2):
    ''' Euclidean distance between p1, p2.'''
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def get_shell_points(points, center, r, delta):
    ''' Returns a list of points between r and r + delta around the center
    point.'''
    shell_points = []
    for point in points:
        d = distance(center, point)
        if d >= r and d <= r + delta:
            shell_points.append(point)

    return shell_points

def get_ball_points(points, center, r):
    ball_points = []
    for point in points:
        d = distance(center, point)
        if d < r:
            ball_points.append(point)

    return ball_points

def rips_vietoris_graph(delta, points):
    ''' Constructs the Rips-Vietoris graph of parameter delta whose nodes
    are points of the shell.'''
    n = len(points)
    nodes = []
    edges = []
    for i in range(n):
        p1 = points[i]
        nodes.append(p1)
        for j in range(i, n):
            p2 = points[j]
            if not p1.equal(p2) and distance(p1, p2) < delta:
                edges.append([p1, p2])

    return EmbeddedGraph(nodes, edges)

def reconstruct(point_list, delta=3, r=2, p11=1.5, show=False):
    ''' Implementation of Aanjaneya's metric graph reconstruction algorithm.'''
    ## label the points as edge or vertex points
    for center in point_list.points:
        shell_points = get_shell_points(point_list.points, center, r, delta)
        rips_embedded = rips_vietoris_graph(delta, shell_points)

        if rips_embedded.k == 2:
            center.label = 'E'
        else:
            center.label = 'V'
    if show:
        canvas = Canvas('After labeling')
        draw_points(canvas, point_list.points)

    # re-label all the points withing distance p11 from vertex points as vertices
    for center in point_list.vertex_points:
        ball_points = get_ball_points(point_list.edge_points, center, p11)
        for ball_point in ball_points:
            ball_point.label = 'V'
    if show:
        canvas = Canvas('After re-labeling')
        draw_points(canvas, point_list.points)

    # reconstruct the graph structure
    # compute the connected components of Rips-Vietoris graphs:
    # R_delta(vertex_points), R_delta(edge_points)
    rips_V = rips_vietoris_graph(delta, point_list.vertex_points)
    rips_E = rips_vietoris_graph(delta, point_list.edge_points)
    cmpts_V = rips_V.components
    cmpts_E = rips_E.components

    nodes_emb_G = []
    for i, cmpt_V in cmpts_V.items():
        nodes_emb_G.append(cmpt_V.center)

    n = len(nodes_emb_G)
    edges_emb_G = []
    for i in range(n):
        for j in range(i + 1, n):
            for cmpt_E in cmpts_E.values():
                if cmpts_V[i].distance(cmpt_E) < delta and \
                   cmpts_V[j].distance(cmpt_E) < delta:
                    edges_emb_G.append([nodes_emb_G[i], nodes_emb_G[j]])

    emb_G = EmbeddedGraph(nodes_emb_G, edges_emb_G)
    if show:
        canvas = Canvas('Result')
        draw_points(canvas, point_list.points)
        draw_graph(canvas, emb_G)

    return emb_G

def draw_labeling(point_list, delta=3, r=2, p11=1.5):
    ''' Draw the labeling step of the algorithm.'''

    canvas = Canvas('Labeling points as edge or vertex points')
    draw_points(canvas, point_list.points)

    i = int(np.floor(len(point_list.points)/4)) - 2
    center = point_list.points[i]

    draw_ball(canvas, center, r, 'black')
    draw_ball(canvas, center, r + delta, color='black')

    shell_points = get_shell_points(point_list.points, center, r, delta)
    rips_embedded = rips_vietoris_graph(delta, shell_points)

    draw_graph(canvas, rips_embedded, color='red')

    plt.show()

def draw_re_labeling(point_list, delta=3, r=2, p11=1.5):
    # label points as edge or vertex
    for center in point_list.points:
        shell_points = get_shell_points(point_list.points, center, r, delta)
        rips_embedded = rips_vietoris_graph(delta, shell_points)

        if rips_embedded.k == 2:
            center.label = 'E'
        else:
            center.label = 'V'

    canvas = Canvas('Re-labeling points as vertex points')
    draw_points(canvas, point_list.points)

    i = int(np.floor(len(point_list.points)/4)) - 2
    center = point_list.points[i]

    draw_ball(canvas, center, radius=p11, color='black')

    ball_points = get_ball_points(point_list.edge_points, center, p11)
    for ball_point in ball_points:
        draw_point(canvas, ball_point, color='green')

    plt.show()

def draw_graph(canvas, emb_G, color='black'):
    for pt in emb_G.nodes.points:
        draw_point(canvas, pt, color=color)

    for edge in emb_G.edges:
        draw_edge(canvas, edge.points[0], edge.points[1], color=color)

    plt.show()

def draw_ball(canvas, pt, radius=5, color='blue', **kwargs):
    """ Draws a ball."""
    # draw_point(canvas, pt, radius=0.2, color=color)
    circle = patches.Circle((pt.x, pt.y),
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
                              linewidth='3.3',
                              **kwargs)
    canvas.ax.add_patch(line)


if __name__ == "__main__":
    ''' Testing the reconstruction algorithm.'''
    # draw number 7
    diagonal = [Point(i, i) for i in range(1, M, 1)]
    top = [Point(i, M - 1) for i in range(1, M, 1)]
    middle = [Point(i, 14) for i in range(7, 20, 1)]
    points_7 = diagonal + top + middle

    # add noise
    points_7_noise = []
    sigma = 0.1
    for point in points_7:
        x = point.x + list(np.random.normal(0, sigma, 1))[0]
        y = point.y + list(np.random.normal(0, sigma, 1))[0]
        points_7_noise.append(Point(x, y))

    # inputs to the algorithm
    point_list = PointList(points_7_noise)
    delta, r, p11 = 2, 1, 1.5

    # draw the steps of the algorithm
    draw_labeling(point_list, delta, r, p11)
    draw_re_labeling(point_list, delta, r, p11)

    reconstructed = reconstruct(point_list, delta, r, p11)
    canvas = Canvas('Reconstructed graph')
    draw_points(canvas, point_list.points)
    draw_graph(canvas, reconstructed)
