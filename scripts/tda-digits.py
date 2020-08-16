# conda activate old_py
from __future__ import print_function   # if you are using Python 2
import dionysus as d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

def get_adj_mat_from(A):
    '''
    Return graph adjacency matrix from the image matrix
    We treat the pixels as vertices and add edges between
    adjacent pixels (including diagonals).
    '''
    ics, jcs = np.nonzero(A)
    n = len(ics)
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if neighbours( (ics[i], jcs[i]), (ics[j], jcs[j]) ):
                G[i, j] = 1
    return G

def neighbours(p, q):
    list1 = list(range(p[0] - 1, p[0] + 2))
    list2 = list(range(p[1] - 1, p[1] + 2))
    nbd = [(i, j) for i in list1 for j in list2]
    return q in nbd

def get_example(show=False):
    coords = [[0, 2], [1, 1], [1, 3], [2, 3], [3, 2], [3, 3], [4, 3]]
    m = 5
    A = np.zeros((m, m))
    for coord in coords:
        A[coord[0], coord[1]] = 1

    if show:
        plt.imshow(A)
        plt.show()

    return A

def get_number(n=8, show=False):
    digits = load_digits()
    A = digits.images[n]

    if show:
        plt.imshow(A)
        plt.show()

    return A

def binarize(A, threshold=0, show=False):
    B = np.copy(A)
    if threshold == 0:
        threshold = np.mean(A)/2
    B[B <= threshold] = 0
    B[B > threshold] = 1

    if show:
        plt.imshow(B)
        plt.show()

    return B

if __name__ == "__main__":
    A = get_number(n=3, show=True)
    B = binarize(A, show=True)

    m = B.shape[0]
    f = d.Filtration()

    sweeps = ['right', 'left', 'top', 'bottom']
    sweep = 'right'
    #TODO: for sweep in sweeps: extract features:
    # 4 sweeps and Betti 0, Betti 1 barcodes = 8 features vectors per 1 image

    if sweep in ['right', 'bottom']:
        sweep_range = range(1, m + 1)
    else:
        sweep_range = range(m - 1, -1, -1)

    # TODO: How should the time go:
    #   * or from 0 to m  (now)
    #   * as the sweep_line

    time = 1
    for sweep_line in sweep_range:
        C = np.zeros((m, m))

        if sweep == 'right':
            C[:, 0:sweep_line] = B[:, 0:sweep_line]
        elif sweep == 'left':
            C[:, sweep_line:m] = B[:, sweep_line:m]
        elif sweep == 'top':
            C[sweep_line:m, :] = B[sweep_line:m, :]
        else:
            C[0:sweep_line, :] = B[0:sweep_line, :]

        if np.nonzero(C): # add to the simplex
            G = get_adj_mat_from(C)
            simplices = []

            # add vertices
            for i in range(G.shape[0]):
                simplices.append( ([i], time) )

            # add edges
            for i in range(G.shape[0]):
                for j in range(i+1, G.shape[1]):
                        if G[i, j] > 0:
                            simplices.append( ([i, j], time) )

            for vertices, time in simplices:
                f.append(d.Simplex(vertices, time))

        time += 1 # increase time

    f.sort()
    m = d.homology_persistence(f)
    dgms = d.init_diagrams(m, f)
    for i, dgm in enumerate(dgms):
        print('Betti ', i)
        for pt in dgm:
            print(pt.birth, pt.death)
        print()

    d.plot.plot_bars(dgms[0], show = True)
    d.plot.plot_bars(dgms[1], show = True)
