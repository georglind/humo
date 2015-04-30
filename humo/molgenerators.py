from __future__ import division, print_function
# import matplotlib.pyplot as plt
import numpy as np
# from scipy.spatial.distance import pdist, squareform  # distance between realspace points
# from humo import HubbardModel
import molecule


class Alkene(molecule.PiSystem):

    def __init__(self, n, dimer=1, U=10, model='hubbard'):
        self.n = n
        self.d = dimer
        self.U = U
        self.model = model

        self.atoms = ['C']*n
        self.links = [(i, i+1, 2 if i % 2 == 0 else 1) for i in range(n-1)]

    def setup(self, model, atoms=False, links=False):
        n, d, U = self.n, self.d, self.U

        H1E = np.zeros((n, n))
        H1T = np.zeros((n, n))

        for i in xrange(n-1):
            H1T[i, i+1] = -d*2.5 if i % 2 == 0 else -2.5

        H1T = H1T + H1T.T

        xy = np.zeros((n, 3))
        xy[0:n:2, 0] = 1.4*np.cos(np.pi/6)*np.arange(0, n, 2)
        xy[1:n:2, 0] = 1.4*np.cos(np.pi/6)*np.arange(1, n, 2)
        xy[1:n:2, 1] = 1.4*np.sin(np.pi/6)*np.ones((np.floor(n/2),))

        self.xyz = xy

        W = [1]*n
        zs = [1]*n

        H1U = molecule.PiSystem.HU(model, xy, U, W)

        return (H1E, H1T, H1U, U, W, zs)
