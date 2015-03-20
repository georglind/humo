from __future__ import division, print_function
# import matplotlib.pyplot as plt
import numpy as np


class HubbardModel:

    def __init__(self, H1E=None, H1T=None, H1U=None):
        self.n = H1T.shape[0]
        self.H1T = H1T
        self.H1E = H1E
        self.H1U = H1U

    def HU_operator(self, symmetrized=True):
        return HubbardModel.HU_factory(self.H1U, symmetrized=True)

    @staticmethod
    def HU_factory(H1U, symmetrized=True):
        HUii = np.diag(H1U)
        HUij = H1U - np.diag(HUii)

        if symmetrized:
            fcn = lambda statesu, statesd: \
                np.sum(.5 * ((statesu + statesd - 1).dot(HUij))*(statesu + statesd - 1), axis=1) \
                + ((statesu - .5)*(statesd - .5)).dot(HUii)
        else:
            fcn = lambda statesu, statesd: \
                np.sum(.5 * ((statesu + statesd).dot(HUij))*(statesu + statesd), axis=1) \
                + ((statesu)*(statesd)).dot(HUii)

        return fcn

    @property
    def real(self):
        return np.sum(np.abs(np.imag(self.H1T))) < 1e-10


# This class provides the functionality we want. You only need to look at
# this if you want to know how this works. It only needs to be defined
# once, no need to muck around with its internals.
class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False
