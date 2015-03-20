from __future__ import division, print_function
import sys
import warnings

import numpy as np
import matplotlib.pylab as plt

import humo

import exdcore as core
import exdkondo as kondo
import exddyson as dyson
import exdgf as gf

from util import *

# General interface

class (humo.HubbardModel):

    def __init__(self):

        super.__init__(self)



    def FDOS():
        dyson.orbitals()


        plt.plot()


