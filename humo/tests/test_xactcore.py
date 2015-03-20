from __future__ import division, print_function

from nose.tools import *
import numpy as np

import humo
import xactcore as xore


def Sdelta_test():
    Sd = xore.Sdelta(+1, 4, 8)
    assert(Sd == Sd)


def chargestate_solve_test():
    H1E = np.zeros((4, 4))
    H1T = -1*np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]])
    H1U = np.diag([2]*4)
    q = humo.HubbardModel(H1E=H1E, H1T=H1T, H1U=H1U)

    p = xore.ChargeState(q, 4)
    p.solve()

    assert(np.abs(p.eigenstates[0]['E'] - (-4.8759428090)) < 1e-6
           and
           np.abs(np.abs(p.eigenstates[0]['V'][0]) - 0.019178364) < 1e-6)
