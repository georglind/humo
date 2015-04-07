from __future__ import division, print_function

import numpy as np
import warnings
import xactcore as xore
import xactlang as lang
import matplotlib.pyplot as plt

#      $$$$$$\   $$$$$$\ $$$$$$$$\ $$\   $$\ $$\   $$\
#     $$  __$$\ $$  __$$\\__$$  __|$$ |  $$ |$$$\  $$ |
#     $$ /  \__|$$ /  $$ |  $$ |   $$ |  $$ |$$$$\ $$ |
#     $$ |      $$ |  $$ |  $$ |   $$ |  $$ |$$ $$\$$ |
#     $$ |      $$ |  $$ |  $$ |   $$ |  $$ |$$ \$$$$ |
#     $$ |  $$\ $$ |  $$ |  $$ |   $$ |  $$ |$$ |\$$$ |
#     \$$$$$$  | $$$$$$  |  $$ |   \$$$$$$  |$$ | \$$ |
#      \______/  \______/   \__|    \______/ \__|  \__|


def fermi_function(Es, mu, T):
    """
    Fermi-Dirac Statistics
    """
    return 1/(np.exp((Es-mu)/T)+1)


def fermi_selector(Es, dE, V, Ts):

    VL, VR = V/2, - V/2
    TL, TR = Ts

    # fermi functions
    FL = fermi_function(Es, VL, TL)
    FR = fermi_function(Es, VR, TR)
    FdL = fermi_function(Es + dE, VL, TL)
    FdR = fermi_function(Es + dE, VR, TR)

    return (1-FdR)*FL - (1-FdL)*FR


class Transport:
    """
    Transport trough a given charge state of some Hubbard model
    """
    def __init__(self, humo, cs):
        self.humo = humo
        self.cs = {}

        for c in cs:
            self.cs[c] = xore.ChargeState(humo, c)
            self.cs[c].solve(k=1)

    def gate(self, c):
        """
        Return the gate range of the "c" charge state
        """
        Em = self.cs[c-1].eigenstates[0]['E']
        E0 = self.cs[c].eigenstates[0]['E']
        Ep = self.cs[c+1].eigenstates[0]['E']

        return (E0 - Ep, Em - E0)

    def bias(self, c):
        """
        Return the bias range of the number c charge state
        """
        Vg = self.gate(c)
        Vb = np.abs(Vg[1]-Vg[0])/2

        return (-Vb, Vb)

    def range(self, c, Ts):
        """
        Calculate the relevant transport energies
        """
        Vg = self.gate(c)
        Vb = self.bias(c)

        mid = sum(Vg)/2

        range = max([
            max(Vg) - mid,
            mid - min(Vg),
            max(Vb),
            min(Vb)]) + 2*max(Ts)

        omegas = np.linspace(
            mid-range,
            mid+range,
            max([len(Vb), len(Vg), 1000]))

        return omegas

    @staticmethod
    def amplitudes(chs, imp, omegas, eta, cutoff, iterations, functional=False):
        """
        Calculate the differential conductance.

        Parameters
        ----------
        chs : xore.ChargeState Object
            Current charge state
        im : int
            Right lead attachment site
        ip : int
            Left lead attachment site
        iterations: int
            Number of Lanczos inversion iterations
        real : bool
            Is the Hamiltonian real or not
        functional : bool
            Should we calculate the Hamiltonian product with a state using
            a functional approach (False uses an efficient sparse matrix)
        """

        if chs.eigenstates is False:
            chs.solve(functional=functional)  # solve in standard way

        states = [c for c in chs.eigenstates if c['E'] - chs.eigenstates[0]['E'] < cutoff]

        ampls = [None]*len(states)
        for i, aplet in enumerate(states):
            for j, bplet in enumerate(states):
                if i == j:
                    ampls[i][i] = lang.elastic_amplitude(
                        chs,
                        aplet,
                        imp[0], imp[1],
                        omegas, eta,
                        iterations,
                        functional)
                else:
                    ampls[i][j] = lang.inelastic_amplitude(
                        chs,
                        aplet,
                        bplet,
                        imp[0], imp[1],
                        omegas, eta,
                        iterations,
                        functional)

        return ampls

    def currents(self, c, im, ip, Ts, Vgs=None, Vs=None,
                 iterations=False, functional=False):

        cutoff = np.max(Vs) + 2*max(Ts)

        # calculate omegas
        omegas = 0

        ampls = Transport.amplitudes(self.chs[c], (im, ip), omegas, eta, cutoff, iterations, functional)

        curs = np.zeros((len(Vs), len(Vgs)), dtype=np.float64)
        for i, V in enumerate(Vs):
            for j, Vg in enumerate(Vgs):
                curs[i, j] = self.current(c, omegas, ampls, im, ip, Ts, Vg, V,
                                          iterations, functional)

        return curs

    def current(self, c, omegas, ampls, im, ip, Ts, Vg, V, iterations=False, functional=False):

        cur = 0
        for i, aplet in enumerate(self.chs[c].eigenstates):
            for j, bplet in enumerate(self.chs[c].eigenstates):
                cur += np.vdot(fermi_selector(omegas - Vg, aplet['E'] - bplet['E'], V, Ts), ampls[i][j])

        return cur

    def diamond(self, c, im, ip, Ts, Vgs=None, Vs=None,
                iterations=False, functional=False):
        """
        Calculate the Coulomb diamond for number c charge state

        Parameters
        ----------
        c : int
            Number of electrons in given charge state
        im : int
            site i where we remove one electron
        ip : int
            site i where  we add one electron
        Ts : list
            temperature of each lead
        Vgs : ndarray
            gate voltage values
        Vs : ndarray
            bias voltage values
        iterations: int
            number of Lanczos iterations
        functional : bool
            Use functional or matrix representations
        """

        if Vgs is None:
            gate = self.gate(c)
            Vgs = np.linspace(gate[0], gate[1], 1000)

        if Vs is None:
            bias = self.bias(c)
            Vs = np.linspace(bias[0], bias[1], 1000)

        curs = self.currents(self.chs[c], [im, ip], Vgs, Vs, Ts,
                             iterations, functional)

        return np.diff(curs, axis=1)

    def zerobias(self, c, im, ip, Ts, iterations=False, functional=False):

        return self.diamond(c, im, ip, Ts, Vs=np.array([0, 1e-4]),
                            iterations=iterations, functional=functional)

# humo.HubbardModel.transport = transport
