from __future__ import division, print_function

import numpy as np
import warnings
import xactcore as xore
import xactlang as lang


class Transport:

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

        didv = didv(self.chs[c], [im, ip], Vgs, Vs, Ts,
                         iterations, functional)

        return didv

    def zerobias(self, c, im, ip, Ts, iterations=False, functional=False):

        return self.diamond(c, im, ip, Ts, Vs=np.array([0, 1e-4]),
                            iterations=iterations, functional=functional)




def fermi_function(Es, mu, T):
    """
    Fermi-Dirac Statistics
    """
    return 1/(np.exp((Es-mu)/T)+1)

    
    def didv(chs, imp, Vgs, Vs, Ts, eta, iterations, functional=False):
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

        mid = (np.max(Vgs) + np.min(Vgs))/2

        range = max([
            np.max(Vgs) - mid,
            mid - np.min(Vgs),
            np.max(Vs),
            np.min(Vs)]) + max(Ts)

        omegas = np.linspace(
            mid-range,
            mid+range,
            max([len(Vs), len(Vgs), 1000]))

        if chs.eigenstates is False:
            chs.solve(functional=functional)  # solve in standard way

        ampls = [None]*len(chs.eigenstates)
        for i, aplet in enumerate(chs.eigenstates):
            for j, bplet in enumerate(chs.eigenstates):
                if i == j:
                    if aplet['E'] - chs.eigenstates[0]['E'] < np.max(Vs) + max(Ts):
                        ampls[i][i] = elastic_amplitudes(
                            chs,
                            aplet,
                            imp[0], imp[1],
                            omegas, eta,
                            iterations,
                            functional)
                else:
                    if aplet['E'] - chs.eigenstates[0]['E'] < np.max(Vs) + max(Ts) \
                       and \
                       bplet['E'] - chs.eigenstates[0]['E'] < np.max(Vs) + max(Ts):

                        ampls[i][j] = inelastic_amplitudes(
                            chs,
                            aplet,
                            bplet,
                            imp[0], imp[1],
                            omegas, eta,
                            iterations,
                            functional)

        cond = np.zeros((len(Vgs), len(Vs)), dtype=np.float64)
        for ig, Vg in enumerate(Vgs):
            for ib, V in enumerate(Vs):
                cond[ig, ib] = conductance(Vg, V, ampls, omegas)

        return cond






    
    def fermi_selector(Es, dE, V, Ts):

        VL, VR = V/2, - V/2
        TL, TR = Ts

        # fermi functions
        FL = fermi_function(Es, VL, TL)
        FR = fermi_function(Es, VR, TR)
        FdL = fermi_function(Es + dE, VL, TL)
        FdR = fermi_function(Es + dE, VR, TR)

        return (1-FdR)*FL - (1-FdL)*FR


# humo.HubbardModel.transport = transport
