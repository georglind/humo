from __future__ import division, print_function

import numpy as np
import warnings
import pyprind
import xactcore as xore
import xactlang as lang
# import matplotlib.pyplot as plt

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
    Fermi-Dirac Statistics distribution
    """
    return 1/(np.exp((Es-mu)/T)+1)


def fermi_selector(Es, dE, V, Ts):
    """
    Return a fermi-selector function: f_L(E) (1 - f_R(E))
    """
    VL, VR = V/2, - V/2
    TL, TR = Ts

    # fermi functions
    # FL = fermi_function(Es, VL, TL)
    # FR = fermi_function(Es, VR, TR)
    # FdL = fermi_function(Es + dE, VL, TL)
    # FdR = fermi_function(Es + dE, VR, TR)

    # return (1-FdR)*FL - (1-FdL)*FR
    return (1 - 1/(np.exp((Es+dE-VR)/TR)+1))*1/(np.exp((Es-VL)/TL)+1) - \
           (1 - 1/(np.exp((Es+dE-VL)/TL)+1))*1/(np.exp((Es-VR)/TR)+1)


class Transport:
    """
    Transport trough a given charge state of some Hubbard model
    """
    def __init__(self, humo, cs):
        self.humo = humo
        self.cs = {}

        for c in cs:
            self.cs[c] = xore.ChargeState(humo, c)
            self.cs[c].solve(ns=10)

    def gate(self, c):
        """
        Return the gate range of the "c" charge state
        """
        Em = self.cs[c-1].eigenstates[0]['E']
        E0 = self.cs[c].eigenstates[0]['E']
        Ep = self.cs[c+1].eigenstates[0]['E']

        return (E0 - Em, Ep - E0)

    def bias(self, c):
        """
        Return the bias range of the number c charge state
        """
        Vg = self.gate(c)
        Vb = np.abs(Vg[1]-Vg[0])

        return (-Vb, Vb)

    def range(self, c, Ts):
        """
        Calculate the relevant transport energies
        """
        if type(Ts) is not list:
            Ts = [Ts, Ts]

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

        #progress bar
        progress = pyprind.ProgBar(len(states)**2,
                                   monitor=True,
                                   title="Amplitudes")

        ampls = [None]*len(states)
        for i, aplet in enumerate(states):
            ampls[i] = [None]*len(states)
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

                progress.update()  # update progress bar

        return ampls

    def amplitudess(self, c, imp, eta, cutoff, iterations, functional=False):
        """
        Calculate the cotunneling amplitudes for processes within the same charge state.
        """
        im, ip = imp

        omegas = 0

        # different site amplitudes:
        amplsmp = Transport.amplitudes(self.cs[c], (im, ip), omegas, eta, cutoff,
                                       iterations, functional)

        # same site amplitudes:
        amplsmm = Transport.amplitudes(self.cs[c], (im, im), omegas, eta, cutoff,
                                       iterations, functional)
        amplspp = Transport.amplitudes(self.cs[c], (ip, ip), omegas, eta, cutoff,
                                       iterations, functional)

        N = len(self.cs[c].eigenstates)

        tampls = [None]*N
        for i in xrange(N):
            tampls[i] = [None]*N
            for j in xrange(N):
                tampls[i][j] = amplsmp[i][j] + amplsmm[i][j] + amplspp[i][j]

        return amplsmp, tampls

    def occupations(self, c, omegas, Vg, V, amplitudes):
        """
        Calculate the occupations from rate equations an cotunneling amplitudes
        """
        N = len(self.cs[c].eigenstates)  # number of state

        rates = np.zeros((N, N))  # rates matrix
        # fill rates matrix
        for i, aplet in enumerate(self.cs[c].eigenstates):
            for j, bplet in enumerate(self.cs[c].eigenstates):
                rates[i][j] = np.vdot(fermi_selector(omegas + Vg, aplet['E'] - bplet['E'], V, Ts), ampls[i][j])

        rateq = rates - np.diag(np.sum(rates, axis=0))
        p = np.linalg.null(rateq)
        return p

    def currents(self, c, im, ip, Ts, Vgs=None, Vs=None,
                 cache=True, iterations=False, functional=False):
        """
        Calculate the current for c'th charge state over a range of
        bias voltages and backgate voltages.

        Parameters
        ----------
        c : int
            Number of electron in the given charge state.
        im : int
            site i where we remove one electron.
        ip : int
            site i where we add one electron.
        Ts : list
            Temperatures of left and right electrode.
        Vgs : ndarray
            List of backgate voltages.
        Vs : ndarray
            List of backgate voltages.
        iterations : int
            Number of Lanczos iterations to perform.
        functional : boolean
            Whether to perform functional or matrix inversion.
        """
        cutoff = 1e4
        # np.max(Vs) + 10*max(Ts)

        # calculate omegas
        omegab = np.min([np.min(Vgs), np.min(Vs)]) - np.max(Ts)
        omegat = np.max([np.max(Vgs), np.max(Vs)]) + np.max(Ts)

        N = 2*np.round((omegat-omegab)/np.min(Ts))

        omegas = np.linspace(omegab, omegat, N)
        eta = 1e-3  # arbitrary setting

        # Recache if the cache is empty
        if not hasattr(self, 'ampls'):
            cache = False

        # Recache if the former sampling was too small for the current
        # calculation
        if hasattr(self, 'omegas'):
            if (self.omegas[0] > omegas[0]
               or self.omegas[-1] < omegas[-1]
               or len(self.omegas) < len(omegas)):
                cache = False

        if cache is False:
            self.omegas = omegas
            self.ampls = Transport.amplitudes(self.cs[c], (im, ip), self.omegas, eta, cutoff,
                                              iterations, functional)

        curs = np.zeros((len(Vs), len(Vgs)), dtype=np.float64)

        # progress bar
        progress = pyprind.ProgBar(len(Vs)*len(Vgs),
                                   monitor=True,
                                   title="Current")

        for i, V in enumerate(Vs):
            for j, Vg in enumerate(Vgs):
                curs[i, j] = self.current(c, self.omegas, self.ampls, im, ip, Ts, Vg, V,
                                          iterations, functional)
                progress.update()  # update progress bar

        return curs

    def current(self, c, omegas, ampls, im, ip, Ts, Vg, V,
                iterations=False, functional=False):
        """
        Calculate the current for c'th charge state over a range of omegas
        based on some pre-calculated amplitudes.

        Parameters
        ----------
        c : int
            Number of electron in the given charge state.
        omegas : ndarray
            Array of numbers where we perform the calculation.
        ampls : Object
            The continued fraction amplitudes.
        im : int
            site i where we remove one electron.
        ip : int
            site i where we add one electron.
        Ts : list
            Temperatures of left and right electrode.
        Vg : float64
            The current backgate voltage.
        V : float64
            The current bias voltage.
        iterations : int
            Number of Lanczos iterations to perform.
        functional : boolean
            Whether to perform functional or matrix inversion.
        """
        cur = 0
        # only consider simplest occupation:
        i = 0
        aplet = self.cs[c].eigenstates[i]
        # for i, aplet in enumerate(self.cs[c].eigenstates):
        for j, bplet in enumerate(self.cs[c].eigenstates):
            cur += np.vdot(fermi_selector(omegas - Vg, aplet['E'] - bplet['E'], V, Ts), ampls[i][j])

        return cur


class Diamond(Transport):

    def diamond(self, c, im, ip, Ts, Vgs=None, Vs=None,
                cache=True, iterations=False, functional=False):
        """
        Calculate the Coulomb diamond for the c'th charge state

        Parameters
        ----------
        c : int
            Number of electrons in the given charge state
        im : int
            site i where we remove one electron
        ip : int
            site i where we add one electron
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

        if type(Ts) is not list:
            Ts = [Ts, Ts]

        curs = self.currents(c, im, ip, Ts, Vgs, Vs,
                             cache, iterations, functional)

        return (Vgs, Vs, curs)

    def zerobias(self, c, im, ip, Ts, Vgs=None, Vs=None, 
                 iterations=False, functional=False):

        # defaults
        if Vs is None:
            Vs = np.array([0, 1e-4])

        # recast as numpy array
        if isinstance(Vs, list):
            Vs = np.array(Vs)

        return self.diamond(c, im, ip, Ts, Vgs=Vgs, Vs=Vs,
                            iterations=iterations, functional=functional)


import scipy.optimize as optimize
import scipy.interpolate as interpolate


class ThermoVoltage(Diamond):

    def signal(self, c, im, ip, Ts, Vgs=None, Vs=None,
               cache=True, iterations=False, functional=False):
        """
        The ThermoVoltage signal
        """
        # Find the signal
        Vgs, Vs, curs = self.diamond(c, im, ip, Ts, Vgs=Vgs, Vs=Vs,
                                     cache=cache,
                                     iterations=iterations,
                                     functional=functional)

        Tv = np.zeros(Vgs.shape)
        for i, Vg in enumerate(Vgs):
            ff = interpolate.interp1d(Vs, curs[:, i], kind='cubic')
            Tv[i] = optimize.brentq(ff, np.min(Vs), np.max(Vs))  #, 1e-4*np.max(Vs))
            # idx = (curs[:, i] >= 0).argmax()
            # Tv[i] = Vs[idx]

        return Vgs, Tv



# humo.HubbardModel.transport = transport
