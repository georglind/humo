from __future__ import division, print_function
# import matplotlib.pyplot as plt
import numpy as np
# from scipy import sparse
# import psutil
# import sys
import warnings
import xactcore as xore
# import xactlintable as lin
import xactlanczos as lanczos
import util

#     $$\
#     $$ |
#     $$ |      $$$$$$\  $$$$$$$\   $$$$$$\
#     $$ |      \____$$\ $$  __$$\ $$  __$$\
#     $$ |      $$$$$$$ |$$ |  $$ |$$ /  $$ |
#     $$ |     $$  __$$ |$$ |  $$ |$$ |  $$ |
#     $$$$$$$$\\$$$$$$$ |$$ |  $$ |\$$$$$$$ |
#     \________|\_______|\__|  \__| \____$$ |
#                                  $$\   $$ |
#                                  \$$$$$$  |
#                                   \______/


def fermi_function(Es, mu, T):
    """
    Fermi-Dirac Statistics
    """
    return 1/(np.exp((Es-mu)/T)+1)


class Lang:
    """
    Greens functions calculated by Lanczos inversion algorithm.
    """

    @staticmethod
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
                        ampls[i][i] = Lang.elastic_amplitudes(
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

                        ampls[i][j] = Lang.inelastic_amplitudes(
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
                cond[ig, ib] = Lang.conductance(Vg, V, ampls, omegas)

        return cond

    @staticmethod
    def elastic_conductanc(Vg, V, Ts, ampltitude):

        if nW % 2 == 0:
            ampls[0:W]


         FL = fermi_function(Es, VL, TL)
         FR = fermi_function(Es, VL, TL)


    # @staticmethod
    # def conductance(Vg, V, Ts, amples, omegas):
    #     # symmetric bias
    #     VL, VR = V/2, -V/2
    #     Tl, TR = Ts
        
    #     FL = fermi_function(Es, VL, TL)
    #     FR = fermi_function(Es, VL, TL)



    # Working here! qqq

    @staticmethod
    def elastic_amplitudes(chs, multiplet, im, ip, omegas, eta,
                           iterations, functional=False):
        """
        The elastic amplitudes

        Parameters
        ----------
        chs : xore.ChargeState Object
            The relevant ChargeState.
        multiplet : ndarray
            The initial and final multiplet as defined by
            xore.ChargeState.eigenstates.
        im : int
            site where one electron is subtracted.
        ip : int
            site where one electron is added.
        omegas : ndarray
            The sampling energies.
        eta : float
            Imaginary part of the resolvent.
        iterations : int
            Number of Lanczos iversion iterations.
        functional : bool
            Whether Hamiltonian is represented as a functional or
            a sparse matrix.
        """
        # coefficients
        coeffs = Lang.elastic_coefficients(chs, multiplet, im, ip,
                                           iterations, functional)

        # number of channels
        nchan = len(coeffs['J']) + len(coeffs['W'])
        ampls = [0]*nchan

        Es = (multiplet['E'], multiplet['E'])

        for i, W in enumerate(coeffs['W']):

            ampls[i] = map(
                lambda w: Lang.amplitude(w, Es, eta, W[0], W[3]),
                omegas)

            if W[1] is not None:
                ampls[i] += map(
                    lambda w: Lang.amplitude(w, Es, eta, W[1], W[2]),
                    omegas)

        for i, J in enumerate(coeffs['J']):

            ampls[i + len(coeffs['W'])] = map(
                lambda w: Lang.amplitude(w, Es, eta, J[0], J[1]),
                omegas)

        return ampls, len(coeffs['W'])

    @staticmethod
    def inelastic_ampltiudes(chs, aplet, bplet, im, ip, omegas, eta,
                             iterations, functional=False):
        """
        The inelastic amplitudes

        Parameters
        ----------
        chs : xore.ChargeState Object
            The relevant ChargeState.
        aplet : ndarray
            The initial multiplet as defined by xore.ChargeState.eigenstates.
        bplet : ndarray
            The final multiplet as defined by xore.ChargeState.eigenstates.
        im : int
            site where one electron is subtracted.
        ip : int
            site where one electron is added.
        omegas : ndarray
            The sampling energies.
        eta : float
            Imaginary part of the resolvent.
        iterations : int
            Number of Lanczos iversion iterations.
        functional : bool
            Whether Hamiltonian is represented as a functional or
            a sparse matrix.
        """
        coeffs = Lang.inelastic_coefficients(chs, aplet, bplet, im, ip,
                                             iterations, functional)

        nchan = len(coeffs['J']) + len(coeffs['W'])
        ampls = [0]*nchan

        for i, W in enumerate(coeffs['W']):

            ampls[i] = map(
                lambda w: Lang.amplitude(w, (aplet['E'], bplet['E']), eta, W[0], W[3]),
                omegas)

            if W[1] is not None:
                ampls[i] += map(
                    lambda w: Lang.amplitude(w, (aplet['E'], bplet['E']), eta, W[1], W[2]),
                    omegas)

        for i, J in enumerate(coeffs['J']):

            ampls[i + len(coeffs['W'])] = map(
                lambda w: Lang.amplitude(w, (aplet('E'), bplet['E']), eta, J[0], J[1]),
                omegas)

        return ampls, len(coeffs['W'])

    @staticmethod
    def amplitude(omega, energies, eta, pes, ipes):
        """
        Return the amplitude as a function of energy.

        Parameters
        ----------
        omega : ndarray
            relevant energies
        energy :
            offset energy of given state
        eta : float
            constant imaginary contribution to the energies
        pes : list
            list of  coefficient list for particle transport
            processes
        ipes : list
            list of  coefficient list for hole transport
            processes
        """
        if not hasattr(eta, '__iter__'):
            etas = [eta]*2

        if len(energies) < 2:
            Epes = Eipes = energies
        else:
            Epes, Eipes = energies

        A = B = 0
        # for i, _ in xange(len(pes)):
        A += Lang.amplitude_real(omega + Epes, etas[0], pes) - \
            Lang.amplitude_real(-omega + Eipes, -etas[1], ipes)
        B += Lang.amplitude_imag(omega + Epes, etas[0], pes) - \
            Lang.amplitude_imag(-omega + Eipes, -etas[1], ipes)

        return A + 1j*B

    @staticmethod
    def amplitude_real(omega, eta, gp):
        """
        The real part of the propagator
        """

        A = lanczos.continued_fraction(
            omega + 1j*eta - gp['ase'],
            gp['bse']
        ) * gp['wve']

        if 'aso' in gp:
            A += -lanczos.continued_fraction(
                omega + 1j*eta - gp['aso'],
                gp['bso']
            ) * gp['wvo']
            A /= 4

        return A

    @staticmethod
    def amplitude_imag(omega, eta, gp):
        """
        Iamginary part of the propagator
        """
        B = 0
        if 'asem' in gp:
            B = lanczos.continued_fraction(
                omega + 1j*eta - gp['asem'],
                gp['bsem']
            ) * gp['wvem']
            B += - lanczos.continued_fraction(
                omega + 1j*eta - gp['asom'],
                gp['bsom']
            ) * gp['wvom']
            B /= 4
        return B

    @staticmethod
    def elastic_coefficients(chs, multiplet, im, ip,
                               iterations, functional=False):
        """
        Cotunneling amplitude for transport through this spin-multiplet

        Parameters
        ----------
        chs : xore.ChargeState Object
            The current charge state
        multiplet : list
            A single member of the list of eigenstates produced
            by xore.ChargeState.solve
        im : int
            Right lead attachement site
        ip : int
            Left lead attachment site
        iterations: int
            Number of Lanczos iterations
        real : bool
            Is the Hamiltonian real or not?
        functional : bool
            Should we calculate the Hamiltonian product with a state using
            a functional approach (False uses an efficient sparse matrix)
        """
        reps = len(multiplet['V'])  # positive size of multiplet (i.e. # m > 0)

        coeffs = {}  # amplitudes
        coeffs['W'] = [0]*reps  # potential scattering amplitudes
        for i in xrange(reps):
            ns, state = (multiplet['n'][i], multiplet['V'][i])
            ket = xore.State(state, chs.humo.n, nu=ns[0], nd=ns[1])
            coeffs['W'][i] = \
                Lang.potential_coefficients(chs, ns, ket, ket, im, ip,
                                            iterations, functional)

        coeffs['J'] = [0]*(reps - (chs.ne + 1) % 2)  # spin change amplitudes

        dJ = 0  # index change
        if chs.ne % 2 == 1:          # if we have an odd number electrons
            nL = multiplet['ns'][0]  # nu - nd = +1
            ketL = xore.State(multiplet['V'][0],
                              nu=nL[0], nd=nL[1])  # state m =.5
            nR = reversed(ns)        # nu - nd = -1
            ketR = ketL.reversed()   # state with m = -.5

            coeffs['J'][0] = \
                Lang.exchange_coefficients(chs, nR, ketL, ketR, im, ip,
                                           iterations, functional)
            # increase index
            dJ = 1
            coeffs.append(0)

        for i in xrange(reps-1):  # spin change amplitudes
            ns, state = (multiplet['n'][i], multiplet['V'][i])
            ketR = xore.State(state, chs.humo.n, nu=ns[0]+1, nd=ns[1]-1)

            ns, state = (multiplet['n'][i+1], multiplet['V'][i+1])
            ketL = xore.State(state, chs.humo.n, nu=ns[0], nd=ns[1])

            coeffs['J'][i+dJ] = \
                Lang.exchange_coefficients(chs, ns, ketL, ketR, im, ip,
                                           iterations, functional)

        return coeffs

    @staticmethod
    def inelastic_coefficients(chs, aplet, bplet, im, ip,
                               iterations, functional=False):
        """
        Cotunneling amplitude for transport through this spin-multiplet

        Parameters
        ----------
        chs : xore.ChargeState Object
            The current charge state
        multiplet : list
            A single member of the list of eigenstates produced
            by xore.ChargeState.solve
        im : int
            Right lead attachement site
        ip : int
            Left lead attachment site
        iterations: int
            Number of Lanczos iterations
        real : bool
            Is the Hamiltonian real or not?
        functional : bool
            Should we calculate the Hamiltonian product with a state using
            a functional approach (False uses an efficient sparse matrix)
        """
        areps = len(aplet['V'])  # positive size of multiplet (i.e. # m > 0)
        breps = len(bplet['V'])

        coeffs = {}  # amplitudes
        coeffs['W'] = [None]*min(areps, breps)  # potential scattering amplitudes
        for i in xrange(len(coeffs['W'])):
            ns = aplet['n'][i]
            aket = xore.State(aplet['V'][i], chs.humo.n, nu=ns[0], nd=ns[1])
            bket = xore.State(bplet['V'][i], chs.humo.n, nu=ns[0], nd=ns[1])

            coeffs['W'][i] = \
                Lang.potential_coefficients(chs, ns, bket, aket, im, ip,
                                            iterations, functional)

        coeffs['J'] = [None]*(min(areps, breps + 1) - (chs.ne + 1) % 2)  # spin change amplitudes

        dJ = 0  # index change
        if chs.ne % 2 == 1:          # if we have an odd number electrons
            nL = aplet['ns'][0]  # nu - nd = +1
            # state m =.5
            ketL = xore.State(bplet['V'][0], nu=nL[0], nd=nL[1])
            nR = reversed(ns)    # nu - nd = -1
            # state m = -.5
            ketR = xore.State(aplet['V'][0], nu=nL[0], nd=nL[1]).reversed()

            coeffs['J'][0] = \
                Lang.exchange_coefficients(chs, nR, ketL, ketR, im, ip,
                                           iterations, functional)
            # increase index
            dJ = 1
            coeffs.append(0)

        for i in xrange(len(coeffs['J'])-dJ):  # spin change amplitudes
            ns = bplet['ns'][i+1]
            ketR = xore.State(aplet['V'][i], chs.humo.n, nu=ns[0]+1, nd=ns[1]-1)
            ketL = xore.State(bplet['V'][i+1], chs.humo.n, nu=ns[0], nd=ns[1])

            coeffs['J'][i+dJ] = \
                Lang.exchange_coefficients(chs, ns, ketL, ketR, im, ip,
                                           iterations, functional)

        return coeffs


    @staticmethod
    def potential_coefficients(chs, ns, ketL, ketR, im, ip,
                               iterations, functional):
        """
        Potential scattering processes

        Parameters
        ----------
        chs : xore.ChargeState Object
            The current charge state
        ns : tuple
            ns[0] = number of spin-up electrons, ns[1] = number of spin-down
            electrons
        ketL : xore.State Object
            The initial state
        im : int
            Right lead attachment site
        ip : int
            Left lead attachment site
        iterations: int
            Number of Lanczos iterations
        real : bool
            Is the Hamiltonian real or not?
        functional : bool
            Should we calculate the Hamiltonian product with a state using
            a functional approach (False: uses an efficient sparse matrix)
        """
        W = [None]*4

        # Add spin up and remove spin up
        dR = (1, ip, 1)
        dL = (-1, im, 1)

        nu, nd = ns[0] + 1, ns[1]

        _, Hm = xore.SzState.hamiltonian(chs.humo, nu + nd, nu, functional)

        W[0] = Lang.coefficients(ketL, dL, Hm, dR, ketR,
                                 iterations=iterations, real=chs.humo.real,
                                 functional=functional, same_kets=True)

        # Remove spin down and add spin down
        dR = (-1, im, -1)
        dL = (1, ip, -1)

        nu, nd = ns[0], ns[1] - 1

        _, Hm = xore.SzState.hamiltonian(chs.humo, nu + nd, nu, functional)

        W[3] = Lang.coefficients(ketL, dL, Hm, dR, ketR,
                                 iterations=iterations, real=chs.humo.real,
                                 functional=functional, same_kets=True)

        if ns[0] != ns[1]:
            # Add spin down and remove spin down
            dR = (1, ip, -1)
            dL = (-1, im, -1)

            nu, nd = ns[0], ns[1] + 1

            _, Hm = xore.SzState.hamiltonian(chs.humo, nu + nd, nu, functional)

            W[1] = Lang.coefficients(ketL, dL, Hm, dR, ketR,
                                     iterations=iterations, real=chs.humo.real,
                                     functional=functional, same_kets=True)

            # Remove spin up and add spin up
            dR = (-1, im, 1)
            dL = (1, ip, 1)

            nu, nd = ns[0] - 1, ns[1]

            _, Hm = xore.SzState.hamiltonian(chs.humo, nu + nd, nu, functional)

            W[2] = Lang.coefficients(ketL, dL, Hm, dR, ketR,
                                     iterations=iterations, real=chs.humo.real,
                                     functional=functional, same_kets=True)

        return W

    @staticmethod
    def exchange_coefficients(chs, ns, ketL, ketR, im, ip,
                              iterations, functional):
        """
        Processes which change the molecular spin state

        Parameters
        ----------
        chs : xore.ChargeState Object
            The current charge state
        ns : tuple
            ns[0] = number of spin-up electrons, ns[1] = number of spin-down
            electrons. All with respect to ketR.
        ketL : xore.State Object
            The initial molecular state
        ketR : xore.State Object
            The final state
        im : int
            Right lead attachment site
        ip : int
            Left lead attachment site
        iterations: int
            Number of Lanczos iterations
        real : bool
            Is the Hamiltonian real or not?
        functional : bool
            Should we calculate the Hamiltonian product with a state using
            a functional approach (False uses an efficient sparse matrix)
        """
        J = [None]*2

        # Add a spin up and remove a spin down
        dR = (1, ip, 1)
        dL = (-1, im, -1)

        nu, nd = ns[0] + 1, ns[1]

        _, Hm = xore.SzState.hamiltonian(chs.humo, nu + nd, nu, functional)

        J[0] = Lang.coefficients(ketL, dL, Hm, dR, ketR,
                                 iterations=iterations, real=chs.humo.real,
                                 functional=functional, same_kets=False)

        # Remove a spin down and add a spin up
        dR = (-1, im, -1)
        dL = (1, ip, 1)

        nu, nd = ns[0], ns[1] - 1

        _, Hm = xore.SzState.hamiltonian(chs.humo, nu + nd, nu, functional)

        J[1] = Lang.coefficients(ketL, dL, Hm, dR, ketR,
                                 iterations=iterations, real=chs.humo.real,
                                 functional=functional, same_kets=False)

        return J

    @staticmethod
    def coefficients(ketL, dL, Hm, dR, ketR,
                     iterations=10, real=True, functional=False,
                     same_kets=False):

        if iterations < 1:  # no iterations
            return False

        nonortho = 0

        dL = (- dL[0], dL[1], dL[2])  # reverse add or remove

        if dL[0] == dR[0] and dL[1] == dR[1] and same_kets:

            v = xore.createannihilate(*dL, state=ketL)
            wve = np.vdot(v.v, v.v)

            ase, bse, vend = lanczos.lanczos(v.v, Hm, iterations)

            if vend > 1e-4:
                nonortho = 1

            G = {'ase': ase, 'bse': bse, 'wve': wve}

        elif real:

            va = xore.createannihilate(*dL, state=ketL)
            vb = xore.createannihilate(*dR, state=ketR)

            ve, vo = (va.v + vb.v, va.v - vb.v)
            wve, wvo = (np.vdot(ve, ve), np.vdot(vo, vo))

            ase, bse, veend = lanczos.lanczos(ve, Hm, iterations, functional)

            if veend > 1e-4:
                nonortho = 1

            aso, bso, voend = lanczos.lanczos(vo, Hm, iterations, functional)

            if voend > 1e-4:
                nonortho = 1

            G = {}
            G['ase'], G['bse'], G['wve'] = (ase, bse, wve)
            G['aso'], G['bso'], G['wvo'] = (aso, bso, wvo)

        else:

            va = xore.createannihilate(*dL, state=ketL)
            vb = xore.createannihilate(*dR, state=ketR)

            ve, vo = (va.v + vb.v, va.v - vb.v)
            wve, wvo = (np.vdot(ve, ve), np.vdot(vo, vo))

            vem, vom = (va.v + 1j*vb.v, va.v - 1j*vb.v)
            wvem, wvom = (np.vdot(vem, vem), np.dot(vom, vom))

            ase, bse, veend = lanczos.lanczos(ve, Hm,
                                              iterations, functional)

            if veend > 1e-4:
                nonortho = 1

            aso, bso, voend = lanczos.lanczos(vo, Hm,
                                              iterations, functional)

            if voend > 1e-4:
                nonortho = 1

            asem, bsem, vemend = lanczos.lanczos(vem, Hm,
                                                 iterations, functional)
            if vemend > 1e-4:
                nonortho = 1

            asom, bsom, vomend = lanczos.lanczos(vom, Hm,
                                                 iterations, functional)
            if vomend > 1e-4:
                nonortho = 1

            G = {}
            G['ase'], G['bse'], G['wve'] = ase, bse, wve
            G['aso'], G['bso'], G['wvo'] = aso, bso, wvo
            G['asem'], G['bsem'], G['wvem'] = asem, bsem, wvem
            G['asom'], G['bsom'], G['wvom'] = asom, bsom, wvom

        if nonortho == 1:
            warnings.warn('Non-orthogonal Lanczos inversion', UserWarning)

        return G


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
            numebr of Lanczos iterations
        functional : bool
            Use functional or matrix representations
        """

        if Vgs is None:
            gate = self.gate(c)
            Vgs = np.linspace(gate[0], gate[1], 1000)

        if Vs is None:
            bias = self.bias(c)
            Vs = np.linspace(bias[0], bias[1], 1000)

        didv = Lang.didv(self.chs[c], [im, ip], Vgs, Vs, Ts,
                         iterations, functional)

        return didv

    def zerobias(self, c, im, ip, Ts, iterations=False, functional=False):

        return self.diamond(c, im, ip, Ts, Vs=np.array([0, 1e-4]),
                            iterations=iterations, functional=functional)




# humo.HubbardModel.transport = transport
