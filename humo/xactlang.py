from __future__ import division, print_function

import numpy as np
import warnings
import xactcore as xore
import lanczos

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

# Calculate resolvents by Lanczos inversion algorithm.


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
    coeffs = elastic_coefficients(chs, multiplet, im, ip,
                                  iterations, functional)

    # number of channels
    ampls = {}
    ampls['W'] = [0]*len(coeffs['W'])

    Es = (multiplet['E'], multiplet['E'])

    for i, W in enumerate(coeffs['W']):

        ampls['W'][i] = map(
            lambda w: amplitude(w, Es, eta, W[0], W[3]),
            omegas)

        if W[1] is not None:
            ampls[i] += map(
                lambda w: amplitude(w, Es, eta, W[1], W[2]),
                omegas)

    if len(coeffs['J']) > 0:
        ampls['J'] = [0*len(coeffs['J'])]

        for i, J in enumerate(coeffs['J']):

            ampls['J'][i] = map(
                lambda w: amplitude(w, Es, eta, J[0], J[1]),
                omegas)

    return ampls


def inelastic_amplitudes(chs, aplet, bplet, im, ip, omegas, eta,
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
    coeffs = inelastic_coefficients(chs, aplet, bplet, im, ip,
                                    iterations, functional)

    # number of channels
    ampls = {}
    ampls['W'] = [0]*len(coeffs['W'])

    for i, W in enumerate(coeffs['W']):

        ampls[i] = map(
            lambda w: amplitude(w, (aplet['E'], bplet['E']), eta, W[0], W[3]),
            omegas)

        if W[1] is not None:
            ampls[i] += map(
                lambda w: amplitude(w, (aplet['E'], bplet['E']), eta, W[1], W[2]),
                omegas)

    if len(ampls['J']) > 0:
        ampls['J'] = [0*len(coeffs['J'])]

        for i, J in enumerate(coeffs['J']):
            ampls[i + len(coeffs['W'])] = map(
                lambda w: amplitude(w, (aplet('E'), bplet['E']), eta, J[0], J[1]),
                omegas)

    return ampls


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
    A += amplitude_real(omega + Epes, etas[0], pes) - \
        amplitude_real(-omega + Eipes, -etas[1], ipes)
    B += amplitude_imag(omega + Epes, etas[0], pes) - \
        amplitude_imag(-omega + Eipes, -etas[1], ipes)

    return A**2 + B**2


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
            potential_coefficients(chs, ns, ket, ket, im, ip,
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
            exchange_coefficients(chs, nR, ketL, ketR, im, ip,
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
            exchange_coefficients(chs, ns, ketL, ketR, im, ip,
                                  iterations, functional)

    return coeffs


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
            potential_coefficients(chs, ns, bket, aket, im, ip,
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
            exchange_coefficients(chs, nR, ketL, ketR, im, ip,
                                  iterations, functional)
        # increase index
        dJ = 1
        coeffs.append(0)

    for i in xrange(len(coeffs['J'])-dJ):  # spin change amplitudes
        ns = bplet['ns'][i+1]
        ketR = xore.State(aplet['V'][i], chs.humo.n, nu=ns[0]+1, nd=ns[1]-1)
        ketL = xore.State(bplet['V'][i+1], chs.humo.n, nu=ns[0], nd=ns[1])

        coeffs['J'][i+dJ] = \
            exchange_coefficients(chs, ns, ketL, ketR, im, ip,
                                  iterations, functional)

    return coeffs


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

    W[0] = coefficients(ketL, dL, Hm, dR, ketR,
                        iterations=iterations, real=chs.humo.real,
                        functional=functional, same_kets=True)

    # Remove spin down and add spin down
    dR = (-1, im, -1)
    dL = (1, ip, -1)

    nu, nd = ns[0], ns[1] - 1

    _, Hm = xore.SzState.hamiltonian(chs.humo, nu + nd, nu, functional)

    W[3] = coefficients(ketL, dL, Hm, dR, ketR,
                        iterations=iterations, real=chs.humo.real,
                        functional=functional, same_kets=True)

    if ns[0] != ns[1]:
        # Add spin down and remove spin down
        dR = (1, ip, -1)
        dL = (-1, im, -1)

        nu, nd = ns[0], ns[1] + 1

        _, Hm = xore.SzState.hamiltonian(chs.humo, nu + nd, nu, functional)

        W[1] = coefficients(ketL, dL, Hm, dR, ketR,
                            iterations=iterations, real=chs.humo.real,
                            functional=functional, same_kets=True)

        # Remove spin up and add spin up
        dR = (-1, im, 1)
        dL = (1, ip, 1)

        nu, nd = ns[0] - 1, ns[1]

        _, Hm = xore.SzState.hamiltonian(chs.humo, nu + nd, nu, functional)

        W[2] = coefficients(ketL, dL, Hm, dR, ketR,
                            iterations=iterations, real=chs.humo.real,
                            functional=functional, same_kets=True)

    return W


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

    J[0] = coefficients(ketL, dL, Hm, dR, ketR,
                        iterations=iterations, real=chs.humo.real,
                        functional=functional, same_kets=False)

    # Remove a spin down and add a spin up
    dR = (-1, im, -1)
    dL = (1, ip, 1)

    nu, nd = ns[0], ns[1] - 1

    _, Hm = xore.SzState.hamiltonian(chs.humo, nu + nd, nu, functional)

    J[1] = coefficients(ketL, dL, Hm, dR, ketR,
                        iterations=iterations, real=chs.humo.real,
                        functional=functional, same_kets=False)

    return J


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
