from __future__ import division, print_function
import psutil
import sys
import warnings

import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as slinalg

import xactlintable as lin
import xactlanczos as lanczos

import util

try:
    verbal
except NameError:
    verbal = True

#  Welcome to:
#
#  $$\   $$\  $$$$$$\   $$$$$$\ $$$$$$$$\
#  $$ |  $$ |$$  __$$\ $$  __$$\\__$$  __|
#  \$$\ $$  |$$ /  $$ |$$ /  \__|  $$ | $$$$$$$\  $$$$$$\   $$$$$$\   $$$$$$\
#   \$$$$  / $$$$$$$$ |$$ |        $$ |$$  _____|$$  __$$\ $$  __$$\ $$  __$$\
#   $$  $$<  $$  __$$ |$$ |        $$ |$$ /      $$ /  $$ |$$ |  \__|$$$$$$$$ |
#  $$  /\$$\ $$ |  $$ |$$ |  $$\   $$ |$$ |      $$ |  $$ |$$ |      $$   ____|
#  $$ /  $$ |$$ |  $$ |\$$$$$$  |  $$ |\$$$$$$$\ \$$$$$$  |$$ |      \$$$$$$$\
#  \__|  \__|\__|  \__| \______/   \__| \_______| \______/ \__|       \_______|


# Conventions
# states are always built in the basis:
#   basis[0]    = basisu[0] basisd[0]
#   basis[1]    = basisu[1] basisd[0]
#   basis[2]    = basisu[2] basisd[0]
#   ...
#   basis[Nu-1] = basisu[Nu-1] basisd[0]
#   basis[Nu]   = basisu[0] basisd[1]
#   basis[Nu+1] = basisu[1] basisd[1]
#   ...
#
# and basis states have create_up operators before create_down operators

def diagonalize(A, k=10, return_eigenvectors=True, ncv=None):
    """
    Computes the low energy eigenstates

    Parameters
    ----------
    A : sparse matrix or LinearOperator
        The linear system
    k : int
        Number of eigenstates to compute
    return_eigenvectors : bool
        Whether or not to return eigenstates
    ncv : int
        Number of Lanczos vectors generated
    """
    if ncv is None:
        ncv = min(A.shape[0], 2*k + 1)

    if (A.shape[0] == 1):
        return (np.array([A[0, 0]]), np.array([[1]]))

    k = min(k, A.shape[0] - 2)

    try:
        w, v = slinalg.eigsh(
            A,
            k=k,
            which='SA',
            return_eigenvectors=return_eigenvectors,
            ncv=ncv,
            maxiter=1e4)
    except slinalg.ArpackNoConvergence as inst:
        warnings.warn(
            "Error: No convergence. Sparse diagonalization did not converge.",
            UserWarning)
        w = inst['eigenvalues']
        v = inst['eigenvectors']

    # sort eigenvalues
    idx = w.argsort()
    w = w[idx]
    v = v[:, idx]

    return (w, v)


class ChargeState:

    def __init__(self, humo, ne):
        self.humo = humo
        self.ne = ne
        # init
        self.eigenstates = []
        self.spinstates = []

    def solve(self, ns=10, cache=True, spin_extend=True):
        """
        Solve the system

        Parameters
        ----------
        ns : int
            Number of spin states
        nss : int
            Number of states to return
        spin_extend : bool
            Whether to extend the spin
        """

        # remember the cache
        if cache and ns < len(self.eigenstates):
            if not spin_extend or self.eigenstates[0]['d']:
                ds = max([s['d'] for s in self.eigenstates])

                util.message(
                    ('Cached {0} - ne={1} charge state - lowest {2} states - ' +
                     'Sz={3} to {4}').format(
                        self.humo.__class__.__name__,
                        self.ne,
                        ns,
                        (np.ceil(self.ne/2) - np.floor(self.ne/2))/2,
                        (np.ceil(self.ne/2) - np.floor(self.ne/2))/2 + ds/2),
                    verbal)  # verbal
                return

        nu = np.ceil(self.ne/2)
        nd = np.floor(self.ne/2)

        sz0 = SzState(self.humo, nu, nd)  # lowest sz spin state
        w, v = sz0.solve(ns=ns)  # get the <ns> lowest states

        ns = len(w)  # update ns

        states = [
            {
                'E': w[i],
                'V': [v[:, i]],
                'd': False, 'n': [(nu, nd)]
            }
            for i in range(ns)]

        ds = 0
        if spin_extend is True:  # construct the spin-degenerate states
            (states, ds) = ChargeState.spin_extend(states, nu, nd, self.humo.n)

        self.eigenstates = states

        util.message(
            ('Solved {0} - ne={1} charge state - lowest {2} states - ' +
             'Sz={3} to {4}').format(
                self.humo.__class__.__name__,
                self.ne,
                ns,
                (nu - nd)/2,
                (nu - nd)/2 + ds),
            verbal)  # verbal

    @staticmethod
    def spin_extend(states, nu, nd, n):
        """
        Extend states by adding spin-degenerate states

        Parameters
        ----------
        states : list
            List of states on the form {E, V, d}
        nu : int
            Number of spin up electrons
        nd : int
            Number of spin down electrons
        """
        ns = len(states)                             # number of states
        v = np.array([s['V'][0] for s in states]).T  # create array of states

        for i in xrange(ns):
            # if degeneracy has not been set
            if not states[i]['d']:
                states[i]['d'] = 1 + (nu-nd) % 2    # set starting degeneracy

        active = np.array([True]*ns, dtype=bool)    # elements to try Splussing
        ds = -1                                     # Delta Sz

        while any(active):
            ds += 1                                 # spin pre increment

            Sp = Splus(nu+ds, nd-ds, n)   # construct splus
            v = Sp.dot(v)                           # resulting array

            weights = np.sqrt(np.sum(np.abs(v)**2, axis=0))

            # update index of surviving multiplet
            active[active] = weights > 0.5
            v = v[:, weights > 0.5]                 # surviving states

            if any(active):
                for i, idx in enumerate(np.flatnonzero(active)):
                    # append states
                    states[idx]['V'].append(v[:, i] / weights[i])
                    states[idx]['d'] += 2
                    states[idx]['n'].append((nu+ds+1, nd-ds-1))

        return (states, ds)

    def spectrum(self, ns=10):
        """
        Caclulate and return the eigenspectrum

        Parameters
        ----------
        ns : int
            Number of states to include
        spindeg : bool
            Whether to calculate spin degeneracy of
        """
        self.solve(ns=ns, cache=True, spin_extend=False)
        return [s['E'] for s in self.eigenstates]

    @staticmethod
    def order_degeneracy(states, tolerance=1e-5):

        Es = [st['E'] for st in states]

        degs = [-1]

        for i, E in enumerate(Es):
            if i > 0 and np.abs(E[i] - E[i-1]) < tolerance:
                degs[-1].append(i)
            else:
                degs.append([i])

        return degs


class SzState:
    """
    SpinState of a given ChargeState
    """

    def __init__(self, humo, nu, nd):
        self.n = humo.n
        self.humo = humo
        self.nu = nu
        self.nd = nd
        self.sz = nu-nd
        self.lin = lin.Table(nu, nd, humo.n)

        # init
        self.w = None
        self.v = None

    def solve(self, ns=10, return_eigenvectors=True):
        """
        Solves (i.e. find the low energy eigenstates) of a certain spin system
        with a given z component of spin.

        Parameters
        ----------
        ns : int
            number of states
        functional : boolean
            Use functional or matrix representation of the Hamiltonian
        return_eigenvectors: boolean
            Return the eigenvectors along with the ..
        """
        # what are trying to deal with?
        systemsize = (util.choose(self.n, self.nu) *
                      util.choose(self.n, self.nd) *
                      80)

        # what resources do we have
        svmem = psutil.virtual_memory()
        mcap = np.floor(svmem.free/systemsize)
        cap = 2*mcap if self.humo.real else mcap

        # construct the best solution to the problem
        functional = False

        if cap < 2:
            print('Error. Not enough free virtual memory for \
                   exact diagonalization of this system')
            sys.exit('Not enough memory')

        if cap < ns:
            # Solve
            util.message('cap<ns', verbal)
            ns = np.max(cap - 2, 1)
            functional = True

        if cap < 40:
            util.message('cap40', verbal)
            functional = True

        self.lita, H = SzState.hamiltonian(
            self.humo,
            self.nu+self.nd,
            self.nu,
            functional=functional)

        if functional:  # functional definition
            Nu, Nd = self.lita.Ns
            HTu, HTd, HU = H
            H = slinalg.LinearOperator(
                (Nu*Nd, Nu*Nd),
                dtype=HTu.dtype,
                matvec=lambda v: lanczos.Hv(HTu, HTd, HU, v))

        self.w, self.v = diagonalize(
            H,
            k=min(ns, np.prod(self.lita.Ns)-1),
            return_eigenvectors=return_eigenvectors)

        return self.w, self.v

    @staticmethod
    def hamiltonian(humo, ne, nu, functional=False):
        """
        The total hamiltonian
        """
        lita, Hm = SzState.hamiltonian_setup(humo, ne, nu)
        if functional is True:
            H = Hm
        else:
            H = SzState.hamiltonian_matrix(lita.Ns, Hm)

        return lita, H

    @staticmethod
    def hamiltonian_setup(humo, ne, nu):
        """
        Creates functional Hamiltonian
        """
        n = humo.n
        nd = ne - nu

        lita = lin.Table(n, nu, nd)  # create the lin table for this szstate

        if nu > n or nd > n:
            H = np.zeros((1, 1))
            return lita, (H, H, H)

        # hopping for spin up
        HTu = SzState.hopping_hamiltonian(lita.basisu, lita.Juv, humo.H1T,
                                          real=humo.real)

        # hopping for spin down
        HTd = SzState.hopping_hamiltonian(lita.basisd, lita.Jdv, humo.H1T,
                                          real=humo.real)

        HU = SzState.coulomb_energies(humo, ne, nu)

        return lita, (HTu, HTd, HU)

    @staticmethod
    def hamiltonian_matrix(Ns, Hm):
        """
        Creates matrix version of the Hamiltonian
        """
        Nu, Nd = Ns
        HTu, HTd, HU = Hm

        # fold up system
        H = SzState.foldup(HTu, Nu, Nd) + SzState.folddown(HTd, Nu, Nd)

        # add coulomb interaction
        return H + \
            sparse.coo_matrix((HU, (np.arange(Nu*Nd), np.arange(Nu*Nd))),
                              shape=(Nu*Nd, Nu*Nd)).tocsr()

    @staticmethod
    def hamiltonian_function(Ns, Hm):
        """
        Creates a LinearOperator Hamiltonian
        """
        Nu, Nd = Ns
        HTu, HTd, HU = Hm
        return slinalg.LinearOperator(
            (Nu*Nd, Nu*Nd),
            dtype=HTu.dtype,
            matvec=lambda v: lanczos.Hv(HTu, HTd, HU, v))

    @staticmethod
    def coulomb_energies(humo, ne, nu, real=True):
        """
        The Coulomb energy of each state
        """
        n = humo.n
        nd = ne-nu

        if nu > n or nd > n:
            H = np.array([0])
            return H

        lita = lin.Table(n, nu, nd)
        Nu, Nd = lita.Ns

        HU = humo.HU_operator()

        # The coulomb energy
        HUv = np.zeros(
            (Nd*Nu,),
            dtype=np.float64 if real else np.complex128)

        for cN in xrange(Nd):

            index = cN*Nu + np.arange(Nu)

            statesu, statesd = lin.index2state(
                index,
                n=n,
                Juv=lita.Juv,
                Jdv=lita.Jdv)   # get all states

            HUv[index] = HU(statesu, statesd) + \
                np.sum(statesd.dot(humo.H1E), axis=1) + \
                np.sum(statesu.dot(humo.H1E), axis=1)

        return HUv

    @staticmethod
    def hopping_hamiltonian(states, Jv, H1T, real=True):
        """
        The kinetic part of the hamiltonian
        """
        if len(Jv) == 1:
            return np.array([1, 1, 0])

        n = H1T.shape[0]                        # number of site

        ne = np.sum(states[0, :])               # number of electrons

        # only the upper triangular part
        H1T = np.triu(H1T)
        # count number of hopping elements
        nH1T = np.sum(H1T != 0)
        ts = np.transpose(np.nonzero(H1T))      # ts

        # J = {J: i for i, J in enumerate(Jv)}   # reverse index
        NJ = len(Jv)
        HS = sparse.dok_matrix(
            (NJ, NJ),
            dtype=np.float64 if real else np.complex128)

        for c in xrange(nH1T):

            H1H = np.eye(n, dtype=int)
            i = ts[c, 0]
            j = ts[c, 1]

            H1H[j, j] = 0
            H1H[i, i] = 0
            H1H[i, j] = 1

            vouts = states.dot(H1H).astype(int)
            vouts[states[:, i]+states[:, j] == 0, :] = 0
            vouts[states[:, i]+states[:, j] == 2, :] = 0

            idx = np.flatnonzero(np.sum(vouts, axis=1) == ne)

            if len(idx) > 0:
                touts = (-1)**np.mod(
                    np.sum(states[idx, min(i, j):max(i, j)], axis=1) - 1,
                    2)*H1T[i, j]
                HS[idx, lin.state2index(vouts[idx, :], Jv)] = touts

        return HS.tocsr() + HS.tocsr().H

    @staticmethod
    def foldup(HT, Nu, Nd):
        if Nu == 1:
            return sparse.coo_matrix((Nu*Nd, Nu*Nd)).tocsr()
        return sparse.kron(np.eye(Nd), HT, 'dok')

    @staticmethod
    def folddown(HT, Nu, Nd):
        if Nd == 1:
            return sparse.coo_matrix((Nu*Nd, Nu*Nd)).tocsr()
        return sparse.kron(HT, np.eye(Nu), 'dok')


#      $$$$$$\    $$\                $$\
#     $$  __$$\   $$ |               $$ |
#     $$ /  \__|$$$$$$\    $$$$$$\ $$$$$$\    $$$$$$\
#     \$$$$$$\  \_$$  _|   \____$$\\_$$  _|  $$  __$$\
#      \____$$\   $$ |     $$$$$$$ | $$ |    $$$$$$$$ |
#     $$\   $$ |  $$ |$$\ $$  __$$ | $$ |$$\ $$   ____|
#     \$$$$$$  |  \$$$$  |\$$$$$$$ | \$$$$  |\$$$$$$$\
#      \______/    \____/  \_______|  \____/  \_______|

class State:
    """
    Manipulate many-body states by adding or substracting fermions.
    """

    def __init__(self, v, n, lita=None, nu=None, nd=None):
        self.v = v

        if lita is not None:
            if nu is None or nd is None:
                nu = lita.nu
                nd = lita.nd
        if lita is None:
            lita = lin.Table(n, nu, nd)

        self.n = n
        self.nu = nu
        self.nd = nd
        self.lita = lita

    def reversed(self):

        nu, nd = self.lita.ns
        Nu, Nd = self.lita.Ns

        staten = np.reshape(np.reshape(self.v, (Nu, Nd)), (Nu*Nd, 1))

        if np.min(nu, nd) % 2 == 1:
            staten = -staten

        return State(staten, nu=nd, nd=nu)

    def created(self, site, spin):
        return createannihilate(1, site, spin, self)

    def annihilated(self, site, spin):
        return createannihilate(-1, site, spin, self)


def createannihilate(aor, site, spin, state):
    """
    Create or annihilate a fermion in our system.

    Parameters
    ----------
    aor: int
        add (+1) or remove (-1)
    site: int
        site index
    spin: int
        spin up (+1) or spin down (-1)
    state: State
        State with (psi, J, n) specifying the initial state
    """
    psi, lita, n, nu, nd = (state.v, state.lita, state.n, state.nu, state.nd)

    dnu = int(aor*(spin+1)/2)
    dnd = int(aor*(-spin+1)/2)

    litam = lin.Table(n, nu=nu + dnu, nd=nd + dnd)
    Nu, Nd = litam.Ns

    if spin == 1:
        basis = np.copy(lita.basisu)
    else:
        basis = np.copy(lita.basisd)

    # touched states indices
    index = np.nonzero(basis[:, site] == (1-aor)/2)[0]

    psim = np.zeros((Nd*Nu,), dtype=np.complex128)      # new state prealloc

    if len(index) < 1:  # nothing touched
        return psim

    if spin == -1:  # spin down
        signs = (-1)**nu*(-1)**np.sum(basis[index, 0:site], 1)
        basis[index, site] = (aor + 1)/2

        Im = lin.state2index(basis[index, :], litam.Jdv)
        for i in xrange(len(index)):
            psim[Im[i]*Nu:(Im[i] + 1)*Nu] = \
                signs[i]*psi[index[i]*Nu:(index[i]+1)*Nu]

    else:           # spin up
        signs = (-1)**np.sum(basis[index, 0:site], 1)
        basis[index, site] = (aor + 1)/2

        Im = lin.state2index(basis[index, :], litam.Juv)
        for i in xrange(len(index)):
            psim[Im[i]:np.prod(litam.Ns):litam.Nu] = \
                signs[i]*psi[index[i]:np.prod(lita.Ns):lita.Nu]

    return State(v=psim, n=n, lita=litam, nu=nu + dnu, nd=nd + dnd)


#      $$$$$$\            $$\
#     $$  __$$\           \__|
#     $$ /  \__| $$$$$$\  $$\ $$$$$$$\
#     \$$$$$$\  $$  __$$\ $$ |$$  __$$\
#      \____$$\ $$ /  $$ |$$ |$$ |  $$ |
#     $$\   $$ |$$ |  $$ |$$ |$$ |  $$ |
#     \$$$$$$  |$$$$$$$  |$$ |$$ |  $$ |
#      \______/ $$  ____/ \__|\__|  \__|
#               $$ |
#               $$ |
#               \__|


def Splus(nu, nd, n):
    return Spm(+1, nu, nd, n)


def Sminus(nu, nd, n):
    return Spm(-1, nu, nd, n)


def Spm(pom, nu, nd, n):

    Sus = Sdelta(pom, nu, n)
    Sds = Sdelta(-pom, nd, n)

    [Ndu, Nu] = Sus[0].shape
    [Ndd, Nd] = Sds[0].shape

    Sp = sparse.csr_matrix((Ndu*Ndd, Nu*Nd), dtype=int)

    for i in xrange(n):
        Sp = Sp + sparse.kron((-1)**nu*Sds[i], Sus[i], 'csr')

    return Sp


def Sdelta(aor, nup, n):

    Jp, Np, statesp = lin.states(n, nup)
    Jdp, Ndp, _ = lin.states(n, nup+aor)

    Sps = [0]*n
    # loop through all sites and in/de-creases the spin by one
    for i in xrange(n):
        Sps[i] = sparse.dok_matrix((Ndp, Np), dtype=int)  # operator for site i
        idx = statesp[:, i] == (- aor + 1)/2  # index of states with occupation
        statesmp = statesp[idx, :]            # the occupation states
        statesmp[:, i] = (aor + 1)/2          # increase spin on states
        odx = lin.state2index(statesmp, Jdp)  # get indices of resulting states
        vals = (-1)**(np.sum(statesmp[:, 0:i], axis=1, dtype=int))
        Sps[i][odx, idx] = vals

    return Sps
