from __future__ import division, print_function
import numpy as np

#     $$\
#     $$ |
#     $$ |      $$$$$$\  $$$$$$$\   $$$$$$$\ $$$$$$$$\  $$$$$$\   $$$$$$$\
#     $$ |      \____$$\ $$  __$$\ $$  _____|\____$$  |$$  __$$\ $$  _____|
#     $$ |      $$$$$$$ |$$ |  $$ |$$ /        $$$$ _/ $$ /  $$ |\$$$$$$\
#     $$ |     $$  __$$ |$$ |  $$ |$$ |       $$  _/   $$ |  $$ | \____$$\
#     $$$$$$$$\\$$$$$$$ |$$ |  $$ |\$$$$$$$\ $$$$$$$$\ \$$$$$$  |$$$$$$$  |
#     \________|\_______|\__|  \__| \_______|\________| \______/ \_______/


def lanczos(v, H, iterations, functional=False):
    if functional:
        return flanczos(v, H[0], H[1], H[2], iterations)
    else:
        return mlanczos(v, H, iterations)


# matrix lanczos
def mlanczos(vi, H, iterations):
    """
    Lanczos inversion algorithm working with full matrices. Calculates coefficients to calculate for v.1/H.v.

    v : ndarray
        state
    H : ndarray
        Full many-body Hamiltonian
    iterations: int
        Maximum number of iterations
    """

    # prealloc
    a = np.zeros((iterations,), dtype=np.complex128)
    b = np.zeros((iterations,), dtype=np.complex128)
    # rr = np.zeros(vi.shape, dtype=np.complex128)

    # make sure v is normalized
    v = vi/np.sqrt(np.vdot(vi, vi))
    r = H.dot(v)
    a[0] = np.vdot(v, r)
    r = r - a[0]*v
    b[0] = np.sqrt(np.vdot(r, r))
    pv = v
    v = r/b[0]

    # iteration
    for i in xrange(1, iterations):
        r = H.dot(v) - b[i-1]*pv
        a[i] = np.vdot(r, v)
        r = r - a[i]*v
        b[i] = np.sqrt(np.vdot(r, r))
        pv = v
        v = r/b[i]

    return (a, b, np.dot(v, vi))


def flanczos(vi, HTu, HTd, HU, iterations):
    """
    Lanczos inversion algorithm working with functions. Calculates coefficients to calculate for v.1/H.v.

    v : ndarray
        state
    H : function
        Full many-body Hamiltonian
    iterations: int
        Maximum number of iterations
    """

    # prealloc
    a = np.zeros((iterations,))
    b = np.zeros((iterations,))

    # make sure v is normalized
    v = vi/np.sqrt(np.vdot(vi, vi))
    r = Hv(HTu, HTd, HU, v)

    a[0] = np.vdot(v, r)
    r = r - a[0]*v
    b[0] = np.sqrt(np.vdot(r, r))
    pv = v
    v = r/b[0]

    # iteration
    for i in xrange(1, iterations):
        r = Hv(HTu, HTd, HU, v) - b[i-1]*pv
        a[i] = np.vdot(r, v)
        r = r - a[i]*v
        b[i] = np.sqrt(np.vdot(r, r))
        pv = v
        v = r/b[i]

    return (a, b, np.dot(v, vi))


def Hv(S1, S2, C, v):
    """"
    Hamiltonian funcitonal

    Parameters
    ----------
    S1 : ndarray
        Hopping for spin system 1
    S2 : ndarray
        Hopping for spin system 2
    C : ndarray
        vector with diagonal elements
    v : ndarray
        input state to act on with the hamiltonian

    Returns
    -------
    ndarray
        result state
    """
    n1 = S1.shape[0]
    n2 = S2.shape[0]

    y = np.zeros(v.shape, dtype=v.dtype)

    for i in xrange(n1):
        y[i*n2:(i+1)*n2] = S2.dot(v[i*n2:(i+1)*n2])

    for i in xrange(n2):
        y[i:n2:] += S1.dot(v[i:n2:])

    return y + C*v


def continued_fraction(a, b, maxiter=None):
    """
    Evaluate matrix elements from Lanczos inversion coefficients.

    Parameters
    ----------
    a : ndarray
        the a coefficients
    b : ndarray
        the b coefficients
    maxiter : int
        maximum number of iterations
    """
    cutoff = len(a)
    if maxiter is not None:
        cutoff = np.min(cutoff, maxiter)

    f = np.ones(a.shape)
    for i in xrange(cutoff-1, -1, -1):
        f = a[i] - b[i]**2/f

    return 1/f
