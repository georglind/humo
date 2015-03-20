#     $$\   $$\                     $$\                 $$\ 
#     $$ |  $$ |                    $$ |                $$ |
#     $$ |  $$ |$$\   $$\  $$$$$$$\ $$ |  $$\  $$$$$$\  $$ |
#     $$$$$$$$ |$$ |  $$ |$$  _____|$$ | $$  |$$  __$$\ $$ |
#     $$  __$$ |$$ |  $$ |$$ /      $$$$$$  / $$$$$$$$ |$$ |
#     $$ |  $$ |$$ |  $$ |$$ |      $$  _$$<  $$   ____|$$ |
#     $$ |  $$ |\$$$$$$  |\$$$$$$$\ $$ | \$$\ \$$$$$$$\ $$ |
#     \__|  \__| \______/  \_______|\__|  \__| \_______|\__|

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np


def transmission(energies, H, GL, GR, RealSigLR = None):
    """
    Huckel transmission from the Landauer formula T(E) = Tr(G Gamma_L G^\dagger Gamma_R)

    Parameters
    ----------
    energies: ndarray
        list of energies
    H : ndarray
        Huckel Hamiltonian
    GL : ndarray
        vector of couplings to left lead
    GR : ndarray
        vector of couplings to right lead
    RealSigLR : ndarray
        vector containing the real part of the self-energy
    """
    # left and right coupling matrices
    GamL = -np.diag(GL)
    GamR = -np.diag(GR)

    # Lead self energies Sigma_L + Sigma_R here assumed entirely imaginary
    SigLR = -1j/2*(GamL + GamR)

    if (RealSigLR is not None):
        SigLR = SigLR + RealSigLR

    T = np.zeros((len(energies)), dtype=np.float64)

    for i, E in enumerate(energies):
        try:
            G = np.linalg.inv(E*np.eye(H.shape[0]) - H - SigLR)
            T[i] = np.abs(np.trace(GamL.dot(G).dot(GamR).dot(np.conjugate(G.T))))
        except np.linalg.LinAlgError:
            print("Green's function not defined at E={0}".format(E))

    return T


def seebeck_coefficient(Es, H, nm, eta, T = 0):
    """ 
    Seebeck coefficient from the Mott relation S = d ln(T)/ d (units of V/K)
    """
    T = Huckel.transmission(Es, H, nm, eta)
    S = np.pi**2/3*1/(11604)*T/(11604)*np.diff(np.log(T))/(Es[1] - Es[0])

    return S