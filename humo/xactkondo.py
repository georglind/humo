from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np


class Kondo:
    def __init__(self):
        return True

    def lnTkD(JLL, JLR, JRR):
        """
        Return natural log of the Kondo temperatures.
        """
        g0 = (JLL + JRR)/2

        nx = np.real(JLR)
        ny = np.imag(JLR)
        nz = (JLL - JRR)/2
        n = np.sqrt(nx**2 + ny**2 + nz**2)

        lnT1 = -1/(g0+n)
        lnT2 = -1/(g0-n)

        return lnT1, lnT2

        return True

    def poor_mans_scaling(T, JLL, JLR, JRR):
        """
        Return results of Poor man scaling over energy range T
        """
        g0 = (JLL + JRR)/2

        nx = np.real(JLR)
        ny = np.imag(JLR)
        nz = (JLL - JRR)/2
        n = np.sqrt(nx**2 + ny**2 + nz**2)

        sinphi = nx/n
        cosphi = nz/n

        T1 = np.exp(-1/(g0+n))
        T2 = np.exp(-1/(g0-n))

        g0s = (1/np.log(T/T1)+1/np.log(T/T2))/2
        ns = (1/np.log(T/T1)-1/np.log(T/T2))/2

        JLRs = ns*sinphi
        JRRs = g0s-ns*cosphi
        JLLs = g0s+ns*cosphi

        return JLLs, JLRs, JRRs