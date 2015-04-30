from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np


class Kondo:
    def __init__(self):
        return True

    def lnTkD(JLL, JLR, JRR):
        return True

        # function [T1 T2] = lnTkD(JLL,JLR,JRR)
        #     %  g0
        #     g0 = (JLL+JRR)./2;
        #     %  n
        #     nx = real(JLR);
        #     ny = imag(JLR);
        #     nz = (JLL-JRR)./2;
        #     n = sqrt(nx.^2+ny.^2+nz.^2);

        #     % Kondo temperatures
        #     T1 = -1./(g0+n);
        #     T2 = -1./(g0-n);
        # end

    def poor_mans_scaling(T, JLL, JLR, JRR):
        return True
        # function [JsLL,JsLR,JsRR] = scaling(T,JLL,JLR,JRR)

        #     g0 = (JLL+JRR)/2;
        #     nx = real(JLR);
        #     ny = imag(JLR);
        #     nz = (JLL-JRR)/2;
        #     n = sqrt(nx^2+ny^2+nz^2);

        #     sinphi = nx/n;
        #     cosphi = nz/n

        #     T1 = exp(-1./(g0+n))
        #     T2 = exp(-1./(g0-n))

        #     g0s = 1/2*(1./log(T./T1)+1./log(T./T2));
        #     ns  = 1/2*(1./log(T./T1)-1./log(T./T2));

        #     JsLR = ns.*sinphi;
        #     JsRR = (g0s-ns.*cosphi);
        #     JsLL = (g0s+ns.*cosphi);

        # end
