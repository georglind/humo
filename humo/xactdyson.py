from __future__ import division, print_function
# import matplotlib.pyplot as plt
import numpy as np

import xactcore as xore
import molskeleton
import matplotlib.pylab as plt


# Feynman-Dyson orbitals

def FDOs(csf, csi, ini=(0, 1), fini=(0, 10)):
    # pm = csf.ne - csi.ne
    csi.solve(cache=True, ns=ini[1])
    csf.solve(cache=True, ns=fini[1])

    ini = (ini[0], len(csi.eigenstates))  # update boundary
    fini = (fini[0], len(csf.eigenstates))  # update boundary

    orbs = {}

    for i in xrange(ini[0], ini[1]):  # each state

        orbs[i] = {}

        for ii in xrange(len(csi.eigenstates[i]['V'])):  # each degenerate state

            orbs[i][ii] = {}

            ei = csi.eigenstates[i]['V'][ii]
            ni = csi.eigenstates[i]['n'][ii]

            for f in xrange(fini[0], fini[1]):  # each pmstate

                orbs[i][ii][f] = {}

                for ff in xrange(len(csf.eigenstates[f]['V'])):  # each degenerate pmstate

                    ef = csf.eigenstates[f]['V'][ff]
                    nf = csf.eigenstates[f]['n'][ff]

                    # calculate the dyson orbital
                    orbs[i][ii][f][ff] = \
                        FDO(xore.State(ef, csf.humo.n, nu=nf[0], nd=nf[1]),
                            xore.State(ei, csi.humo.n, nu=ni[0], nd=ni[0]))

    return orbs


def FDO(sf, si):
    """
    Return the Feynman Dyson overlap between state |f> and state |i>.

    Parameters
    ----------
    f : Object
        State object
    i : Object
        State object
    """
    dn = np.array([sf.nu - si.nu, sf.nd - si.nd])

    if np.sum(np.abs(dn)) != 1:
        return False

    pm = np.sum(dn)
    ds = pm*(dn[0] - dn[1])

    orb = np.zeros((sf.n,), dtype=np.complex128)
    for j in xrange(sf.n):
        m = xore.createannihilate(pm, j, ds, si)  # intermediary
        orb[j] = np.vdot(sf.v, m.v)

    return orb


def drawFDO(humo, orb, fig=None, ax=None):
    assert hasattr(humo, 'xyz')
    assert hasattr(humo, 'atoms')
    assert hasattr(humo, 'links')

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    xy = molskeleton.draw(fig, ax, humo.xyz, humo.atoms, humo.links)

    for i in xrange(humo.n):
        color = 'r' if np.real(orb[i]) < 0 else 'b'
        # print(tuple(xy[i, 0:2]))
        # print(orb[i])
        circle = plt.Circle(tuple(xy[i, 0:2]), np.abs(orb[i])*60, color=color, zorder=10)
        ax.add_artist(circle)


def drawFDOS(humo, orbs, Esi, Esf, superscript=''):
    """
    Draw many orbitals
    """
    if len(orbs) > 1:
        figs = [0]*len(orbs)
        axs = [0]*len(orbs)
        for i in xrange(len(orbs)):
            figs[i], axs[i] = drawFDOS(humo, [orbs[i]], [Esi[i]], Esf, superscript=superscript)
        return figs, axs

    fig, axes = plt.subplots(len(orbs[0]), len(orbs[0][0]))
    fig.set_figheight(len(orbs[0])*2)
    fig.set_figwidth(len(orbs[0][0])*2)

    axes = np.atleast_2d(axes)

    for ii in xrange(len(orbs[0])):
        for f in xrange(len(orbs[0][ii])):
            for ff in xrange(len(orbs[0][ii][f])):
                if orbs[0][ii][f][ff] is not False:
                    drawFDO(humo, orbs[0][ii][f][ff], fig, axes[ii, f])
                    axes[ii, f].set_title(r'$\varepsilon_{{{0}}}^{{{1}}} = {2:.2f}$'.format(f, superscript, Esf[f] - Esi[0]))

    return fig, axes


# Extend the ChargeState
def ChargeStateFDOs(self, pm, ini=None, fini=None):
    csf = xore.ChargeState(self.humo, self.ne + pm)

    if ini is None:
        ini = (0, 1)

    if fini is None:
        orbs = FDOs(csf, self)
    else:
        orbs = FDOs(csf, self, ini, fini)

    if not hasattr(self, 'FDorbs'):
        self.FDorbs = {}
        self.FDes = {}

    self.FDorbs[pm] = orbs
    self.FDes[pm] = [s['E'] for s in csf.eigenstates]

    return orbs

xore.ChargeState.FDOs = ChargeStateFDOs


# extend the ChargeState
def ChargeStateDrawFDOs(self, pm, ini=None, fini=None):
    orbs = self.FDOs(pm, ini, fini)

    Esi = [pm*s['E'] for s in self.eigenstates]
    Esf = [pm*E for E in self.FDes[pm]]

    hp = 'p' if pm > 0 else 'h'

    fig, ax = drawFDOS(self.humo, orbs, Esi, Esf, hp)

    return fig, ax

xore.ChargeState.drawFDOs = ChargeStateDrawFDOs
