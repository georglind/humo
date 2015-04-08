from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform # distance between realspace points
import molskeleton as mold
import molparser as molp
from humo import HubbardModel

#     $$\      $$\           $$\                               $$\
#     $$$\    $$$ |          $$ |                              $$ |
#     $$$$\  $$$$ | $$$$$$\  $$ | $$$$$$\   $$$$$$$\ $$\   $$\ $$ | $$$$$$\
#     $$\$$\$$ $$ |$$  __$$\ $$ |$$  __$$\ $$  _____|$$ |  $$ |$$ |$$  __$$\
#     $$ \$$$  $$ |$$ /  $$ |$$ |$$$$$$$$ |$$ /      $$ |  $$ |$$ |$$$$$$$$ |
#     $$ |\$  /$$ |$$ |  $$ |$$ |$$   ____|$$ |      $$ |  $$ |$$ |$$   ____|
#     $$ | \_/ $$ |\$$$$$$  |$$ |\$$$$$$$\ \$$$$$$$\ \$$$$$$  |$$ |\$$$$$$$\
#     \__|     \__| \______/ \__| \_______| \_______| \______/ \__| \_______|


class Molecule(HubbardModel):

    def __init__(self, xyz, atoms, links):
        """
        Init function
        """
        return True

    def import_from_mol(filename):
        return True


class SigmaSystem(HubbardModel):

    def __init__(self, xyz, atoms, links, model):

        H1E, H1T, H1U = self.setup(xyz, atoms, links, model)

        HubbardModel.__init__(self, H1T)

#     $$$$$$$\  $$\  $$$$$$\                        $$\
#     $$  __$$\ \__|$$  __$$\                       $$ |
#     $$ |  $$ |$$\ $$ /  \__|$$\   $$\  $$$$$$$\ $$$$$$\    $$$$$$\  $$$$$$\$$$$\
#     $$$$$$$  |$$ |\$$$$$$\  $$ |  $$ |$$  _____|\_$$  _|  $$  __$$\ $$  _$$  _$$\
#     $$  ____/ $$ | \____$$\ $$ |  $$ |\$$$$$$\    $$ |    $$$$$$$$ |$$ / $$ / $$ |
#     $$ |      $$ |$$\   $$ |$$ |  $$ | \____$$\   $$ |$$\ $$   ____|$$ | $$ | $$ |
#     $$ |      $$ |\$$$$$$  |\$$$$$$$ |$$$$$$$  |  \$$$$  |\$$$$$$$\ $$ | $$ | $$ |
#     \__|      \__| \______/  \____$$ |\_______/    \____/  \_______|\__| \__| \__|
#                             $$\   $$ |
#                             \$$$$$$  |
#                              \______/


class PiSystem(HubbardModel):

    # model parameters
    ohno_pars = {
        'C' : {'ts': -2.22, 'td': -2.684, 'E':      0, 'U': 10.86, 'z': 1, 'tb': -2.539},
        'O' : {'ts':  -1.5, 'td':   -2.4, 'E':  -9.78, 'U': 18.89, 'z': 1},
        'O:': {'ts':  -1.5, 'td':   -2.4, 'E':  -9.78, 'U': 18.89, 'z': 1},
        'N' : {'ts': -2.05, 'td':  -2.05, 'E':   -3.4, 'U': 14.97, 'z': 1},
        'N:': {'ts': -2.05, 'td':  -2.05, 'E': -18.43, 'U':    15, 'z': 2},
        'S' : {'ts':  -3.0, 'td':   -3.0, 'E': -10.86, 'U':  9.85, 'z': 1},
        'S:': {'ts': -1.37, 'td':     [], 'E':   -7.8, 'U':     5, 'z': 2}
    }

    cm_pars = {
        'C' : {'ts': -2.22, 'td': -2.684, 'E':     0, 'U':     8, 'z': 1, 'tb': -2.539},
        'O' : {'ts': -2.22, 'td': -2.684, 'E':     0, 'U': 10.86, 'z': 1, 'tb': -2.539},
        'O:': {'ts': -2.22, 'td': -2.684, 'E':     0, 'U': 10.86, 'z': 1, 'tb': -2.539},
        'N' : {'ts': -2.22, 'td': -2.684, 'E':     0, 'U': 10.86, 'z': 1, 'tb': -2.539},
        'N:': {'ts': -2.22, 'td': -2.684, 'E':     0, 'U': 10.86, 'z': 1, 'tb': -2.539},
        'S' : {'ts': -2.22, 'td': -2.684, 'E':     0, 'U': 10.86, 'z': 1, 'tb': -2.539},
        'S:': {'ts': -2.22, 'td': -2.684, 'E':     0, 'U': 10.86, 'z': 1, 'tb': -2.539}
    }

    mn_pars = {
        'C' : {'ts': -2.22, 'td': -2.684, 'E':      0, 'U': 8.325, 'z': 1, 'tb': -2.539},
        'O' : {'ts': -2.22, 'td': -2.684, 'E':      0, 'U': 8.325, 'z': 1, 'tb': -2.539},
        'O:': {'ts': -2.22, 'td': -2.684, 'E':      0, 'U': 8.325, 'z': 1, 'tb': -2.539},
        'N' : {'ts': -2.22, 'td': -2.684, 'E':      0, 'U': 8.325, 'z': 1, 'tb': -2.539},
        'N:': {'ts': -2.22, 'td': -2.684, 'E':      0, 'U': 8.325, 'z': 1, 'tb': -2.539},
        'S' : {'ts': -2.22, 'td': -2.684, 'E':      0, 'U': 8.325, 'z': 1, 'tb': -2.539},
        'S:': {'ts': -2.22, 'td': -2.684, 'E':      0, 'U': 8.325, 'z': 1, 'tb': -2.539}
    }

    hubbard_pars = {
        'C' : {'ts': -2.22, 'td': -2.684, 'E':      0, 'U': 10.86, 'z': 1, 'tb': -2.539}
    }

    parameters = {'ohno': ohno_pars, 'mn': mn_pars, 'cm': cm_pars, 'hubbard': hubbard_pars}

    def __init__(self, xyz, atoms, links, model='mn'):

        self.xyz = xyz
        self.atoms = atoms
        self.links = links
        self.model = model.lower()

        H1E, H1T, H1U, U, W, zs = self.setup(model, atoms, links, xyz)

        self.U = U
        self.W = W
        self.zs = zs

        HubbardModel.__init__(self, H1E, H1T, H1U)

        self.HfU = self.HU_operator(self.model != 'hubbard')

    @staticmethod
    def load(file, model='mn'):
        _, xyz, atoms, links = molp.parse(file)

        connects = PiSystem.count_connections(links)

        pi = []; patoms = []; pxyz = []
        for i, atom in enumerate(atoms):
            if atom not in ['B', 'C', 'N', 'O', 'S']:
                continue

            if connects[i] > 3:
                continue

            if atom == 'N' and connects[i] == 1:
                atom = 'N:'

            if atom == 'O' and connects[i] == 1:
                atom = 'O:'

            if atom == 'S' and connects[i] == 1:
                atom = 'S:'

            # if np.abs(xyz[i, 2]) > .1:
            #     continue

            patoms.append(atom)
            pxyz.append(list(xyz[i, :]))
            pi.append(i)

        plinks = []
        for i, link in enumerate(links):

            if link[0] in pi and link[1] in pi:
                plink = [pi.index(link[0]), pi.index(link[1]), link[2]]
                plinks.append(plink)

        return PiSystem(np.array(pxyz), patoms, plinks, model)

    def inspect(self):
        print('atoms: {0}'.format(self.atoms))
        print('#links: {0}'.format(PiSystem.count_connections(self.links)))
        print('the core charge (z): {0}'.format(self.zs))

    @staticmethod
    def count_connections(links):
        """
        Count number of connections for each atom
        """
        N = np.max(np.array(links))

        connects = [0]*(N+1)

        for link in links:

            connects[link[0]] += 1
            connects[link[1]] += 1

        return connects

    def setup(self, model, atoms, links, xyz):
        """
        Procures the parts of the Hamiltonian
        """
        bonds = {1: 's', 2: 'd', 3: 't'}
        pars = self.parameters[model]

        # onsite energy
        HE = []
        for atom in atoms:
            HE.append(pars[atom]['E'])

        H1E = np.diag(HE)

        # hopping
        HT = np.zeros((len(atoms), len(atoms)), dtype=np.complex128)
        for link in links:
            t = pars['C']['t' + bonds[link[2]]]

            if (atoms[link[0]] != 'C'):
                t = pars[atoms[link[0]]]['t' + bonds[link[2]]]

            if (atoms[link[1]] != 'C'):
                t = pars[atoms[link[1]]]['t' + bonds[link[2]]]

            HT[link[0], link[1]] = t

        H1T = HT + HT.T

        # detect aromatic phenyl circles and implement them 
        # in the hopping Hamiltonian
        phenyls = PiSystem.phenyls(atoms, links, H1T)
        for cycle in phenyls:
            for i in xrange(6):
                H1T[cycle[i], cycle[(i+1) % 6]] = pars['C']['tb']
                H1T[cycle[(i+1) % 6], cycle[i]] = pars['C']['tb']

        # interactions
        U = pars['C']['U']
        W = np.array([pars[atom]['U']/pars['C']['U'] for atom in atoms])
        zs = np.array([pars[atom]['z'] for atom in atoms])

        H1U = PiSystem.HU(model, xyz, U, W)

        return (H1E, H1T, H1U, U, W, zs)

    def HU_operator(self, symmetrized=True):
        """
        Creates the interaction functional
        """
        if self.H1U is False:
            self.H1U = PiSystem.HU(self.model, self.xyz, self.U, self.W)

        HUii = np.diag(self.H1U)
        HUij = self.H1U - np.diag(HUii)

        if symmetrized:
            fcn = lambda statesu, statesd: np.sum(.5*(statesu + statesd - self.zs).dot(HUij)*(statesu + statesd - self.zs), axis=1) \
                                            + ((statesu - .5)*(statesd - .5)).dot(HUii)
        else:
            fcn = lambda statesu, statesd: np.sum(.5 * ((statesu + statesd).dot(HUij))*(statesu + statesd), axis=1) \
                                                  + ((statesu)*(statesd)).dot(HUii)

        return fcn

    @staticmethod
    def HT(model, xyz):
        """
        Semi-emperical values of hopping parameters

        From Eq[6] of The Journal of Chemical Physics 135, 194103 (2011); doi: 10.1063/1.3659294
        """
        d = squareform(pdist(xyz))

        a0 = 5.291772e-1  # Bohr radius in Angstrom
        A = -9.858271     # A parameter in eV
        xi = 1.540*a0     # Slater exponent for carbon from J. Chem. Phys. 135, 194103
        # to = -2.719577  # value at R = 1.4 eV
        SR = np.exp(-xi*d)*(1+ + 2*(xi*d)**2/5 + (xi*d)**3/15)  # overlap between p_z orbitals

        return A*SR

    @staticmethod
    def HU(model, xyz, U, W=False):
        """
        Interaction matrices based on different models
        """
        H1U = False
        for case in switch(model):
            if case('hubbard'):
                H1U = PiSystem.HU_hubbard(xyz.shape[0], U)
                break
            if case('ohno'):
                H1U = PiSystem.HU_ohno(xyz, U, W)
                break
            if case('mn'):
                H1U = PiSystem.HU_mn(xyz, U, W)
                break
            if case('cm'):
                H1U = PiSystem.HU_cm(xyz, U, W)
                break
        return H1U

    @staticmethod
    def HU_hubbard(n, U):
        """
        Simple onsite Hubbard U
        """
        return U*np.eye(n)

    @staticmethod
    def HU_ohno(xyz, U, W=None):
        """
        The empirical Ohno parameterization. Interaction on-site U and off-site.

        Parameters:
        ----------
        xyz : ndarray
            Cartesian coordinates of sites in units of Angstrom.
        U : float
            Scale of Interaction
        W : ndarray
            List of ...
        """

        d = squareform(pdist(xyz))  # matrix of distances

        alpha = 14.3996

        if W is not None:
            # U average
            Wd = (np.tile(W, (len(W), 1)) + np.tile(W, (len(W), 1)).T)/2  # average of W
            return 1/np.sqrt(1/(U * Wd)**2 + (d / alpha)**2)

        return 1/(np.sqrt(1/U**2 + (d / alpha)**2))

        # [n,dims] = size(xyz);

        # a = reshape(xyz,1,n,dims);
        # b = reshape(xyz,n,1,dims);
        # d = sqrt(sum((a(ones(n,1),:,:) - b(:,ones(n,1),:)).^2,3));

        # W = false;
        # if (length(obj.H1W)>0)
        #     [n,dims] = size(diag(obj.H1W));

        #     W1 = reshape(diag(obj.H1W),1,n);
        #     W2 = reshape(diag(obj.H1W),n,1);

        #     W = (W1(ones(n,1),:,:)+W2(:,ones(n,1),:))./2; % average of involved Us
        # end

        # % averages W if necessary
        # if (W)
        #     H1U = 14.4./sqrt((14.4./(U.*W)).^2+d.^2);
        # else
        #     % Mikkel settings
        #     H1U = U ./ sqrt(1 + 0.6117 * d.^2);

        #     % H1U = 14.4./sqrt((14.4./(U))^2+d.^2);

    @staticmethod
    def HU_mn(xyz, U, W=None):
        """
        Mataga-Nishimoto formula U = 1/(1/U + R)
        See N. Mataga and K. Nishimoto, Z. Phys. Chem. 13 (1957) 140.
        """
        d = squareform(pdist(xyz))

        alpha = 14.3996  # e**2/4 pi epsilon_0 measured in eV per Angstrom

        if W is not None:
            W = np.ones((xyz.shape[0], 1))

        return 1/(1/np.diag(U*W) + alpha*d)

    @staticmethod
    def HU_cm(xyz, U, W=None):
        """
        Alternative to the Ohno parameterization. See physRevB.55.1486
        """
        d = squareform(pdist(xyz))  # matrix of distances

        if W is not None:
            return False

        return U / (2*np.sqrt(1 + 0.6117 * d**2))

         # [n,dims] = size(xyz);

        # a = reshape(xyz,1,n,dims);
        # b = reshape(xyz,n,1,dims);
        # d = sqrt(sum((a(ones(n,1),:,:) - b(:,ones(n,1),:)).^2,3));

        # W = false;
        # if (length(obj.H1W)>0)
        #     [n,dims] = size(diag(obj.H1W));

        #     W1 = reshape(diag(obj.H1W),1,n);
        #     W2 = reshape(diag(obj.H1W),n,1);

        #     W = (W1(ones(n,1),:,:)+W2(:,ones(n,1),:))./2; % average of involved Us
        # end

        # if (W)
        #     H1U = 14.4./(2*sqrt((14.4./(U.*W)).^2+d.^2));
        # else
        #     H1U = U ./ (2*sqrt(1 + 0.6117 * d.^2));
        # end

        # for i = 1:1:length(H1U)
        #     H1W = obj.H1W;
        #     if (length(H1W)<obj.n)
        #         H1W = eye(obj.n);
        #     end
        #     H1U(i,i) = U*H1W(i,i);
        # end

    def draw(self, fig=None, ax=None, enumerate=False):
        """
        Draw the molecule
        """
        if fig is None:
            fig, ax = plt.subplots(figsize=(4, 4))

        xy = mold.draw(fig, ax, self.xyz, self.atoms, self.links)

        if enumerate is True:
            for i in xrange(self.n):
                x = xy[i, 0]
                y = xy[i, 1]
                circle = plt.Circle((x, y), 15, color='#e5e5e5', zorder=10)
                ax.add_artist(circle)
                ax.text(x, y, '{0}'.format(i), zorder=20, ha='center', va='center')

        plt.show()

    @staticmethod
    def phenyls(atoms, links, H1T):
        """
        Find all phenylic rings within a molecule.
        """
        phenyls = []
        sixrings = [cyc for cyc in Graph.cycles(H1T) if len(cyc) == 6]
        for ring in sixrings:
            carbons = True
            alternation = [0, 0]
            for i in xrange(6):
                if atoms[ring[i]] != 'C':
                    carbons = False
                    break
                for link in links:
                    if (link[0] == ring[i] and link[1] == ring[(i+1) % 6]) or \
                       (link[1] == ring[i] and link[0] == ring[(i+1) % 6]):
                        alternation[i % 2] += link[2] - 1
            if carbons and np.abs(alternation[0] - alternation[1]) == 3:
                phenyls.append(ring)

        return phenyls


class Graph:
    """
    Simple class for analyzing the molecular graph
    """
    @staticmethod
    def cycles(A):
        """
        Return a list of smaller cycles described by adjencecy matrix A.
        """
        rcycs, _ = Graph.branches(A)

        Nc = len(rcycs)

        # First we clean the cycles
        ccycs = []
        for i in xrange(Nc):
            j = 0
            while rcycs[i][j] == rcycs[i][-j-1]:
                j += 1
            ccycs.append(rcycs[i][j-1:-j])

        # retain only the smallest cycle representation of the graph
        cycles = []
        appended = [False]*Nc
        for i in xrange(Nc):
            for j in xrange(i):
                ncycs = Graph.cycleincycle(ccycs[i], ccycs[j])
                if len(ncycs) > 1:
                    cycles.append(ncycs[0]).append(ncycs[1])

        # append additional cycles
        for i in xrange(Nc):
            if not appended[i]:
                cycles.append(ccycs[i])

        return cycles

    @staticmethod
    def cycleincycle(c1, c2):
        """
        Detects whether two cycles lie wihtin one another.
        """
        if len(c1) < len(c2):
            c2, c1 = c1, c2  # c1 is always containing c2

        cn = 0

        if True:
            return []

        return cn, c2

    @staticmethod
    def branches(A):
        """
        Return list of cycles in graph described by adjacency matrix A.
        """
        N = A.shape[0]  # Number of nodes
        links = np.transpose(np.nonzero(A))  # array of links

        visited = [False]*N  # mark visited sites
        visited[0] = True

        branches = []
        asches = [[0]]
        while not all(visited):
            ches = []
            for branch in asches:
                hits = list(links[links[:, 0] == branch[-1], 1]) \
                    + list(links[links[:, 1] == branch[-1], 0])
                if len(hits) <= 2:
                    branches.append(branch)
                else:
                    for hit in set(hits):
                        if not visited[hit]:
                            ches.append(branch + [hit])
                            visited[hit] = True
                        elif len(branch) > 1 and branch[-2] != hit:
                            branches.append(branch + [hit])
            asches = ches
        branches += asches

        endtwo = [branch[-2] for branch in branches]
        endone = [branch[-1] for branch in branches]

        cycles = []
        Nb = len(branches)
        appended = [False]*Nb
        for i in xrange(Nb):
            for j in xrange(i):
                if endone[i] == endtwo[j] and endone[j] == endtwo[i]:
                    cycles.append(branches[i][:] + list(reversed(branches[j][:-2])))
                    appended[i], appended[j] = True, True

        chains = []
        for i in xrange(Nb):
            if not appended[i]:
                chains.append(branches[i])

        return cycles, chains


# This class provides the functionality we want. You only need to look at
# this if you want to know how this works. It only needs to be defined
# once, no need to muck around with its internals.
class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args:  # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False
