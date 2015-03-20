from __future__ import division, print_function
import numpy as np
import molparser


def draw_mol(fig, ax, molfile):
    """
    Parse a mol file and draw it
    """
    mol = molparser.parse(molfile)

    if not mol:
        return False

    name, xyz, atoms, links = mol
    xy = draw(fig, ax, xyz, atoms, links)

    return xy


def draw(fig, ax, xyc, atoms, links):
    """
    Draw a skeleton structure of a molecule
    """
    # axis
    w, h = axis_size(fig, ax)
    axsize = np.max([w, h])

    # positions
    xyz = np.array(xyc, copy=True)  # copy

    xmax = np.max(xyz[:, 0])
    xmin = np.min(xyz[:, 0])

    ymax = np.max(xyz[:, 1])
    ymin = np.min(xyz[:, 1])

    dm = max([xmax - xmin, ymax - ymin])

    xoff = (xmax+xmin)/2
    yoff = (ymax+ymin)/2

    # norm
    norm = 100/np.sqrt((xyz[links[0][0], 0] - xyz[links[0][1], 0])**2 + (xyz[links[0][0], 1] - xyz[links[0][1], 1])**2)

    # scale
    scale = axsize/(dm*norm) * 200/3.2
    fontsize = scale*(28 - 2) + 2

    # atoms
    for i, atom in enumerate(atoms):
        xyz[i,0] = (xyz[i, 0] - xoff)*norm
        xyz[i,1] = (xyz[i, 1] - yoff)*norm

        if atom != 'C':
            ax.text(xyz[i, 0]-scale*(10-9)-9, xyz[i, 1]-scale*(10-9)-9, atom, fontsize=fontsize)

    # links
    for i, link in enumerate(links):
        p1 = [xyz[link[0], 0], xyz[link[0], 1]]
        p2 = [xyz[link[1], 0], xyz[link[1], 1]]

        am1 = (18 - 16)/scale + 16
        if (atoms[link[0]] == 'C'):
            am1 = 2

        am2 = (18 - 16)/scale + 16
        if (atoms[link[1]] == 'C'):
            am2 = 2

        p1, p2 = shorten(p1, p2, am1, am2)

        if link[2] == 2:
            q1, q2 = shift(p1, p2, 5)
            r1, r2 = shift(p1, p2, -5)
            ax.plot([q1[0], q2[0]], [q1[1], q2[1]], 'black', linewidth=1.4*scale)
            ax.plot([r1[0], r2[0]], [r1[1], r2[1]], 'black', linewidth=1.4*scale)

        if link[2] == 3:
            q1, q2 = shift(p1, p2, 9)
            r1, r2 = shift(p1, p2, -9)
            ax.plot([q1[0], q2[0]], [q1[1], q2[1]], 'black', linewidth=1.4*scale)
            ax.plot([r1[0], r2[0]], [r1[1], r2[1]], 'black', linewidth=1.4*scale)

        if link[2] != 2:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'black', linewidth=1.4*scale)

    ax.axis('equal')
    ax.axis('off')

    return xyz


def axis_size(fig, ax):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    return (width, height)


def shorten(p1, p2, am1, am2):
    """
    Return coordinates of line from p1 to p2 shortened by amount.
    """
    xavg = (p1[0] + p2[0])/2
    yavg = (p1[1] + p2[1])/2

    lp = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    lq1 = lp/2 - am1
    lq2 = lp/2 - am2

    q1 = [xavg + (p1[0] - xavg)*2*lq1/lp, yavg + (p1[1] - yavg)*2*lq1/lp]
    q2 = [xavg + (p2[0] - xavg)*2*lq2/lp, yavg + (p2[1] - yavg)*2*lq2/lp]

    return (q1, q2)


def shift(p1, p2, amount):
    """
    Return coordinates of line from p1 to p2 shifted perpendicular to the line by amount.
    """
    lp = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    dx = (p2[0] - p1[0])/lp
    dy = (p2[1] - p1[1])/lp

    q1 = [p1[0] - dy*amount, p1[1] + dx*amount]
    q2 = [p2[0] - dy*amount, p2[1] + dx*amount]

    return (q1, q2)
