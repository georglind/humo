from __future__ import division, print_function
import sys
import numpy as np
import re

# molparser function
# parses mol file string into a molecule


def parse(molfile):
    with open(molfile, "r") as mfile:
        data = mfile.readlines()

    mol = (''.join(data)).splitlines()

    try:
        res = read_mol(mol)
    except Exception as inst:
        sys.exit('Cannot read mol file format: ' + str(inst))

    return res


def read_mol(mol):
    meta = read_meta(mol)

    if not meta:
        return False

    xyz, atoms = read_atoms(mol, meta[1], meta[2])
    links = read_links(mol, meta[1] + meta[2], meta[3])

    return (meta[0], xyz, atoms, links)


def read_meta(mol):
    prog = re.compile('^([\(\w\d\)-]+)')
    res = prog.match(mol[0])

    name = False
    if res is not None:
        name = res.group(0)

    line = False
    prog = re.compile('\s+(\d+)\s+(\d+)\s.*?V2000$')
    for i, line in enumerate(mol):
        res = prog.match(line)
        if res is not None:
            n_atoms = int(res.group(1))
            n_links = int(res.group(2))
            line = i + 1
            break

    if not line:
        return False
    
    return (name, line, n_atoms, n_links)


def read_atoms(mol, start, length):
    atoms = []
    xyz = []
    for i in np.arange(start, start+length):
        atom = read_atom(mol[i])

        atoms.append(atom['element'])
        xyz.append(atom['xyz'])

    return np.array(xyz), atoms


def read_atom(line):
    atom = {'element': 'C', 'xyz': [0,0,0]}

    floats = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    element = re.findall('[A-z]+', line)

    if floats is None or element is None:
        return False

    atom['element'] = element[0]
    atom['xyz'] = [float(floats[0]), float(floats[1]), float(floats[2])]

    return atom


def read_links(mol, start, length):
    links = []
    for i in np.arange(start, start+length):
        n = read_link(mol[i])
        links.append([n[0]-1, n[1]-1, n[2]])
    return links


def read_link(line):
    return [int(n) for n in re.findall('\d+', line)]
