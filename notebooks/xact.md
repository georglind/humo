

    import numpy as np
    import sys
    sys.path.append('../humo/humo')


    import humo
    import molecule
    import xactcore as xore
    import xactlintable as lin
    import util
    
    # set presisions when printing
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    
    # load matplotlib for plotting
    import matplotlib as mpl
    mpl.rcParams['text.usetex']=True
    mpl.rcParams['text.latex.unicode']=True
    import matplotlib.pylab as plt
    %matplotlib inline


    reload(humo)      # load electronic system
    reload(molecule)  # load molecule
    reload(xore)       # load exd
    reload(util)      # load utildds (for messages)




    <module 'util' from '../humo/humo/util.pyc'>



## Test exd.py

### Test build a simple dimer


    E, t, U = (0, -1, 4)
    
    H1E = E*np.zeros((2,2))
    H1T = t*np.array([[0,1],[1,0]])
    H1U = U*np.eye(2)
    
    dimer = humo.HubbardModel(H1E, H1T, H1U)


    dimer2 = xore.ChargeState(dimer, 2)
    dimer2.solve(2)

    Solved HubbardModel - ne=2 charge state - lowest 2 states - Sz=0.0 to 1.0



    # A, B, C = dd2.hamiltonian_setup(dimer, 2, 1)


    dimer2.eigenstates




    [{'E': -2.8284271247461912,
      'V': [array([ 0.054+0.265j,  0.129+0.64j ,  0.129+0.64j ,  0.054+0.265j])],
      'd': 1.0},
     {'E': -2.0000000000000013,
      'V': [array([ 0.000+0.j   , -0.608-0.361j,  0.608+0.361j,  0.000+0.j   ]),
       array([ -5.424e+14 -3.216e+14j])],
      'd': 3.0}]



### Meta-benzene molecule


    # Load a specific molecule from a mol-file
    q = molecule.PiSystem.load('molecules/metabenzene.mol', 'ohno')
    # renormalize the distances to approximately fit the molecule
    q.xyz = 1.4/0.825*q.xyz
    # construct the correct parameters for the PPP model
    q.H1E, q.H1T, q.H1U, q.U, q.W, q.zs = q.setup(q.model, q.atoms, q.links, q.xyz)


    # check the parameters
    q.inspect()

    atoms: ['C', 'C', 'C', 'C', 'C', 'C', 'S:', 'S:']
    #links: [2, 3, 2, 3, 2, 2, 1, 1]
    the core charge (z): [1 1 1 1 1 1 2 2]



    # Let us solve the system with 12 electrons
    mbenz12 = xore.ChargeState(q, 12)
    mbenz12.solve()
    print('')
    print('Energy of 10 lowest lying states: {0}'.format([round(s['E'],2) for s in mbenz12.eigenstates]))
    print('Spin-degeneracy of each level: {0}'.format([s['d'] for s in mbenz12.eigenstates]))

    Solved PiSystem - ne=12 charge state - lowest 10 states - Sz=0.0 to 1.0
    
    Energy of 10 lowest lying states: [-42.14, -41.83, -41.79, -41.49, -39.98, -39.93, -37.86, -37.82, -37.58, -37.31]
    Spin-degeneracy of each level: [3.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0]



    # Let us solve the system with 10 electrons
    mbenz10 = xore.ChargeState(q, 10)
    mbenz10.solve()
    print('')
    print('Energy of 10 lowest lying states: {0}'.format([round(s['E'],2) for s in mbenz10.eigenstates]))
    print('Spin-degeneracy of each level: {0}'.format([s['d'] for s in mbenz10.eigenstates]))

    Solved PiSystem - ne=10 charge state - lowest 10 states - Sz=0.0 to 1.0
    
    Energy of 10 lowest lying states: [-60.46, -56.77, -56.14, -56.07, -56.05, -55.24, -55.08, -54.55, -54.44, -54.43]
    Spin-degeneracy of each level: [1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 3.0]



    # Let us solve the system with 8 electrons
    mbenz8 = xore.ChargeState(q, 8)
    mbenz8.solve()
    print('')
    print('Energy of 10 lowest lying states: {0}'.format([round(s['E'],2) for s in mbenz8.eigenstates]))
    print('Spin-degeneracy of each level: {0}'.format([s['d'] for s in mbenz8.eigenstates]))

    Solved PiSystem - ne=8 charge state - lowest 10 states - Sz=0.0 to 1.0
    
    Energy of 10 lowest lying states: [-48.16, -48.13, -47.16, -46.86, -46.01, -45.92, -45.87, -45.71, -45.47, -45.3]
    Spin-degeneracy of each level: [3.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0]



    # Let us solve the system with 6 electrons
    mbenz6 = xore.ChargeState(q, 6)
    mbenz6.solve()
    print('')
    print('Energy of 10 lowest lying states: {0}'.format([round(s['E'],2) for s in mbenz6.eigenstates]))
    print('Spin-degeneracy of each level: {0}'.format([s['d'] for s in mbenz6.eigenstates]))

    Solved PiSystem - ne=6 charge state - lowest 10 states - Sz=0.0 to 1.0
    
    Energy of 10 lowest lying states: [-20.59, -17.78, -17.66, -17.41, -17.25, -16.57, -16.49, -16.46, -16.12, -16.1]
    Spin-degeneracy of each level: [1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0]



    mbenz6.eigenstates




    [{'E': -20.587069320542678,
      'V': [array([ 0.000-0.j   ,  0.000-0.j   ,  0.000-0.001j, ...,  0.007-0.015j,  0.011-0.023j,  0.003-0.005j])],
      'd': 1.0},
     {'E': -17.775149704685884,
      'V': [array([-0.000+0.j   ,  0.000-0.j   , -0.000+0.j   , ...,  0.001-0.001j, -0.001+0.001j, -0.000+0.j   ]),
       array([ -1.170e+11 +8.355e+10j,   2.556e+10 -1.825e+10j,  -6.951e+10 +4.964e+10j, ...,   2.288e+13 -1.634e+13j,   6.007e+12 -4.290e+12j,  -3.824e+11 +2.731e+11j])],
      'd': 3.0},
     {'E': -17.65857792265567,
      'V': [array([ 0.000-0.j   ,  0.001+0.001j, -0.000-0.j   , ...,  0.000+0.j   ,  0.001+0.001j,  0.000-0.j   ]),
       array([-0.001-0.001j,  0.001+0.001j, -0.000-0.j   , ..., -0.015-0.014j, -0.003-0.003j,  0.001+0.001j])],
      'd': 3.0},
     {'E': -17.405368860345437,
      'V': [array([-0.000+0.001j, -0.000+0.001j, -0.001+0.001j, ..., -0.002+0.004j, -0.003+0.007j, -0.001+0.001j])],
      'd': 1.0},
     {'E': -17.253425877700884,
      'V': [array([-0.000-0.j   , -0.001-0.001j, -0.000-0.j   , ...,  0.008+0.005j,  0.009+0.006j,  0.001+0.001j])],
      'd': 1.0},
     {'E': -16.571708392463545,
      'V': [array([-0.000+0.j   , -0.000-0.j   ,  0.001+0.j   , ..., -0.011-0.001j, -0.018-0.002j,  0.000+0.j   ]),
       array([ 0.000+0.j   , -0.001-0.j   , -0.000-0.j   , ..., -0.077-0.009j, -0.048-0.006j, -0.026-0.003j])],
      'd': 3.0},
     {'E': -16.485273108217406,
      'V': [array([ 0.000+0.j   , -0.000-0.j   , -0.001-0.001j, ..., -0.009-0.011j, -0.002-0.003j,  0.000+0.j   ]),
       array([  1.379e+09 +1.645e+09j,   7.008e+09 +8.362e+09j,   5.277e+09 +6.296e+09j, ...,  -1.444e+11 -1.723e+11j,  -7.042e+10 -8.402e+10j,  -2.095e+10 -2.499e+10j])],
      'd': 3.0},
     {'E': -16.4568467736052,
      'V': [array([ 0.000+0.j   ,  0.000+0.j   ,  0.001+0.j   , ...,  0.008+0.004j,  0.012+0.006j,  0.005+0.002j])],
      'd': 1.0},
     {'E': -16.121979621502465,
      'V': [array([ 0.002-0.j,  0.001-0.j,  0.002-0.j, ..., -0.007+0.j,  0.006-0.j,  0.002-0.j])],
      'd': 1.0},
     {'E': -16.103746402326028,
      'V': [array([ 0.000-0.j   ,  0.000-0.j   , -0.000+0.j   , ..., -0.003+0.003j, -0.005+0.004j,  0.000+0.j   ]),
       array([ -6.940e+09 +5.985e+09j,   1.131e+10 -9.758e+09j,  -3.024e+08 +2.608e+08j, ...,   6.200e+11 -5.347e+11j,   6.108e+10 -5.267e+10j,  -1.066e+11 +9.196e+10j])],
      'd': 3.0}]




    
