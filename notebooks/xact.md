    import numpy as np
    import sys
    sys.path.append('../humo/humo')  # quick and dirty import

    import humo
    import molecule
    import xactcore as xore
    import xactlintable as lin
    import util
    
    # set presisions when printing
    np.set_printoptions(precision=3, suppress=True, linewidth=200)

Returns

    <module 'util' from '../humo/humo/util.pyc'>

## Test xact.py

### Build a simple dimer

    E, t, U = (0, -1, 4)  # initial setup
    
    # Build matrices
    H1E = E*np.zeros((2,2))
    H1T = t*np.array([[0,1],[1,0]])
    H1U = U*np.eye(2)
    
    # Construct the dimer model
    dimer = humo.HubbardModel(H1E, H1T, H1U)

    # construct a Charge State with two electrons
    dimer2 = xore.ChargeState(dimer, 2)
    dimer2.solve(2)

Returns

    Solved HubbardModel - ne=2 charge state - lowest 2 states - Sz=0.0 to 1.0

The solutions can be accessed as ```dimer2.eigenstates```:

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

Returns

    atoms: ['C', 'C', 'C', 'C', 'C', 'C', 'S:', 'S:']
    #links: [2, 3, 2, 3, 2, 2, 1, 1]
    the core charge (z): [1 1 1 1 1 1 2 2]

#### Solving a system with 12 electrons

    # Let us solve the system with 12 electrons
    mbenz12 = xore.ChargeState(q, 12)
    mbenz12.solve()
    print('Energy of 10 lowest lying states: {0}'.format([round(s['E'],2) for s in mbenz12.eigenstates]))
    print('Spin-degeneracy of each level: {0}'.format([s['d'] for s in mbenz12.eigenstates]))

Returns

    Solved PiSystem - ne=12 charge state - lowest 10 states - Sz=0.0 to 1.0
    Energy of 10 lowest lying states: [-42.14, -41.83, -41.79, -41.49, -39.98, -39.93, -37.86, -37.82, -37.58, -37.31]
    Spin-degeneracy of each level: [3.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0]

#### Solving a system with 10 electrons

    # Let us solve the system with 10 electrons
    mbenz10 = xore.ChargeState(q, 10)
    mbenz10.solve()
    print('')
    print('Energy of 10 lowest lying states: {0}'.format([round(s['E'],2) for s in mbenz10.eigenstates]))
    print('Spin-degeneracy of each level: {0}'.format([s['d'] for s in mbenz10.eigenstates]))

Returns 

    Solved PiSystem - ne=10 charge state - lowest 10 states - Sz=0.0 to 1.0
    
    Energy of 10 lowest lying states: [-60.46, -56.77, -56.14, -56.07, -56.05, -55.24, -55.08, -54.55, -54.44, -54.43]
    Spin-degeneracy of each level: [1.0, 3.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 3.0]


#### Solving with 8 electrons

    mbenz8 = xore.ChargeState(q, 8)
    mbenz8.solve()
    print('')
    print('Energy of 10 lowest lying states: {0}'.format([round(s['E'],2) for s in mbenz8.eigenstates]))
    print('Spin-degeneracy of each level: {0}'.format([s['d'] for s in mbenz8.eigenstates]))

Returns

    Solved PiSystem - ne=8 charge state - lowest 10 states - Sz=0.0 to 1.0
    
    Energy of 10 lowest lying states: [-48.16, -48.13, -47.16, -46.86, -46.01, -45.92, -45.87, -45.71, -45.47, -45.3]
    Spin-degeneracy of each level: [3.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0]

#### Solving a system with 6 electrons

    # Let us solve the system with 6 electrons
    mbenz6 = xore.ChargeState(q, 6)
    mbenz6.solve()
    print('')
    print('Energy of 10 lowest lying states: {0}'.format([round(s['E'],2) for s in mbenz6.eigenstates]))
    print('Spin-degeneracy of each level: {0}'.format([s['d'] for s in mbenz6.eigenstates]))

Returns 

    Solved PiSystem - ne=6 charge state - lowest 10 states - Sz=0.0 to 1.0
    
    Energy of 10 lowest lying states: [-20.59, -17.78, -17.66, -17.41, -17.25, -16.57, -16.49, -16.46, -16.12, -16.1]
    Spin-degeneracy of each level: [1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0]




    
