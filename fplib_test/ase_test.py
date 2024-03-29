import os
import sys
import numpy as np
import fplib_GD
import ase.io
from ase.optimize import BFGS, LBFGS, BFGSLineSearch, QuasiNewton, FIRE
from ase.optimize.sciopt import SciPyFminBFGS, SciPyFminCG
from ase.constraints import StrainFilter, UnitCellFilter
from ase.io.trajectory import Trajectory

from fp_GD_api4ase import fp_GD_Calculator

atoms = ase.io.read('.'+'/'+'POSCAR')

trajfile = 'opt.traj'

calc = fp_GD_Calculator(
            cutoff = 6.0,
            contract = False,
            znucl = np.array([3], int),
            lmax = 0,
            nx = 300,
            ntyp = 1
            )
# atoms.set_calculator(calc)
atoms.calc = calc
print ("fp_energy:\n", atoms.get_potential_energy())
print ("fp_forces:\n", atoms.get_forces())

############################## Relaxation type ##############################
'''
Ref : 
    https://wiki.fysik.dtu.dk/ase/ase/optimize.html#module-optimize
    https://wiki.fysik.dtu.dk/ase/ase/constraints.html
'''

af = atoms
# af = StrainFilter(atoms)
# af = UnitCellFilter(atoms)

############################## Relaxation method ##############################
opt = BFGS(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = FIRE(af, maxstep = 1.e-1, trajectory = trajfile)
# opt = LBFGS(af, maxstep = 1.e-1, trajectory = trajfile, memory = 10, use_line_search = True)
# opt = LBFGS(af, maxstep = 1.e-1, trajectory = trajfile, memory = 10, use_line_search = False)
# opt = SciPyFminCG(af, trajectory = trajfile)
# opt = SciPyFminBFGS(af, trajectory = trajfile)

opt.run(fmax = 1.e-3)

traj = Trajectory(trajfile)
ase.io.write('opt.vasp', traj[-1], direct = True, long_format=True, vasp5 = True)



