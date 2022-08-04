import fplib_GD
import rcovdata
import sys
import numpy as np
# import fplib_GD.readvasp as readvasp
# import ase.io
# import ase.units as units
from ase.atoms import Atoms
from ase.cell import Cell
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import CalculatorSetupError, all_changes


class fp_GD_Calculator(Calculator):
    """Fingerprint calculator for ase"""
    
    implemented_properties = [ 'energy', 'forces', 'stress' ]
    
    default_parameters = {
                          'contract': False,
                          'ntyp': 1,
                          'nx': 100,
                          'lmax': 0,
                          'cutoff': 6.0,
                          }
    
    def __init__(self, atoms=None, **kwargs):
        self._atoms = None
        self.energy = None
        self.forces = None
        self.results = {}
        # Initialize parameter dictionaries
        self._store_param_state()  # Initialize an empty parameter state
        
        Calculator.__init__(self, atoms = atoms, **kwargs)

    def set(self, **kwargs):
        """Override the set function, to test for changes in the
        fingerprint Calculator.
        """
        changed_parameters = {}

        if 'atoms' in kwargs:
            atoms = kwargs.pop('atoms')
            self.atoms = atoms  # Resets results

        changed_parameters.update(Calculator.set(self, **kwargs))

    def reset(self):
        self.atoms = None
        self.clear_results()

    def clear_results(self):
        self.results.clear()

    def calculate(self,
                  atoms = None,
                  properties = [ 'energy', 'forces', 'stress' ],
                  system_changes = tuple(all_changes),
                 ):
        """Do a fingerprint calculation in the specified directory.
        This will read VASP input files (POSCAR) and then execute 
        fp_GD.
        """
        # Check for zero-length lattice vectors and PBC
        # and that we actually have an Atoms object.
        # check_atoms(atoms)

        # self.clear_results()
        '''
        if atoms is not None:
            self.atoms = atoms.copy()
        
        if properties is None:
            properties = self.implemented_properties
        '''
        Calculator.calculate(self, atoms, properties, system_changes)
        if atoms is None:
            atoms = self.atoms
        # self.update_atoms(atoms)
        
        self.results['energy'] = self.get_potential_energy(atoms)
        self.results['forces'] = self.get_forces(atoms)
        self.results['stress'] = self.get_stress(atoms)
        
    def _store_param_state(self):
        """Store current parameter state"""
        self.param_state = dict(
            default_parameters=self.default_parameters.copy()
            )
    
    def check_restart(self, atoms=None, **kwargs):
        if (
            self.atoms
            and np.allclose(self.atoms.cell[:], atoms.cell[:])
            and np.allclose(self.atoms.get_scaled_positions(), atoms.get_scaled_positions())
            and self.energy is not None
            and self.forces is not None
        ):
            return False
        else:
            return True

    def get_potential_energy(self, atoms=None, **kwargs):
        if self.check_restart(atoms):
            # write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            types = fplib_GD.read_types('Li-mp-51.vasp')
            
        # energy = self.results["energypotential"]["TOTAL"].energy * ENERGY_CONV["Hartree"]["eV"]
        # energy = self.results["density"].grid.mp.asum(energy)
        energy = fplib_GD.get_fp_energy(lat, rxyz, types, contract = False, ntyp = 1, nx = 100, \
                                        lmax = 0, znucl = np.array([3], int), cutoff = 6.0)
        return energy

    def get_forces(self, atoms=None, **kwargs):
        if self.check_restart(atoms):
            # write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            types = fplib_GD.read_types('Li-mp-51.vasp')
            # self.get_potential_energy(atoms)
        forces = fplib_GD.get_fp_forces(lat, rxyz, types, contract = False, ntyp = 1, nx = 100, \
                                        lmax = 0, znucl = np.array([3], int), cutoff = 6.0, \
                                        iter_max = 1, step_size = 1e-4)
        return forces



    def get_stress(self, atoms=None, **kwargs):
        if self.check_restart(atoms):
            # write_vasp('input.vasp', atoms, direct=True)
            lat = atoms.cell[:]
            pos = atoms.get_scaled_positions()
            types = fplib_GD.read_types('Li-mp-51.vasp')
            # self.get_potential_energy(atoms)
        stress = fplib_GD.get_FD_stress(lat, pos, types, contract = False, ntyp = 1, nx = 100, \
                                        lmax = 0, znucl = np.array([3], int), cutoff = 6.0, \
                                        iter_max = 1, step_size = 1e-4)
        return stress
     
