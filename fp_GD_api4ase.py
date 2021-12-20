import fplib_GD
import rcovdata
import sys
import numpy as np
# import fplib_GD.readvasp as readvasp
# import ase.io
# import ase.units as units
from ase.atoms import Atoms
from ase.cell import Cell
# from ase.calculators.genericfileio import (CalculatorTemplate,
#                                            GenericFileIOCalculator)
# from ase.calculators.calculator import BaseCalculator, FileIOCalculator

'''
class fp_GD_Template(CalculatorTemplate):
    _label = 'fp_GD'  # Controls naming of files within calculation directory

    def __init__(self):
        super().__init__(
            name='fp_GD',
            implemented_properties=['energy', 'forces'])

        self.input_file = f'{self._label}.in'
        self.output_file = f'{self._label}.log'

    def execute(self, directory, profile) -> None:
        profile.run(directory, self.input_file, self.output_file)

    def write_input(self, directory, atoms, parameters, properties):
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)
        parameters = dict(parameters)
        pp_paths = parameters.pop('pp_paths', None)
        assert pp_paths is not None

        kw = dict(
            xc='LDA',
            smearing=None,
            kpts=None,
            raw=None,
            pps='fhi')
        kw.update(parameters)

        readvasp(
            directory=directory,
            atoms=atoms, properties=properties, parameters=kw,
            pp_paths=pp_paths)

    def read_results(self, directory):
        return io.read_abinit_outputs(directory, self._label)

class fp_GD(GenericFileIOCalculator):
    """Class for doing fp_GD calculations.

    The default parameters are very close to those that the ABINIT
    Fortran code would use.  These are the exceptions::

      calc = fp_GD(label='fp_GD', xc='LDA', ecut=400, toldfe=1e-5)
    """

    def __init__(self, *, profile=None, directory='.', **kwargs):
        """Construct fp_GD-calculator object.

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'fp_GD'.

        Examples
        ========
        Use default values:

        >>> h = Atoms('H', calculator=fp_GD(ecut=200, toldfe=0.001))
        >>> h.center(vacuum=3.0)
        >>> e = h.get_potential_energy()

        """

        if profile is None:
            profile = AbinitProfile(['fp_GD'])

        super().__init__(template=AbinitTemplate(),
                         profile=profile,
                         directory=directory,
                         parameters=kwargs)
'''

class fp_GD_Calculator(object):
    """Fingerprint calculator for ase"""
    
    def __init__(self, restart=None, atoms=None, **kwargs):
        # if parameters is None:
        #     parameters = {}
        # self.parameters = dict(parameters)
        self.atoms = {}
        self.energy = None
        self.forces = None
        # self.results = None

    
    
    def check_restart(self, atoms=None, **kwargs):
        if (
            self.atoms
            and np.allclose(self.atoms.cell[:], atoms.cell[:])
            and np.allclose(self.atoms.atoms.get_scaled_positions(), atoms.get_scaled_positions())
            and self.energy is not None
            and self.forces is not None
        ):
            return False
        else:
            return True

    def get_potential_energy(self, atoms=None, **kwargs):
        if self.check_restart(atoms):
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            types = fplib_GD.read_types('Li-mp-51.vasp')
            '''
            lattice = atoms.cell[:]
            Z = atoms.numbers
            # pos = atoms.get_positions()
            # pos /= LEN_CONV['Bohr']['Angstrom']
            pos = atoms.get_scaled_positions()
            
            if self.results is not None and len(self.atoms) > 0 :
                pseudo = self.results["pseudo"]
                if np.allclose(self.atoms["lattice"], atoms.cell[:]):
                    grid = self.results["density"].grid
                else :
                    grid = None
            else :
                pseudo = None
                grid = None

            # Save the information of structure
            self.atoms["lattice"] = lattice.copy()
            self.atoms["position"] = pos.copy()
            lattice = np.asarray(lattice).T / LEN_CONV["Bohr"]["Angstrom"]
            cell = DirectCell(lattice)
            ions = Atom(Z=Z, pos=pos, cell=cell, basis="Crystal")
            # ions.restart()
            
            if self.results is not None and self.config["MATH"]["reuse"]:
                config, others = ConfigParser(self.config, ions=ions, rhoini=self.results["density"], pseudo=pseudo, grid=grid, mp = self.mp)
                results = OptimizeDensityConf(config, others["struct"], others["E_v_Evaluator"], others["nr2"])
            else:
                config, others = ConfigParser(self.config, ions=ions, pseudo=pseudo, grid=grid, mp = self.mp)
                results = OptimizeDensityConf(config, others["struct"], others["E_v_Evaluator"], others["nr2"])
            self.results = results
            '''
        # energy = self.results["energypotential"]["TOTAL"].energy * ENERGY_CONV["Hartree"]["eV"]
        # energy = self.results["density"].grid.mp.asum(energy)
        energy = fplib_GD.get_fp_energy(lat, rxyz, types, contract = False, ntyp = 1, nx = 300, \
                                        lmax = 0, znucl = np.array([3], int), cutoff = 6.5)
        return energy

    def get_forces(self, atoms=None, **kwargs):
        if self.check_restart(atoms):
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            types = fplib_GD.read_types('Li-mp-51.vasp')
            self.get_potential_energy(atoms)
        forces = fplib_GD.get_fp_forces(lat, rxyz, types, contract = False, ntyp = 1, nx = 300, \
                                        lmax = 0, znucl = np.array([3], int), cutoff = 6.5, \
                                        iter_max = 20, step_size = 1e-4)
        return forces



    def get_stress(self, atoms=None, **kwargs):
        pass
        '''
        if self.check_restart(atoms):
            # if 'Stress' not in self.config['JOB']['calctype'] :
                # self.config['JOB']['calctype'] += ' Stress'
            self.get_potential_energy(atoms)
        # return self.results['stress']['TOTAL'] * STRESS_CONV['Ha/Bohr3']['eV/A3']
        stress_voigt = np.zeros(6)
        if "TOTAL" not in self.results["stress"]:
            # print("!WARN : NOT calculate the stress, so return zeros")
            return stress_voigt
        for i in range(3):
            stress_voigt[i] = self.results["stress"]["TOTAL"][i, i]
        stress_voigt[3] = self.results["stress"]["TOTAL"][1, 2]  # yz
        stress_voigt[4] = self.results["stress"]["TOTAL"][0, 2]  # xz
        stress_voigt[5] = self.results["stress"]["TOTAL"][0, 1]  # xy
        # stress_voigt  *= -1.0
        return stress_voigt * STRESS_CONV["Ha/Bohr3"]["eV/A3"]
        '''
