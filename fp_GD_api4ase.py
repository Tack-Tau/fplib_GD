import fplib_GD
import rcovdata
import sys
import numpy as np
from ase import Atom
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
# from dftpy.atom import Atom
# from dftpy.base import DirectCell
# from dftpy.constants import LEN_CONV, ENERGY_CONV, FORCE_CONV, STRESS_CONV
# from dftpy.interface import ConfigParser, OptimizeDensityConf


class fp_GD_Calculator(object):
    """Fingerprint calculator for ase"""
    
    default_parameters: Dict[str, Any] = {}
    'Default parameters'

    ignored_changes: Set[str] = set()
    'Properties of Atoms which we ignore for the purposes of cache '
    'invalidation with check_state().'

    discard_results_on_any_change = False
    'Whether we purge the results following any change in the set() method.  '
    'Most (file I/O) calculators will probably want this.'

    def __init__(self, restart=None,
                 ignore_bad_restart_file=BaseCalculator._deprecated,
                 label=None, atoms=None, directory='.',
                 **kwargs):
        
        self.atoms = None  # copy of atoms object from last calculation
        self.results = {}  # calculated properties (energy, forces, ...)
        self.parameters = None  # calculational parameters
        self._directory = None  # Initialize
        self.directory = directory
        self.prefix = None
        if label is not None:
            if self.directory == '.' and '/' in label:
                # We specified directory in label, and nothing in the diretory key
                self.label = label
            elif '/' not in label:
                # We specified our directory in the directory keyword
                # or not at all
                self.label = '/'.join((self.directory, label))
            else:
                raise ValueError('Directory redundantly specified though '
                                 'directory="{}" and label="{}".  '
                                 'Please omit "/" in label.'
                                 .format(self.directory, label))

        if self.parameters is None:
            # Use default parameters if they were not read from file:
            self.parameters = self.get_default_parameters()

        if atoms is not None:
            atoms.calc = self
            if self.atoms is not None:
                # Atoms were read from file.  Update atoms:
                if not (equal(atoms.numbers, self.atoms.numbers) and
                        (atoms.pbc == self.atoms.pbc).all()):
                    raise CalculatorError('Atoms not compatible with file')
                atoms.positions = self.atoms.positions
                atoms.cell = self.atoms.cell
    
    @property
    def directory(self) -> str:
        return self._directory

    @directory.setter
    def directory(self, directory: Union[str, os.PathLike]):
        self._directory = str(Path(directory))  # Normalize path.

    @property
    def label(self):
        if self.directory == '.':
            return self.prefix

        # Generally, label ~ directory/prefix
        #
        # We use '/' rather than os.pathsep because
        #   1) directory/prefix does not represent any actual path
        #   2) We want the same string to work the same on all platforms
        if self.prefix is None:
            return self.directory + '/'

        return '{}/{}'.format(self.directory, self.prefix)

    @label.setter
    def label(self, label):
        if label is None:
            self.directory = '.'
            self.prefix = None
            return

        tokens = label.rsplit('/', 1)
        if len(tokens) == 2:
            directory, prefix = tokens
        else:
            assert len(tokens) == 1
            directory = '.'
            prefix = tokens[0]
        if prefix == '':
            prefix = None
        self.directory = directory
        self.prefix = prefix
        
    def set_label(self, label):
        """Set label and convert label to directory and prefix.

        Examples:

        * label='abc': (directory='.', prefix='abc')
        * label='dir1/abc': (directory='dir1', prefix='abc')
        * label=None: (directory='.', prefix=None)
        """
        self.label = label


    def get_default_parameters(self):
        return Parameters(copy.deepcopy(self.default_parameters))

    def todict(self, skip_default=True):
        defaults = self.get_default_parameters()
        dct = {}
        for key, value in self.parameters.items():
            if hasattr(value, 'todict'):
                value = value.todict()
            if skip_default:
                default = defaults.get(key, '_no_default_')
                if default != '_no_default_' and equal(value, default):
                    continue
            dct[key] = value
        return dct
    
    def reset(self):
        """Clear all information from old calculation."""

        self.atoms = None
        self.results = {}
        
    def read(self, label):
        """Read atoms, parameters and calculated properties from output file.

        Read result from self.label file.  Raise ReadError if the file
        is not there.  If the file is corrupted or contains an error
        message from the calculation, a ReadError should also be
        raised.  In case of succes, these attributes must set:

        atoms: Atoms object
            The state of the atoms from last calculation.
        parameters: Parameters object
            The parameter dictionary.
        results: dict
            Calculated properties like energy and forces.

        The FileIOCalculator.read() method will typically read atoms
        and parameters and get the results dict by calling the
        read_results() method."""

        self.set_label(label)
        
    def get_atoms(self):
        if self.atoms is None:
            raise ValueError('Calculator has no atoms')
        atoms = self.atoms.copy()
        atoms.calc = self
        return atoms
    
    @classmethod
    def read_atoms(cls, restart, **kwargs):
        return cls(restart=restart, label=restart, **kwargs).get_atoms()
    
    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2, ...).

        A dictionary containing the parameters that have been changed
        is returned.

        Subclasses must implement a set() method that will look at the
        chaneged parameters and decide if a call to reset() is needed.
        If the changed parameters are harmless, like a change in
        verbosity, then there is no need to call reset().

        The special keyword 'parameters' can be used to read
        parameters from a file."""

        if 'parameters' in kwargs:
            filename = kwargs.pop('parameters')
            parameters = Parameters.read(filename)
            parameters.update(kwargs)
            kwargs = parameters

        changed_parameters = {}

        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            if key not in self.parameters or not equal(value, oldvalue):
                changed_parameters[key] = value
                self.parameters[key] = value

        if self.discard_results_on_any_change and changed_parameters:
            self.reset()
        return changed_parameters
    
    def check_state(self, atoms, tol=1e-15):
        """Check for any system changes since last calculation."""
        return compare_atoms(self.atoms, atoms, tol=tol,
                             excluded_properties=set(self.ignored_changes))
    
    def check_restart(self, atoms=None):
        if (
            self.atoms
            and np.allclose(self.atoms["lattice"], atoms.cell[:])
            and np.allclose(self.atoms["position"], atoms.get_scaled_positions())
            and self.results is not None
        ):
            return False
        else:
            return True

    def get_potential_energy(self, atoms=None, **kwargs):
        if self.check_restart(atoms):
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
            """
            if self.results is not None and self.config["MATH"]["reuse"]:
                config, others = ConfigParser(self.config, ions=ions, rhoini=self.results["density"], pseudo=pseudo, grid=grid, mp = self.mp)
                results = OptimizeDensityConf(config, others["struct"], others["E_v_Evaluator"], others["nr2"])
            else:
                config, others = ConfigParser(self.config, ions=ions, pseudo=pseudo, grid=grid, mp = self.mp)
                results = OptimizeDensityConf(config, others["struct"], others["E_v_Evaluator"], others["nr2"])
            self.results = results
            """
            self.results = results
        # energy = self.results["energypotential"]["TOTAL"].energy * ENERGY_CONV["Hartree"]["eV"]
        # energy = self.results["density"].grid.mp.asum(energy)
        energy = fplib_GD.get_fp_energy(v1)
        
        return energy

    def get_forces(self, atoms):
        """
        if self.check_restart(atoms):
            # if 'Force' not in self.config['JOB']['calctype'] :
                # self.config['JOB']['calctype'] += ' Force'
            self.get_potential_energy(atoms)
        """
        forces = fplib_GD.get_fp_forces(v1)
        # return self.results["forces"]["TOTAL"] * FORCE_CONV["Ha/Bohr"]["eV/A"]
        return forces

    def get_stress(self, atoms):
        pass
        """
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
        """
