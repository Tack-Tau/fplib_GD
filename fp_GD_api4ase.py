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
from ase.calculators.calculator import Calculator

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


class fp_GD_Calculator(BaseCalculator):
    """Base-class for all ASE calculators.

    A calculator must raise PropertyNotImplementedError if asked for a
    property that it can't calculate.  So, if calculation of the
    stress tensor has not been implemented, get_stress(atoms) should
    raise PropertyNotImplementedError.  This can be achieved simply by not
    including the string 'stress' in the list implemented_properties
    which is a class member.  These are the names of the standard
    properties: 'energy', 'forces', 'stress', 'dipole', 'charges',
    'magmom' and 'magmoms'.
    """

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
        """Basic calculator implementation.

        restart: str
            Prefix for restart file.  May contain a directory. Default
            is None: don't restart.
        ignore_bad_restart_file: bool
            Deprecated, please do not use.
            Passing more than one positional argument to Calculator()
            is deprecated and will stop working in the future.
            Ignore broken or missing restart file.  By default, it is an
            error if the restart file is missing or broken.
        directory: str or PurePath
            Working directory in which to read and write files and
            perform calculations.
        label: str
            Name used for all files.  Not supported by all calculators.
            May contain a directory, but please use the directory parameter
            for that instead.
        atoms: Atoms object
            Optional Atoms object to which the calculator will be
            attached.  When restarting, atoms will get its positions and
            unit-cell updated from file.
        """
        self.atoms = None  # copy of atoms object from last calculation
        self.results = {}  # calculated properties (energy, forces, ...)
        self.parameters = None  # calculational parameters
        self._directory = None  # Initialize

        if ignore_bad_restart_file is self._deprecated:
            ignore_bad_restart_file = False
        else:
            warnings.warn(FutureWarning(
                'The keyword "ignore_bad_restart_file" is deprecated and '
                'will be removed in a future version of ASE.  Passing more '
                'than one positional argument to Calculator is also '
                'deprecated and will stop functioning in the future.  '
                'Please pass arguments by keyword (key=value) except '
                'optionally the "restart" keyword.'
            ))

        if restart is not None:
            try:
                self.read(restart)  # read parameters, atoms and results
            except ReadError:
                if ignore_bad_restart_file:
                    self.reset()
                else:
                    raise

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

        self.set(**kwargs)

        if not hasattr(self, 'name'):
            self.name = self.__class__.__name__.lower()

        if not hasattr(self, 'get_spin_polarized'):
            self.get_spin_polarized = self._deprecated_get_spin_polarized
        # XXX We are very naughty and do not call super constructor!
    
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
    
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        """Do the calculation.

        properties: list of str
            List of what needs to be calculated.  Can be any combination
            of 'energy', 'forces', 'stress', 'dipole', 'charges', 'magmom'
            and 'magmoms'.
        system_changes: list of str
            List of what has changed since last calculation.  Can be
            any combination of these six: 'positions', 'numbers', 'cell',
            'pbc', 'initial_charges' and 'initial_magmoms'.

        Subclasses need to implement this, but can ignore properties
        and system_changes if they want.  Calculated properties should
        be inserted into results dictionary like shown in this dummy
        example::

            self.results = {'energy': 0.0,
                            'forces': np.zeros((len(atoms), 3)),
                            'stress': np.zeros(6),
                            'dipole': np.zeros(3),
                            'charges': np.zeros(len(atoms)),
                            'magmom': 0.0,
                            'magmoms': np.zeros(len(atoms))}

        The subclass implementation should first call this
        implementation to set the atoms attribute and create any missing
        directories.
        """

        if atoms is not None:
            self.atoms = atoms.copy()
        if not os.path.isdir(self._directory):
            os.makedirs(self._directory)

    def calculate_numerical_forces(self, atoms, d=0.001):
        """Calculate numerical forces using finite difference.

        All atoms will be displaced by +d and -d in all directions."""

        from ase.calculators.test import numeric_force
        return np.array([[numeric_force(atoms, a, i, d)
                          for i in range(3)] for a in range(len(atoms))])

    def calculate_numerical_stress(self, atoms, d=1e-6, voigt=True):
        """Calculate numerical stress using finite difference."""
        pass

    def _deprecated_get_spin_polarized(self):
        pass
    
    def band_structure(self):
        """Create band-structure object for plotting."""
        pass

'''

class fp_GD_Calculator(object):
    """Fingerprint calculator for ase"""
    
    def __init__(self, parameters=None, atoms=None, **kwargs):
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = dict(parameters)
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
            # self.get_potential_energy(atoms)
        forces = fplib_GD.get_fp_forces(lat, rxyz, types, contract = False, ntyp = 1, nx = 300, \
                                        lmax = 0, znucl = np.array([3], int), cutoff = 6.5, \
                                        iter_max = 1, step_size = 1e-4)
        return forces



    def get_stress(self, atoms=None, **kwargs):
        pass
     
