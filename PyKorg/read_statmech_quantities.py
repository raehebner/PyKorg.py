from .CubicSplines import construct_spline
from .species import Species, Formula
from .constants import kboltz_cgs, hplanck_cgs, kboltz_eV
from .atomic_data import atomic_symbols
from .isotopic_data import isotopic_nuclear_spin_degeneracies, isotopic_abundances
from os.path import join as joinpath
import pandas as pd
import numpy as np
import h5py


_data_dir='PyKorg\\data'

def setup_ionization_energies(fname=joinpath(_data_dir,"barklem_collet_2016","BarklemCollet2016-ionization_energies.dat")):
	'''
	Parses the table of ionization energies and returns it as a dictionary mapping elements to
	their ionization energies, `[χ₁, χ₂, χ₃]` in eV.
	'''
	d = {}
	with open(fname, 'r') as f:
		for line in f:
			line = line.strip()
			if line[0] != '#':
				toks = line.split()
				Z = toks[0]
				d[int(Z)] = [float(i) for i in toks[2:]]
	return d

def setup_partition_funcs_and_equilibrium_constants():
	"""
	Returns two dictionaries. One holding the default partition functions, and one holding the default
	log10 equilibrium constants.

	# Default partition functions

	The partition functions are custom (calculated from NIST levels) for atoms, from Barklem &
	Collet 2016 for diatomic molecules, and from [exomol](https://exomol.com) for polyatomic molecules.
	For each molecule, we include only the most abundant isotopologue.

	Note than none of these partition functions include plasma effects, e.g. via the Mihalas Hummer
	Daeppen occupation probability formalism. They are for isolated species.
	This can lead to a couple percent error for neutral alkalis and to greater errors for hydrogen in
	some atmospheres, particularly those of hot stars.

	# Default equilibrium constants

	Molecules have equilibrium constants in addition to partition functions.  For the diatomics, these
	are provided by Barklem and Collet, which extensively discusses the dissociation energies.  For
	polyatomics, we calculate these ourselves, using atomization energies calculated from the enthalpies
	of formation at 0K from [NIST's CCCDB](https://cccbdb.nist.gov/hf0k.asp).

	Korg's equilibrium constants are in terms of partial pressures, since that's what Barklem and Collet
	provide.
	"""
	mol_U_path = joinpath(_data_dir,"barklem_collet_2016","BarklemCollet2016-molecular_partition.dat")
	partition_funcs = read_Barklem_Collet_table(mol_U_path) | load_atomic_partition_functions() | load_exomol_partition_functions()
	BC_Ks = read_Barklem_Collet_logKs(joinpath(_data_dir, "barklem_collet_2016", "barklem_collet_ks.h5"))

	atomization_Es = pd.read_csv(joinpath(_data_dir, "polyatomic_partition_funcs", "atomization_energies.csv"))
	
	D00 = 0.01036 # convert from kJ/mol to eV
	atom_specs = [Species(spec) for spec in atomization_Es.spec.to_numpy()]
	atom_energies = [x * D00 for x in atomization_Es.energy.to_numpy()]

	def calculate_logK(spec, logT):
		Zs = spec.get_atoms()
		log_Us_ratio = np.log10(np.prod([partition_funcs[Species(Formula(Z),0)].interpolate(logT) for Z in Zs])/partition_funcs[spec].interpolate(logT))
		log_masses_ratio = sum([np.log10(Formula(Z).get_mass()) for Z in Zs]) - np.log10(spec.get_mass())

		T = np.exp(logT)
		log_translational_U_factor = 1.5 * np.log10(2*np.pi*kboltz_cgs * T / hplanck_cgs**2)
		# this is log number-density equilibrium constant
		log_nK = ((len(Zs) - 1) * log_translational_U_factor + 1.5 * log_masses_ratio + log_Us_ratio - D00 / (kboltz_eV * T * np.log(10)))
		# compute the log of the partial-pressure equilibrium constant, log10(pK)
		log_nK + (len(Zs) - 1) * np.log10(kboltz_cgs * T)

	polyatomic_Ks = {spec:calculate_logK(spec,energy) for spec,energy in zip(atom_specs,atom_energies)}

	equilibrium_constants = BC_Ks | polyatomic_Ks

	return partition_funcs, equilibrium_constants

def read_Barklem_Collet_logKs(fname):
	"""
	Reads the equilibrium constants from the HDF5 file produced by the Barklem and Collet 2016 paper.
	Returns a Dict from Korg.Species to Korg.CubicSplines from ln(T) to log10(K).

	As recommended by [Aquilina+ 2024](https://ui.adsabs.harvard.edu/abs/2024MNRAS.531.4538A/), we
	modify the C2 equilibrium constant reflect the dissociation energy reported by
	[Visser+ 2019](https://doi.org/10.1080/00268976.2018.1564849).
	"""
	mols = []
	for f in h5py.File(fname,'r')['mols']:
		j = str(f.decode())
		if j[-1] == '+':
			mols.append(Species(j))
		else:
			mols.append(Species(Formula(j),0))
	lnTs = np.array(h5py.File(fname,'r')['lnTs'])
	logKs = np.array(h5py.File(fname,'r')['logKs'])
	# correct the C2 equilibrium constant from Barklem and Collet to reflect the dissociation
    # energy reported by Visser+ 2019, as recommended by Aquilina+ 2024.
	C2inds = []
	for i,v in enumerate(mols):
		if v.get_atoms() == [6,6]:
			C2inds.append(i)

	BC_C2_E0 = 6.371 # value from Barklem and Collet (in table), eV
	Visser_C2_E0 = 6.24 # value from Visser+ 2019, eV
	correction =  np.log10(np.e) / (kboltz_eV * np.exp(lnTs[:,C2inds[0]])) * (Visser_C2_E0 - BC_C2_E0)

	logKs[:,C2inds[0]] += correction

	out = {}
	for i, mol in enumerate(mols):
		mask = np.isfinite(lnTs[:,i])
		out[mol] = construct_spline(lnTs[:,i][mask],logKs[:,i][mask],extrapolate=True)
	return out


def read_Barklem_Collet_table(fname):
	"""
	Constructs a Dict holding tables containing partition function or equilibrium constant values across
	ln(temperature).  Applies transform (which you can use to, e.g. change units) to each example.
	"""
	temperatures = []
	data_pairs = []
	with open(fname, 'r') as file:
		for line in file:
			if (len(line) >= 9) and ("T [K]" in line):
				temperatures.extend([float(i) for i in line[9:].strip().split()])
			elif line[0] == "#":
				continue
			else:
				row_entries = line.strip().split()
				species_code = row_entries.pop(0)
				if species_code[0:2] != "D_":
					data_pairs.append((Species(species_code),row_entries))

	return {spec:construct_spline(np.log(temperatures),[float(i) for i in vals],extrapolate=True) for spec,vals in data_pairs}

def load_atomic_partition_functions(filename=joinpath(_data_dir, "atomic_partition_funcs","partition_funcs.h5")):
	"""
	Loads saved tabulated values for atomic partition functions from disk. Returns a dictionary mapping
	species to interpolators over log(T).
	"""
	partition_funcs = {}
	logT_min = h5py.File(filename)["logT_min"][()]
	logT_step = h5py.File(filename)["logT_step"][()]
	logT_max = h5py.File(filename)["logT_max"][()]

	num = (logT_max - logT_min)/logT_step

	logTs = np.linspace(logT_min, logT_max, round(num))

	for elem in atomic_symbols:
		for ionization in ["I", "II", "III"]:
			if ((elem == "H") and ionization != "I") or ((elem == "He") and (ionization == "III")):
				continue

			spec = elem + " " + ionization 
			partition_funcs[Species(spec)] = construct_spline(logTs, h5py.File(filename)[spec], extrapolate=True)

	all_ones = np.ones(len(logTs))
	partition_funcs[Species("H II")] = construct_spline(logTs, all_ones, extrapolate=True)
	partition_funcs[Species("He III")] = construct_spline(logTs, all_ones, extrapolate=True)

	return partition_funcs


def load_exomol_partition_functions():
	"""
	Loads the exomol partition functions for polyatomic molecules from the HDF5 archive. Returns a
	dictionary mapping species to interpolators over log(T).
	"""
	out = {}
	with h5py.File(joinpath(_data_dir, "polyatomic_partition_funcs", "polyatomic_partition_funcs.h5")) as f:
		for i in f:
			j = f[i]
			spec = Species(i)

			# total nuclear spin degeneracy, which must be divided out to convert from the
			# "physics" convention for the partition function to the "astrophysics" convention
			total_g_ns = np.prod([isotopic_nuclear_spin_degeneracies[Z][max(isotopic_abundances[Z],key=isotopic_abundances[Z].get)] for Z in spec.get_atoms()])

			# at the moment, we assume all molecules are the most common isotopologue internally
			# difference isotopologues are handled by scaling the log_gf values when parsing
			# the linelist
			
			Ts, Us = j["temp"], j["partition_function"]

			out[spec] = construct_spline(np.log(Ts),Us / total_g_ns, extrapolate=True)

	return out

ionization_energies = setup_ionization_energies()
default_partition_funcs, default_equilibrium_constants = setup_partition_funcs_and_equilibrium_constants()
