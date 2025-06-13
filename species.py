from .atomic_data import MAX_ATOMIC_NUMBER, atomic_symbols, atomic_numbers, atomic_masses
from re import findall, split

MAX_ATOMS_PER_MOLECULE = 6

class Formula:
    def __init__(self, code):
        '''
        `Formula` takes a code and reads it into a list of atoms in the molecule (or single atom). 
        It has a property, `Formula.atoms`, which stores the internal list of atoms in the molecule.
        `Formula.atoms` is sorted into the lightest atoms first, and is always a list of 6, with some
        leading atoms being empty if the molecule is smaller. 

        Input: 
        `code` - can be an integer, list of integers, or a string. 
        A single integer constructs a single atom from its atomic number,
        and a list of integers constructs a molecule from each atomic number.
        The string can be either numerical, in which it is assumed the atomic
        numbers occupy two characters each, or a list of symbols, i.e. H2O. 
        The string does not need to be sorted.
        '''
        if isinstance(code,int):
            assert 1 <= code <= MAX_ATOMIC_NUMBER
            self.atoms = [*[0 for i in range(MAX_ATOMS_PER_MOLECULE-1)],code]

        elif isinstance(code,list) and (all(isinstance(x,int) for x in code)):
            if not len(code):
                raise ValueError("Can't construct an empty Formula")
            if len(code) <= MAX_ATOMS_PER_MOLECULE:
                assert all(1 <= i <= MAX_ATOMIC_NUMBER for i in code)
                self.atoms = [*[0 for i in range(MAX_ATOMS_PER_MOLECULE-len(code))],*code]
            else:
                raise ValueError(f"Can't construct a Formula with atoms {code}. Maximum length is {MAX_ATOMS_PER_MOLECULE}.")

        elif isinstance(code,str):
            if code in atomic_symbols:
                self.atoms = [*[0 for i in range(MAX_ATOMS_PER_MOLECULE-1)],atomic_numbers[code]]
            elif code.isdigit():
                if len(code) <= 2*MAX_ATOMS_PER_MOLECULE:
                    if len(code)%2 == 1:
                        self.atoms = Formula("0"+code).atoms
                    else:
                        self.atoms = Formula([int(code[i:i+2]) for i in range(0,len(code),2)]).atoms
                else:
                    raise ValueError(f"Can't construct a Formula with atoms {code}. Maximum length is {MAX_ATOMS_PER_MOLECULE}.")

            else:
                subcode = findall(r'[A-Z0-9]',code)
                self.atoms = []
                for s in subcode:
                    try:
                        int(s)
                        previous = self.atoms[-1]
                        for _ in range(int(s)-1):
                            self.atoms.append(previous)
                    except:
                        self.atoms.append(atomic_numbers[s])
                if len(self.atoms) > MAX_ATOMS_PER_MOLECULE:
                            raise ValueError(f"Can't construct a Formula with atoms {code}. Maximum length is {MAX_ATOMS_PER_MOLECULE}.")
                else:
                    for i in range(MAX_ATOMS_PER_MOLECULE-len(self.atoms)):
                        self.atoms.append(0)
        
        else:
            raise TypeError(f"Formula expects a string, list of integers, or an integer. Recieved {code}")
    
        self.atoms.sort()


    def get_atoms(self):
        '''
        `Formula.get_atoms()` returns a list of the nonzero atomic numbers in 'Formula.atoms'. 
        '''
        return [i for i in self.atoms if i!=0]

    def ismolecule(self):
        '''
        `Formula.ismolecule()` returns true if there is more than one atom in the formula.
        '''
        return self.atoms[-2] != 0

    def get_atom(self):
        '''
        `Formula.get_atom()` returns as a single integer the atomic number of the formula. 
        Only usable for atomic formulas, see `Formula.get_atoms()` for molecules.
        '''
        if self.ismolecule():
            raise ValueError("Can't get the atomic number of a molecule, use `.get_atoms()` instead")
        else:
            return get_atoms(self)[0]

    def n_atoms(self):
        '''
        `Formula.n_atoms()` returns the number of atoms in the formula as an integer.
        '''
        try:
            return self.atoms[::-1].index(0)
        except:
            return MAX_ATOMS_PER_MOLECULE

    def show(self):
        '''
        `Formula.show()` prints the formula.
        '''
        i = [j for j, a in enumerate(self.atoms) if a][0]
        a = self.atoms[i]
        n_of_this_atom = 1
        i += 1
        while i < len(self.atoms):
            if self.atoms[i] == a:
                n_of_this_atom += 1
            else:
                self.formula_show_helper(a,n_of_this_atom)
                a = self.atoms[i]
                n_of_this_atom = 1
            i += 1
        self.formula_show_helper(a,n_of_this_atom)
        print(sep="",end=" ")

    def formula_show_helper(self, atom, n_of_this_atom):
        if n_of_this_atom == 1:
            print(atomic_symbols[atom-1], sep= "", end="")
        else:
            print(atomic_symbols[atom-1], n_of_this_atom, sep= "", end="")

    def get_mass(self):
        '''
        `Formula.get_mass()` returns the sum of the atomic masses in the formula as a float in grams.
        '''
        return sum([atomic_masses[a-1] for a in self.get_atoms()])


roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

class Species:
    def __init__(self, code, charge=None):
        '''
        `Species` takes a Formula and charge, or a string that encodes both values. It stores these 
        internally as `Species.formula` and `Species.charge`. The charge is an integer greater than 
        or equal to -1. 

        Inputs: 
        `code` - can be either a Formula directly or a string that encodes a formula and charge. 
        If a formula is directly passed, a charge must be specified.

        `charge = None` - must be specified if a Formula is passed as a code. Otherwise, it is ignored.

        Example strings include:
         - "H I" -> H I
         - "H 1" -> H I
         - "H     1" -> H I
         - "H_1" -> H I
         - "H.I" -> H I
         - "H 2" -> H II
         - "H2" -> H₂
         - "H" -> H I
         - "01.00" → H I
         - "02.01" → He II
         - "02.1000" → He II
         - "0608" → CO I

        '''
        if charge != None:
            if charge < -1:
                raise ValueError(f"Can't construct a species with charge < -1: {code} with charge {charge}")
    
        if type(code) == Formula:
            if charge == None:
                raise ValueError("Please specify species charge.")
            self.formula = code
            self.charge = charge
        
        elif type(code) == str:
            code = code.strip(" 0")
            if code[-1] == '+':
                code = code[:-1]+" 2"
            elif code[-1] == '-':
                code = code[:-1]+" 0"
            toks = split(r'[ ._]',code)
            toks = [x for x in toks if x !=""]
            if len(toks) > 2:
                raise ValueError(f"{code} isn't a valid species code")
            self.formula = Formula(toks[0])
            if len(toks) == 1 or len(toks[1]) == 0:
                self.charge = 0 
            else:
                try:
                    self.charge = roman_numerals.index(toks[1])
                except:
                    try: 
                        float(code)
                        self.charge = int(toks[1])
                    except:
                        self.charge = int(toks[1]) - 1

        else:
            raise TypeError(f"Species expects a formula and charge, or a string encoding both. Recieved {code}")

    def ismolecule(self):
        '''
        `Species.ismolecule()` returns true if there is more than one atom in the species.  
        '''
        return self.formula.ismolecule()

    def show(self):
        '''
        `Species.show()` prints the species, including both formula and charge in ionization number notation.
        '''
        self.formula.show()
        if self.ismolecule() and self.charge == 1:
            print("+")
        elif self.ismolecule() and self.charge == 0:
            print()
        elif 0 <= self.charge <= len(roman_numerals) - 1:
            print(roman_numerals[self.charge])
        elif self.charge == -1:
            print("-")
        else:
            print(self.charge)

    def get_mass(self):
        '''
        `Species.get_mass()` returns the sum of the atomic masses in the species as a float in grams.
        '''
        return self.formula.get_mass()

    def get_atoms(self):
        '''
        `Species.get_atoms()` returns a list of the nonzero atomic numbers in 'Species.formula.atoms'. 
        '''
        return self.formula.get_atoms()

    def get_atom(self):
        '''
        `Species.get_atom()` returns as a single integer the atomic number of the formula. 
        Only usable for atomic formulas, see `Species.get_atoms()` for molecules.
        '''
        return self.formula.get_atom()

    def n_atoms(self):
        '''
        `Species.n_atoms()` returns the number of atoms in the formula as an integer.
        '''
        return self.formula.n_atoms()


def all_atomic_species():
    '''
    Returns a list of all atomic species usable by korg with charges 0-2.
    '''
    l = []
    for Z in range(MAX_ATOMIC_NUMBER):
        for c in range(3):
            if c <= Z+1:
                l.append(Species(Formula(Z+1),c))
    return l



#things to consider: 
# to include negative charges/more charges in all_atomic_species
# ending spaces after + or - will error out (this may be true in julia as well)
# cannot return things off of initializing a class as with in julia structs
# using numpy arrays rather than lists
# making get_atoms, etc. into functions removed from the classes
# 