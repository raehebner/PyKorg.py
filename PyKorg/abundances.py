from .atomic_data import *
import numpy as np

"""
The "alpha elements" are defined as O, Ne, Mg, Si, S, Ar, Ca, Ti, i.e. all elements with even atomic
numbers from 8-22. This definition is used by default in [`format_A_X`](@ref),
[`get_metals_H`](@ref), and [`get_alpha_H`](@ref), but can be overridden with a keyword argument.
"""
default_alpha_elements = [8, 10, 12, 14, 16, 18, 20, 22] # O to Ti
# note that changing this requires special consideration of interpolate_marcs, since we must account
# for the definition of the "alphas" they used.


def format_A_X(default_metals_H=0.0, default_alpha_H=None,
                    abundances=dict(),
                    solar_relative=True, solar_abundances=default_solar_abundances,
                    alpha_elements=default_alpha_elements):
    """
    format_A_X(default_metals_H, default_alpha_H, abundances; kwargs... )

    Returns a 92 element vector containing abundances in ``A(X)`` (``\\log_{10}(X/H) + 12``) format for
    elements from hydrogen to uranium.

    # Arguments

    You can specify abundance with these positional arguments.  All are optional, but if
    `default_alpha_H` is provided, `default_metals_H` must be as well.

      - `default_metals_H` (default: 0), i.e. [metals/H] is the ``\\log_{10}`` solar-relative abundance of elements heavier
        than He. It is overridden by `default_alpha` and `abundances` on a per-element basis.
      - `default_alpha_H` (default: same as `default_metals_H`), i.e. [alpha/H] is the ``\\log_{10}``
        solar-relative abundance of the alpha elements (See `alpha_elements`, below).
        It is overridden by `abundances` on a per-element basis.
      - `abundances` is a `Dict` mapping atomic numbers or symbols to [``X``/H] abundances.  (Set
        `solar_relative=false` to use ``A(X)`` abundances instead.) These override `default_metals_H`.
        This is the only way to specify an abundance of He that is non-solar.

    # Keyword arguments

      - `solar_relative` (default: true): When true, interpret abundances as being in \\[``X``/H\\]
        (``\\log_{10}`` solar-relative) format.  When false, interpret them as ``A(X)`` abundances, i.e.
        ``A(x) = \\log_{10}(n_X/n_\\mathrm{H}) + 12``, where ``n_X`` is the number density of ``X``.
        Note that abundances not specified default to the solar value still depend on the solar value, as
        they are set according to `default_metals_H` and `default_alpha_H`.
      - `solar_abundances` (default: `Korg.asplund_2020_solar_abundances`) is the set of solar abundances to
        use, as a vector indexed by atomic number. `Korg.asplund_2009_solar_abundances` and
        `Korg.grevesse_2007_solar_abundances` are also provided for convenience.
      - `alpha_elements` (default: [`Korg.default_alpha_elements`](@ref)): vector of atomic numbers of
        the alpha elements. (Useful since conventions vary.)
    """
    # make sure the keys of abundances are valid, and convert them to Z if they are strings
    if default_alpha_H == None:
        default_alpha_H = default_metals_H

    clean_abundances = dict()
    for (el, abund) in abundances:
        if isinstance(el, str):
            if not(el in atomic_numbers.keys()):
                raise ValueError(f"{el} isn't a valid atomic symbol.")
            elif atomic_numbers[el] in abundances.keys():
                raise ValueError(f"The abundances of {el} was specified by both atomic number and atomic symbol.")
            else:
                clean_abundances[atomic_numbers[el]] = abund
        elif isinstance(el, int):
            if not(1 <= el <= MAX_ATOMIC_NUMBER):
                raise ValueError(f"Z = {el} is not a supported atomic number.")
            else:
                clean_abundances[el] = abund
        else:
            raise ValueError(f"{el} isn't a valid element. Keys of the abundances dict should be strings or integers.")

    correct_H_abund = 0.0 if solar_relative else 12.0
    if 1 in clean_abundances.keys() and clean_abundances[1] != correct_H_abund:
        silly_abundance, silly_value = ("[H/H]", 0) if solar_relative else ("A(H)", 12)
        raise ValueError(f"{silly_abundance} set, but {silly_abundance} = {silly_value} by " +
                            "definition. Adjust \"metallicity\" and \"abundances\" to implicitly " +
                            "set the amount of H")

    #populate A(X) vector
    Aout = []
    for Z in range(1,MAX_ATOMIC_NUMBER+1):
        if Z == 1: #handle hydrogen
            Aout.append(12.0)
        elif Z in clean_abundances.keys(): #if explicitly set
            if solar_relative:
                Aout.append(clean_abundances[Z] + solar_abundances[Z-1])
            else:
                Aout.append(clean_abundances[Z])
        elif Z in alpha_elements:
            Aout.append(solar_abundances[Z-1] + default_alpha_H)
        else: #if not set, use solar value adjusted for metallicity
            Δ = default_metals_H * (Z >= 3) #only adjust for metals, not H or He
            Aout.append(solar_abundances[Z-1] + Δ)

    return Aout
        

# handle case where metallicity and alpha aren't specified but individual abundances are
# format_A_X(abundances::AbstractDict; kwargs...) = format_A_X(0, abundances; kwargs...)
# handle case where alpha isn't specified but individual abundances are
# function format_A_X(default_metallicity::R, abundances::AbstractDict; kwargs...) where R<:Real
#     format_A_X(default_metallicity, default_metallicity, abundances; kwargs...)
# end

"""
    get_metals_H(A_X; kwargs...)

Calculate [metals/H] given a vector, `A_X` of absolute abundances, ``A(X) = \\log_{10}(n_M/n_\\mathrm{H})``.
See also [`get_alpha_H`](@ref).

# Keyword Arguments

  - `solar_abundances` (default: `Korg.asplund_2020_solar_abundances`) is the set of solar abundances to
    use, as a vector indexed by atomic number. `Korg.asplund_2009_solar_abundances`,
    `Korg.grevesse_2007_solar_abundances`, and `Korg.magg_2022_solar_abundances` are also provided for
    convenience.
  - `ignore_alpha` (default: `true`): Whether or not to ignore the alpha elements when calculating
    [metals/H].  If `true`, [metals/H] is calculated using all elements heavier than He.  If `false`,
    then both carbon and the alpha elements are ignored.
  - `alpha_elements` (default: [`Korg.default_alpha_elements`](@ref)): vector of atomic numbers of
    the alpha elements. (Useful since conventions vary.)
"""
def get_metals_H(A_X,
                      solar_abundances=default_solar_abundances, ignore_alpha=True,
                      alpha_elements=default_alpha_elements):
    """
    get_metals_H(A_X; kwargs...)

    Calculate [metals/H] given a vector, `A_X` of absolute abundances, ``A(X) = \\log_{10}(n_M/n_\\mathrm{H})``.
    See also [`get_alpha_H`](@ref).

    # Keyword Arguments

      - `solar_abundances` (default: `Korg.asplund_2020_solar_abundances`) is the set of solar abundances to
        use, as a vector indexed by atomic number. `Korg.asplund_2009_solar_abundances`,
        `Korg.grevesse_2007_solar_abundances`, and `Korg.magg_2022_solar_abundances` are also provided for
        convenience.
      - `ignore_alpha` (default: `true`): Whether or not to ignore the alpha elements when calculating
        [metals/H].  If `true`, [metals/H] is calculated using all elements heavier than He.  If `false`,
        then both carbon and the alpha elements are ignored.
      - `alpha_elements` (default: [`Korg.default_alpha_elements`](@ref)): vector of atomic numbers of
        the alpha elements. (Useful since conventions vary.)
    """

    els = [Z for Z in range(3,MAX_ATOMIC_NUMBER+1) if not(Z in alpha_elements)] if ignore_alpha else range(3,MAX_ATOMIC_NUMBER+1)
    return _get_multi_X_H(A_X, els, solar_abundances)



def get_alpha_H(A_X,
                     solar_abundances=default_solar_abundances,
                     alpha_elements=default_alpha_elements):
    """
    get_alpha_H(A_X; kwargs...)

    Calculate [α/H] given a vector, `A_X` of absolute abundances, ``A(X) = \\log_{10}(n_α/n_\\mathrm{H})``.
    Here, the alpha elements are defined to be O, Ne, Mg, Si, S, Ar, Ca, Ti.  See also
    [`get_metals_H`](@ref).

    # Keyword Arguments

      - `solar_abundances` (default: `Korg.asplund_2020_solar_abundances`) is the set of solar abundances to
        use, as a vector indexed by atomic number. `Korg.asplund_2009_solar_abundances`,
        `Korg.grevesse_2007_solar_abundances`, and `Korg.magg_2022_solar_abundances` are also provided for
        convenience.
      - `alpha_elements` (default: [`Korg.default_alpha_elements`](@ref)): vector of atomic numbers of
        the alpha elements. (Useful since conventions vary.)
    """
    return _get_multi_X_H(A_X, alpha_elements, solar_abundances)




def _get_multi_X_H(A_X, Zs, solar_abundances):
    """
    Given a vector of abundances, `A_X`, get [I+J+K/H], where `Zs = [I,J,K]` is a vector of atomic
    numbers.  This is used to calculate, for example, [α/H] and [metals/H].
    """
    # there is no logsumexp in the julia stdlib, but it would make this more stable.
    # the lack of "12"s here is not a mistake.  They all cancel.
    A_mX = np.log10(sum(10**A_X[Z-1] for Z in Zs))
    A_mX_solar = np.log10(sum(10**solar_abundances[Z-1] for Z in Zs))
    return A_mX - A_mX_solar
