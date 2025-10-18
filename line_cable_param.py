"""
Optimized Python implementation for evaluating Z and Y matrices 
for transmission lines and underground cables.

Based on LineCableParam2025.m
Compatible with Python 3.14+ (no Numba dependency)
Optimized with NumPy vectorization and efficient algorithms.

Author: Converted from Wolfram Mathematica
Date: 2025
"""

import numpy as np
from scipy.special import iv, kv  # Modified Bessel functions
import warnings

# Constants
MU0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
EPSILON0 = 8.854e-12     # Permittivity of free space (F/m)

# ============================================================================
# IMPEDANCE FUNCTIONS
# ============================================================================

def zint_tubo(omega, rhoc, rf, rint, mur=1.0, mu=MU0):
    """
    Internal impedance of tubular conductors.
    
    Parameters:
    -----------
    omega : float or complex
        Angular frequency (rad/s)
    rhoc : float
        Resistivity of conductor (Ohm·m)
    rf : float
        External radius (m)
    rint : float
        Internal radius (m)
    mur : float
        Relative permeability (default 1.0)
    mu : float
        Permeability (default MU0)
    
    Returns:
    --------
    complex : Internal impedance (Ohm/m)
    """
    eta_c = np.sqrt(1j * omega * mur * mu / rhoc)
    ri = rint + 1e-6  # Small offset to avoid singularity
    
    # Modified Bessel functions
    k1_ri = kv(1, eta_c * ri)
    i1_rf = iv(1, eta_c * rf)
    k1_rf = kv(1, eta_c * rf)
    i1_ri = iv(1, eta_c * ri)
    
    k0_rf = kv(0, eta_c * rf)
    i0_rf = iv(0, eta_c * rf)
    
    den = k1_ri * i1_rf - k1_rf * i1_ri
    num = k1_ri * i0_rf + k0_rf * i1_ri
    
    return (rhoc * eta_c * num) / (2 * np.pi * rf * den)


def zin(omega, rho_pr, r_pr, mur=90.0, mu=MU0):
    """
    Internal impedance of cylindrical conductors.
    
    Parameters:
    -----------
    omega : float or complex
        Angular frequency (rad/s)
    rho_pr : float
        Resistivity (Ohm·m)
    r_pr : float
        Conductor radius (m)
    mur : float
        Relative permeability (default 90 for steel)
    mu : float
        Permeability (default MU0)
    
    Returns:
    --------
    complex : Internal impedance (Ohm/m)
    """
    eta_pr = np.sqrt(1j * omega * mu * mur / rho_pr)
    i0 = iv(0, eta_pr * r_pr)
    i1 = iv(1, eta_pr * r_pr)
    
    return (eta_pr * rho_pr * i0) / (2 * np.pi * r_pr * i1)


def z2(omega, rcond, rins1, mur=1.0, mu=MU0):
    """
    Impedance between conductor and first insulation layer.
    
    Parameters:
    -----------
    omega : float
        Angular frequency (rad/s)
    rcond : float
        Conductor outer radius (m)
    rins1 : float
        First insulation outer radius (m)
    mur : float
        Relative permeability
    mu : float
        Permeability
    
    Returns:
    --------
    complex : Impedance (Ohm/m)
    """
    return (1j * omega * mur * mu / (2 * np.pi)) * np.log(rins1 / rcond)


def z3(omega, rins1, rsheath, rho_sheath, mur=1.0, mu=MU0):
    """
    Impedance component for sheath.
    
    Parameters:
    -----------
    omega : float
        Angular frequency (rad/s)
    rins1 : float
        Inner radius of sheath (m)
    rsheath : float
        Outer radius of sheath (m)
    rho_sheath : float
        Sheath resistivity (Ohm·m)
    mur : float
        Relative permeability
    mu : float
        Permeability
    
    Returns:
    --------
    complex : Impedance component (Ohm/m)
    """
    eta_sheath = np.sqrt(1j * omega * mur * mu / rho_sheath)
    
    i1_rsheath = iv(1, eta_sheath * rsheath)
    k1_rins1 = kv(1, eta_sheath * rins1)
    i1_rins1 = iv(1, eta_sheath * rins1)
    k1_rsheath = kv(1, eta_sheath * rsheath)
    
    den = i1_rsheath * k1_rins1 - i1_rins1 * k1_rsheath
    
    i0_rins1 = iv(0, eta_sheath * rins1)
    k0_rins1 = kv(0, eta_sheath * rins1)
    
    num = i0_rins1 * k1_rsheath + k0_rins1 * i1_rsheath
    
    return (rho_sheath * eta_sheath * num) / (2 * np.pi * rins1 * den)


def z4(omega, rins1, rsheath, rho_sheath, mur, mu=MU0):
    """
    Mutual impedance component for sheath.
    
    Parameters:
    -----------
    omega : float
        Angular frequency (rad/s)
    rins1 : float
        Inner radius of sheath (m)
    rsheath : float
        Outer radius of sheath (m)
    rho_sheath : float
        Sheath resistivity (Ohm·m)
    mur : float
        Relative permeability
    mu : float
        Permeability
    
    Returns:
    --------
    complex : Mutual impedance component (Ohm/m)
    """
    eta_sheath = np.sqrt(1j * omega * mur * mu / rho_sheath)
    
    i1_rsheath = iv(1, eta_sheath * rsheath)
    k1_rins1 = kv(1, eta_sheath * rins1)
    i1_rins1 = iv(1, eta_sheath * rins1)
    k1_rsheath = kv(1, eta_sheath * rsheath)
    
    den = i1_rsheath * k1_rins1 - i1_rins1 * k1_rsheath
    
    return rho_sheath / (2 * np.pi * rins1 * rsheath * den)


def z6(omega, rsheath, rins2, mur, mu=MU0):
    """
    Impedance between sheath and second insulation layer.
    
    Parameters:
    -----------
    omega : float
        Angular frequency (rad/s)
    rsheath : float
        Sheath outer radius (m)
    rins2 : float
        Second insulation outer radius (m)
    mur : float
        Relative permeability
    mu : float
        Permeability
    
    Returns:
    --------
    complex : Impedance (Ohm/m)
    """
    return (1j * omega * mur * mu / (2 * np.pi)) * np.log(rins2 / rsheath)


def z_solo(omega, r, h1, h2, sigma_solo, mu=MU0):
    """
    Ground impedance using Carson's equations with Bessel functions.
    
    Parameters:
    -----------
    omega : float
        Angular frequency (rad/s)
    r : float
        Horizontal separation (m)
    h1, h2 : float
        Heights of conductors above ground (m)
    sigma_solo : float
        Ground conductivity (S/m)
    mu : float
        Permeability (default MU0)
    
    Returns:
    --------
    complex : Ground impedance (Ohm/m)
    """
    eta_solo = np.sqrt(1j * omega * mu * sigma_solo)
    
    dist_direct = np.sqrt(r**2 + (h1 - h2)**2)
    dist_image = np.sqrt(r**2 + (h1 + h2)**2)
    
    k0_direct = kv(0, eta_solo * dist_direct)
    k2_image = kv(2, eta_solo * dist_image)
    
    # Exponential correction term
    exp_term = np.exp(-(h1 + h2) * eta_solo) * (1 + (h1 + h2) * eta_solo)
    correction = 2 * exp_term / (eta_solo**2 * (r**2 + (h1 + h2)**2))
    
    factor = ((h1 + h2)**2 - r**2) / (r**2 + (h1 + h2)**2)
    
    return (1j * omega * mu / (2 * np.pi)) * (
        k0_direct + factor * (k2_image - correction)
    )


def yci(omega, rcond, rins, epsilon_rins, epsilon0=EPSILON0):
    """
    Capacitive admittance of insulation layer.
    
    Parameters:
    -----------
    omega : float
        Angular frequency (rad/s)
    rcond : float
        Conductor radius (m)
    rins : float
        Insulation outer radius (m)
    epsilon_rins : float
        Relative permittivity of insulation
    epsilon0 : float
        Permittivity of free space (default EPSILON0)
    
    Returns:
    --------
    complex : Capacitive admittance (S/m)
    """
    return (1j * 2 * np.pi * omega * epsilon_rins * epsilon0) / np.log(rins / rcond)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def kron_reduction(matrix, nc, npr):
    """
    Kron reduction to eliminate ground wires from impedance matrix.
    
    This reduces the system size by eliminating ground wires while
    preserving their effects on phase conductors.
    
    IMPORTANT: This function is only used for IMPEDANCE matrices.
    For ADMITTANCE/CAPACITANCE, use a different approach: invert the 
    full potential coefficient matrix first, then extract the phase-only
    submatrix (do NOT use Kron reduction).
    
    Parameters:
    -----------
    matrix : ndarray
        Full impedance matrix (or potential coefficient for impedance calculation)
    nc : int
        Total number of conductors (phases + ground wires)
    npr : int
        Number of ground wires
    
    Returns:
    --------
    ndarray : Reduced matrix (phase conductors only)
    """
    n_phase = nc - npr
    
    if npr == 0:
        return matrix
    
    # Partition matrix into phase and ground wire blocks
    Zpp = matrix[:n_phase, :n_phase]  # Phase-phase
    Zpg = matrix[:n_phase, n_phase:]  # Phase-ground
    Zgp = matrix[n_phase:, :n_phase]  # Ground-phase
    Zgg = matrix[n_phase:, n_phase:]  # Ground-ground
    
    # Kron reduction formula
    Zgg_inv = np.linalg.inv(Zgg)
    Z_reduced = Zpp - Zpg @ Zgg_inv @ Zgp
    
    return Z_reduced


def bundle_reduction(matrix, nb, nf):
    """
    Reduce bundled conductors to equivalent single phase conductors.
    
    For bundled conductor configurations, this combines all sub-conductors
    in each bundle into a single equivalent conductor.
    
    Parameters:
    -----------
    matrix : ndarray
        Admittance matrix including all bundle sub-conductors
    nb : int
        Number of sub-conductors per bundle
    nf : int
        Number of phases (bundles)
    
    Returns:
    --------
    ndarray : Reduced matrix with one conductor per phase
    """
    result = np.zeros((nf, nf), dtype=complex)
    
    for m in range(nf):
        for n in range(nf):
            # Sum over all sub-conductors in bundles m and n
            i_start = m * nb
            i_end = (m + 1) * nb
            j_start = n * nb
            j_end = (n + 1) * nb
            
            result[m, n] = np.sum(matrix[i_start:i_end, j_start:j_end])
    
    return result


def compute_external_impedance_vectorized(omega, x, y, sigma_s, rf, rpr, npr):
    """
    Vectorized computation of external impedance matrix (Carson's equations).
    
    Uses NumPy broadcasting for efficient calculation of all matrix elements.
    More efficient than nested loops for moderate to large systems.
    
    Parameters:
    -----------
    omega : float
        Angular frequency (rad/s)
    x, y : ndarray
        Horizontal and vertical positions of conductors (m)
    sigma_s : float
        Ground conductivity (S/m)
    rf : float
        Phase conductor radius (m)
    rpr : float
        Ground wire radius (m)
    npr : int
        Number of ground wires
    
    Returns:
    --------
    ndarray : External impedance matrix (Ohm/m)
    """
    nc = len(x)
    p = np.sqrt(1 / (1j * omega * MU0 * sigma_s))
    
    # Create meshgrid for vectorized computation
    x_i, x_j = np.meshgrid(x, x, indexing='ij')
    y_i, y_j = np.meshgrid(y, y, indexing='ij')
    
    # Calculate distances
    dx = x_i - x_j
    dy = y_i - y_j
    sum_y = y_i + y_j
    
    # Off-diagonal elements (mutual impedances)
    term1 = dx**2 + (2*p + sum_y)**2
    term2 = dx**2 + dy**2
    
    # Avoid division by zero for diagonal
    with np.errstate(divide='ignore', invalid='ignore'):
        ze = 0.5 * np.log(term1 / term2)
        ze = np.where(np.isfinite(ze), ze, 0)
    
    # Diagonal elements (self impedances)
    for i in range(nc):
        if i < nc - npr:
            ze[i, i] = np.log(2 * (y[i] + p) / rf)
        else:
            ze[i, i] = np.log(2 * (y[i] + p) / rpr)
    
    ze *= 1j * omega * MU0 / (2 * np.pi)
    
    return ze


def compute_potential_coefficients_vectorized(x, y, rf, rpr, npr):
    """
    Vectorized computation of potential coefficient matrix (Maxwell's coefficients).
    
    Used for calculating shunt capacitance matrix.
    
    Parameters:
    -----------
    x, y : ndarray
        Horizontal and vertical positions of conductors (m)
    rf : float
        Phase conductor radius (m)
    rpr : float
        Ground wire radius (m)
    npr : int
        Number of ground wires
    
    Returns:
    --------
    ndarray : Potential coefficient matrix (dimensionless)
    """
    nc = len(x)
    
    # Create meshgrid
    x_i, x_j = np.meshgrid(x, x, indexing='ij')
    y_i, y_j = np.meshgrid(y, y, indexing='ij')
    
    dx = x_i - x_j
    sum_y = y_i + y_j
    diff_y = y_i - y_j
    
    # Off-diagonal elements (mutual potential coefficients)
    with np.errstate(divide='ignore', invalid='ignore'):
        mp = 0.5 * np.log((dx**2 + sum_y**2) / (dx**2 + diff_y**2))
        mp = np.where(np.isfinite(mp), mp, 0)
    
    # Diagonal elements (self potential coefficients)
    for i in range(nc):
        if i < nc - npr:
            mp[i, i] = np.log(2 * y[i] / rf)
        else:
            mp[i, i] = np.log(2 * y[i] / rpr)
    
    return mp


# ============================================================================
# MAIN Z AND Y MATRIX CALCULATION FUNCTIONS
# ============================================================================

def czyl_overhead_bundled(omega, x, y, sigma_s, rdc, rf, rint, 
                          npr, rdcpr, rpr, nb):
    """
    Calculate Z and Y matrices for overhead line with ground wires and bundled conductors.
    
    This is the most general case for overhead transmission lines.
    
    Parameters:
    -----------
    omega : float
        Angular frequency (rad/s)
    x, y : array-like
        Horizontal and vertical positions of all conductors including
        bundle sub-conductors (m)
    sigma_s : float
        Ground conductivity (S/m)
    rdc : float
        DC resistance per unit length of phase conductor (Ohm/m)
    rf : float
        Phase conductor radius (m)
    rint : float
        Phase conductor inner radius (m); use 0 for solid conductors
    npr : int
        Number of ground wires
    rdcpr : float
        DC resistance per unit length of ground wire (Ohm/m)
    rpr : float
        Ground wire radius (m)
    nb : int
        Number of sub-conductors per bundle
    
    Returns:
    --------
    tuple : (Z, Y) 
        Z : Series impedance matrix (Ohm/m)
        Y : Shunt admittance matrix (S/m)
    
    Example:
    --------
    >>> # 3-phase bundled line (2 conductors per bundle) with 2 ground wires
    >>> # Total: 3*2 + 2 = 8 conductors
    >>> omega = 2 * np.pi * 60
    >>> x = [0, 0.4, 5, 5.4, 10, 10.4, 2.5, 7.5]  # positions
    >>> y = [20, 20, 20, 20, 20, 20, 25, 25]
    >>> Z, Y = czyl_overhead_bundled(omega, x, y, 0.001, 0.1, 0.01, 0, 2, 0.5, 0.005, 2)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nc = len(x)
    nf = (nc - npr) // nb
    
    # Calculate conductor resistivities from DC resistance
    rhoc = rdc * np.pi * (rf**2 - rint**2)
    rhopr = rdcpr * np.pi * rpr**2
    
    # Internal impedance diagonal matrix
    z_internal = np.zeros(nc, dtype=complex)
    z_internal[:nc-npr] = zint_tubo(omega, rhoc, rf, rint, 1.0)
    z_internal[nc-npr:] = zin(omega, rhopr, rpr, 90.0)
    
    zin_matrix = np.diag(z_internal)
    
    # External impedance (Carson's ground return)
    ze = compute_external_impedance_vectorized(omega, x, y, sigma_s, rf, rpr, npr)
    
    # Total impedance
    z_total = zin_matrix + ze
    
    # Apply Kron reduction to eliminate ground wires
    z_reduced = kron_reduction(z_total, nc, npr)
    
    # Apply bundle reduction
    z_reduced_bundle = bundle_reduction(np.linalg.inv(z_reduced), nb, nf)
    Z = np.linalg.inv(z_reduced_bundle)
    
    # Potential coefficient matrix for capacitance calculation
    mp = compute_potential_coefficients_vectorized(x, y, rf, rpr, npr)
    
    # For admittance: invert full matrix first, then extract phase-only part
    # This is different from Kron reduction!
    C_full = np.linalg.inv(mp)
    C_phase = C_full[:nc-npr, :nc-npr]
    
    # Apply bundle reduction to capacitance matrix
    C_reduced = bundle_reduction(C_phase, nb, nf)
    
    # Admittance matrix (small conductance added for numerical stability)
    Y = 3.0e-12 * np.eye(nf) + 1j * omega * 2 * np.pi * EPSILON0 * C_reduced
    
    return Z, Y


def czyl_overhead(omega, x, y, sigma_s, rdc, rf, rint, npr, rdcpr, rpr):
    """
    Calculate Z and Y matrices for overhead line with ground wires (unbundled conductors).
    
    Parameters:
    -----------
    omega : float
        Angular frequency (rad/s)
    x, y : array-like
        Horizontal and vertical positions of conductors (m)
    sigma_s : float
        Ground conductivity (S/m)
    rdc : float
        DC resistance per unit length of phase conductor (Ohm/m)
    rf : float
        Phase conductor radius (m)
    rint : float
        Phase conductor inner radius (m); use 0 for solid conductors
    npr : int
        Number of ground wires
    rdcpr : float
        DC resistance per unit length of ground wire (Ohm/m)
    rpr : float
        Ground wire radius (m)
    
    Returns:
    --------
    tuple : (Z, Y) 
        Z : Series impedance matrix (Ohm/m)
        Y : Shunt admittance matrix (S/m)
    
    Example:
    --------
    >>> # 3-phase line with 2 ground wires
    >>> omega = 2 * np.pi * 60
    >>> x = [0, 5, 10, 2.5, 7.5]  # 3 phases + 2 ground wires
    >>> y = [20, 20, 20, 25, 25]
    >>> Z, Y = czyl_overhead(omega, x, y, 0.001, 0.1, 0.01, 0, 2, 0.5, 0.005)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nc = len(x)
    nf = nc - npr
    
    # Calculate conductor resistivities
    rhoc = rdc * np.pi * (rf**2 - rint**2)
    rhopr = rdcpr * np.pi * rpr**2
    
    # Internal impedance
    z_internal = np.zeros(nc, dtype=complex)
    if rint != 0:
        z_internal[:nc-npr] = zint_tubo(omega, rhoc, rf, rint, 1.0)
        z_internal[nc-npr:] = zin(omega, rhopr, rpr, 1.0)
    else:
        z_internal[:nc-npr] = zin(omega, rhoc, rf, 1.0)
        z_internal[nc-npr:] = zin(omega, rhopr, rpr, 1.0)
    
    zin_matrix = np.diag(z_internal)
    
    # External impedance
    ze = compute_external_impedance_vectorized(omega, x, y, sigma_s, rf, rpr, npr)
    
    # Total impedance with Kron reduction
    z_total = zin_matrix + ze
    Z_inv = kron_reduction(z_total, nc, npr)
    Z = np.linalg.inv(Z_inv)
    
    # Potential coefficient matrix
    mp = compute_potential_coefficients_vectorized(x, y, rf, rpr, npr)
    
    # For admittance: invert full matrix first, then extract phase-only part
    # This is NOT the same as Kron reduction
    C_full = np.linalg.inv(mp)
    C_phase = C_full[:nc-npr, :nc-npr]
    
    Y = 3.0e-12 * np.eye(nf) + 1j * omega * 2 * np.pi * EPSILON0 * C_phase
    
    return Z, Y


def czyl_simple(omega, x, y, sigma_s, rdc, rf, rint):
    """
    Calculate Z and Y matrices for simple overhead line (no ground wires or bundles).
    
    This is the simplest and most common case.
    
    Parameters:
    -----------
    omega : float
        Angular frequency (rad/s)
    x, y : array-like
        Horizontal and vertical positions of phase conductors (m)
    sigma_s : float
        Ground conductivity (S/m)
    rdc : float
        DC resistance per unit length (Ohm/m)
    rf : float
        Conductor radius (m)
    rint : float
        Conductor inner radius (m); use 0 for solid conductors
    
    Returns:
    --------
    tuple : (Z, Y) 
        Z : Series impedance matrix (Ohm/m)
        Y : Shunt admittance matrix (S/m)
    
    Example:
    --------
    >>> # 3-phase transmission line at 60 Hz
    >>> freq = 60.0
    >>> omega = 2 * np.pi * freq
    >>> x = [0.0, 2.5, 5.0]  # Horizontal spacing
    >>> y = [10.0, 10.0, 10.0]  # Height above ground
    >>> Z, Y = czyl_simple(omega, x, y, 0.001, 0.1, 0.01, 0.0)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nc = len(x)
    
    rhoc = rdc * np.pi * (rf**2 - rint**2)
    
    # Internal impedance
    if rint != 0:
        z_internal = zint_tubo(omega, rhoc, rf, rint, 1.0)
    else:
        z_internal = zin(omega, rhoc, rf, 1.0)
    
    zin_matrix = z_internal * np.eye(nc, dtype=complex)
    
    # External impedance
    ze = compute_external_impedance_vectorized(omega, x, y, sigma_s, rf, 0, 0)
    
    Z = zin_matrix + ze
    
    # Potential coefficient matrix
    mp = compute_potential_coefficients_vectorized(x, y, rf, 0, 0)
    
    # Admittance matrix
    Y = 3.0e-12 * np.eye(nc) + 1j * omega * 2 * np.pi * EPSILON0 * np.linalg.inv(mp)
    
    return Z, Y


def czysc_shielded_cable(omega, x, h, r, sigma_s, rho_cond, epsilon_ins1, 
                         rho_blind, epsilon_ins2):
    """
    Calculate Z and Y matrices for shielded underground cable system.
    
    Each cable has a conductor, insulation, metallic shield, and outer jacket.
    The model includes both conductor and shield currents.
    
    Parameters:
    -----------
    omega : float
        Angular frequency (rad/s)
    x : array-like
        Horizontal positions of cables (m)
    h : array-like
        Burial depths of cables (m)
    r : array-like
        Cable radii [conductor, ins1_outer, shield_inner, shield_outer] (m)
    sigma_s : float
        Ground conductivity (S/m)
    rho_cond : float
        Conductor resistivity (Ohm·m)
    epsilon_ins1 : float
        Inner insulation relative permittivity
    rho_blind : float
        Shield resistivity (Ohm·m)
    epsilon_ins2 : float
        Outer insulation relative permittivity
    
    Returns:
    --------
    tuple : (Z, Y) 
        Z : Series impedance matrix (Ohm/m) - size 2n×2n for n cables
        Y : Shunt admittance matrix (S/m) - size 2n×2n
        
    Note:
    -----
    The matrices include both conductor and shield for each cable.
    For n cables, matrix is 2n×2n with ordering:
    [cond1, shield1, cond2, shield2, ..., condn, shieldn]
    
    Example:
    --------
    >>> # 3 single-core shielded cables
    >>> omega = 2 * np.pi * 60
    >>> x = [0.0, 1.0, 2.0]  # 1m spacing
    >>> h = [1.5, 1.5, 1.5]  # 1.5m burial depth
    >>> r = [0.01, 0.025, 0.026, 0.03]  # Cable geometry
    >>> Z, Y = czysc_shielded_cable(omega, x, h, r, 0.001, 1.72e-8, 2.3, 2.5e-8, 2.3)
    """
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    ncables = len(x)
    
    # Calculate cable impedance components
    z1 = zin(omega, rho_cond, r[0], 1.0)
    z2_val = z2(omega, r[0], r[1], 1.0, MU0)
    z3_val = z3(omega, r[1], r[2], rho_blind, 1.0, MU0)
    z4_val = z4(omega, r[1], r[2], rho_blind, 1.0, MU0)
    z5_val = zint_tubo(omega, rho_blind, r[3], r[2])
    z6_val = z6(omega, r[2], r[3], 1.0, MU0)
    
    # Phase impedance block (2×2 for conductor and shield of one cable)
    zp = np.array([
        [z1 + z2_val + z3_val - 2*z4_val + z5_val + z6_val, z5_val + z6_val - z4_val],
        [z5_val + z6_val - z4_val, z5_val + z6_val]
    ], dtype=complex)
    
    # Build full impedance matrix (2n × 2n)
    k = 2 * ncables
    Z = np.zeros((k, k), dtype=complex)
    
    # Fill diagonal blocks with single cable impedance
    for i in range(ncables):
        Z[2*i:2*i+2, 2*i:2*i+2] = zp
    
    # Add ground impedance coupling between cables
    rext = r[3]
    
    # Calculate ground impedance matrix
    z_ground = np.zeros((ncables, ncables), dtype=complex)
    for i in range(ncables):
        for j in range(ncables):
            if i == j:
                z_ground[i, j] = z_solo(omega, rext, h[i], h[j], sigma_s, MU0)
            else:
                z_ground[i, j] = z_solo(omega, abs(x[i] - x[j]), h[i], h[j], sigma_s, MU0)
    
    # Expand ground impedance to full matrix (affects both conductor and shield)
    for i in range(ncables):
        for j in range(ncables):
            Z[2*i:2*i+2, 2*j:2*j+2] += z_ground[i, j]
    
    # Calculate admittance matrix
    y1 = yci(omega, r[0], r[1], epsilon_ins1, EPSILON0)
    y2 = yci(omega, r[2], r[3], epsilon_ins2, EPSILON0)
    
    # Admittance block for one cable (2×2)
    ycond = np.array([
        [y1, -y1],
        [-y1, y1 + y2]
    ], dtype=complex)
    
    # Build full admittance matrix
    Y = np.zeros((k, k), dtype=complex)
    for i in range(ncables):
        Y[2*i:2*i+2, 2*i:2*i+2] = ycond
    
    return Z, Y


# ============================================================================
# LINE NODAL ADMITTANCE
# ============================================================================

def yn_lt(Z, Y, length):
    """
    Calculate nodal admittance matrix from per-unit-length parameters.
    
    Uses exact transmission line equations based on modal decomposition.
    Suitable for power flow analysis and transient studies.
    
    Parameters:
    -----------
    Z : ndarray
        Series impedance matrix per unit length (Ohm/m)
    Y : ndarray
        Shunt admittance matrix per unit length (S/m)
    length : float
        Line length (m)
    
    Returns:
    --------
    tuple : (y11, y12)
        y11 : Self admittance matrix (S)
        y12 : Mutual admittance matrix (S)
        
    Note:
    -----
    The nodal admittance represents:
    [I1]   [y11  y12] [V1]
    [I2] = [y12  y11] [V2]
    
    where subscripts 1,2 denote the two ends of the line.
    
    Example:
    --------
    >>> # Calculate nodal admittance for 50 km line
    >>> Z_per_m, Y_per_m = czyl_simple(omega, x, y, sigma, rdc, rf, rint)
    >>> y11, y12 = yn_lt(Z_per_m, Y_per_m, 50000)  # 50 km
    """
    # Modal decomposition of propagation
    eigenvalues, eigenvectors = np.linalg.eig(Z @ Y)
    
    gamma = np.sqrt(eigenvalues)  # Propagation constants
    Tv = eigenvectors              # Mode transformation matrix
    Tvi = np.linalg.inv(eigenvectors)
    
    # Hyperbolic functions for transmission line equations
    hm = np.exp(-gamma * length)
    
    # Handle potential numerical issues for very long lines
    with np.errstate(divide='ignore', invalid='ignore'):
        coth_term = (1 + hm**2) / (1 - hm**2)
        csch_term = -2.0 * hm / (1 - hm**2)
        
        # Replace inf/nan with appropriate large/small values
        coth_term = np.where(np.isfinite(coth_term), coth_term, 1e15)
        csch_term = np.where(np.isfinite(csch_term), csch_term, 0)
    
    Am = gamma * coth_term  # Characteristic admittance * coth(gamma*length)
    Bm = gamma * csch_term  # Characteristic admittance * csch(gamma*length)
    
    # Transform back to phase domain
    Z_inv = np.linalg.inv(Z)
    y11 = Z_inv @ Tv @ np.diag(Am) @ Tvi
    y12 = Z_inv @ Tv @ np.diag(Bm) @ Tvi
    
    return y11, y12


# ============================================================================
# UTILITY FUNCTIONS FOR ANALYSIS
# ============================================================================

def frequency_sweep(freq_array, calc_func, *args):
    """
    Perform frequency sweep analysis efficiently.
    
    Calculates Z and Y matrices over a range of frequencies.
    Useful for frequency response analysis and harmonic studies.
    
    Parameters:
    -----------
    freq_array : ndarray
        Array of frequencies (Hz)
    calc_func : callable
        Function to calculate Z, Y (e.g., czyl_simple, czyl_overhead)
    *args : additional arguments for calc_func (except omega)
    
    Returns:
    --------
    dict : Dictionary containing:
        'frequencies' : Frequency array (Hz)
        'Z' : Impedance array (n_freq × n_phase × n_phase)
        'Y' : Admittance array (n_freq × n_phase × n_phase)
        
    Example:
    --------
    >>> freqs = np.logspace(0, 4, 100)  # 1 Hz to 10 kHz
    >>> results = frequency_sweep(freqs, czyl_simple, x, y, sigma, rdc, rf, rint)
    >>> # Plot self impedance magnitude
    >>> plt.loglog(results['frequencies'], np.abs(results['Z'][:, 0, 0]))
    """
    n_freq = len(freq_array)
    
    # Get matrix dimensions from first calculation
    omega_test = 2 * np.pi * freq_array[0]
    Z_test, Y_test = calc_func(omega_test, *args)
    n_phases = Z_test.shape[0]
    
    # Pre-allocate arrays
    Z_array = np.zeros((n_freq, n_phases, n_phases), dtype=complex)
    Y_array = np.zeros((n_freq, n_phases, n_phases), dtype=complex)
    
    # Calculate for all frequencies
    for i, freq in enumerate(freq_array):
        omega = 2 * np.pi * freq
        Z_array[i], Y_array[i] = calc_func(omega, *args)
    
    return {
        'frequencies': freq_array,
        'Z': Z_array,
        'Y': Y_array
    }


def extract_sequence_impedances(Z, Y=None):
    """
    Extract positive, negative, and zero sequence impedances from phase impedances.
    
    Useful for unbalanced fault analysis and protection studies.
    
    Parameters:
    -----------
    Z : ndarray
        Phase impedance matrix (3×3 for balanced 3-phase system)
    Y : ndarray, optional
        Phase admittance matrix (not used, for API compatibility)
    
    Returns:
    --------
    dict : Dictionary containing:
        'Z0' : Zero sequence impedance
        'Z1' : Positive sequence impedance
        'Z2' : Negative sequence impedance
        
    Note:
    -----
    For a perfectly symmetric and transposed line: Z1 = Z2
    
    Example:
    --------
    >>> Z, Y = czyl_simple(omega, x, y, sigma, rdc, rf, rint)
    >>> seq = extract_sequence_impedances(Z)
    >>> print(f"Zero sequence: {seq['Z0']}")
    >>> print(f"Positive sequence: {seq['Z1']}")
    """
    if Z.shape[0] != 3:
        raise ValueError("Sequence impedances are only defined for 3-phase systems")
    
    # Symmetrical components transformation matrix
    a = np.exp(2j * np.pi / 3)
    A = np.array([
        [1, 1, 1],
        [1, a**2, a],
        [1, a, a**2]
    ]) / 3
    
    A_inv = np.array([
        [1, 1, 1],
        [1, a, a**2],
        [1, a**2, a]
    ])
    
    # Transform to sequence domain
    Z_seq = A @ Z @ A_inv
    
    return {
        'Z0': Z_seq[0, 0],  # Zero sequence
        'Z1': Z_seq[1, 1],  # Positive sequence
        'Z2': Z_seq[2, 2]   # Negative sequence
    }


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Transmission Line and Cable Parameter Calculator")
    print("="*70)
    
    # Example 1: Simple 3-phase overhead line at 60 Hz
    print("\nExample 1: 3-Phase Overhead Line at 60 Hz")
    print("-" * 70)
    
    freq = 60.0  # Hz
    omega = 2 * np.pi * freq
    
    # Conductor positions (meters)
    x_coords = np.array([0.0, 2.5, 5.0])  # Horizontal spacing
    y_coords = np.array([10.0, 10.0, 10.0])  # Height above ground
    
    # Line parameters
    sigma_ground = 0.001  # Ground conductivity (S/m)
    rdc = 0.1  # DC resistance (Ohm/km) - will be converted
    conductor_radius = 0.01  # meters (1 cm)
    inner_radius = 0.0  # Solid conductor
    
    # Convert DC resistance to per meter
    rdc_per_m = rdc / 1000
    
    # Calculate Z and Y matrices
    Z, Y = czyl_simple(omega, x_coords, y_coords, sigma_ground, 
                       rdc_per_m, conductor_radius, inner_radius)
    
    print("\nSeries Impedance Matrix (Ohm/km):")
    print(Z * 1000)
    
    print("\nShunt Admittance Matrix (μS/km):")
    print(Y * 1000 * 1e6)  # Convert to microsiemens for readability
    
    print("\nSelf Impedance (diagonal):")
    print(f"  Real: {np.real(Z[0,0]) * 1000:.6f} Ohm/km")
    print(f"  Imag: {np.imag(Z[0,0]) * 1000:.6f} Ohm/km")
    
    print("\nMutual Impedance (off-diagonal):")
    print(f"  Real: {np.real(Z[0,1]) * 1000:.6f} Ohm/km")
    print(f"  Imag: {np.imag(Z[0,1]) * 1000:.6f} Ohm/km")
    
    print("\nSelf Admittance (diagonal):")
    print(f"  Imag: {np.imag(Y[0,0]) * 1000 * 1e6:.6f} μS/km")
    
    # Example 2: Nodal admittance for power flow
    print("\n" + "="*70)
    print("Example 2: Nodal Admittance for 50 km Line")
    print("-" * 70)
    
    line_length = 50000  # 50 km in meters
    y11, y12 = yn_lt(Z, Y, line_length)
    
    print("\nNodal Admittance Y11 (self) - magnitude:")
    print(f"  Diagonal: {np.abs(y11[0,0]):.6f} S")
    
    print("\nNodal Admittance Y12 (mutual) - magnitude:")
    print(f"  Diagonal: {np.abs(y12[0,0]):.6f} S")
    
    # Example 3: Sequence impedances
    print("\n" + "="*70)
    print("Example 3: Sequence Impedances")
    print("-" * 70)
    
    seq = extract_sequence_impedances(Z)
    
    print("\nZero Sequence Impedance (Ohm/km):")
    print(f"  {seq['Z0'] * 1000:.6f}")
    
    print("\nPositive Sequence Impedance (Ohm/km):")
    print(f"  {seq['Z1'] * 1000:.6f}")
    
    print("\nNegative Sequence Impedance (Ohm/km):")
    print(f"  {seq['Z2'] * 1000:.6f}")
    
    # Example 4: Frequency sweep
    print("\n" + "="*70)
    print("Example 4: Frequency Sweep (1 Hz to 1 kHz)")
    print("-" * 70)
    
    freqs = np.logspace(0, 3, 10)  # 1 Hz to 1 kHz, 10 points
    results = frequency_sweep(freqs, czyl_simple, 
                             x_coords, y_coords, sigma_ground, 
                             rdc_per_m, conductor_radius, inner_radius)
    
    print(f"\nCalculated parameters for {len(freqs)} frequencies")
    print(f"\nSelf Impedance at 1 Hz: {results['Z'][0, 0, 0] * 1000:.6f} Ohm/km")
    print(f"Self Impedance at 1 kHz: {results['Z'][-1, 0, 0] * 1000:.6f} Ohm/km")
    print(f"\nSelf Admittance at 60 Hz: {np.imag(results['Y'][3, 0, 0]) * 1000 * 1e6:.6f} μS/km")
    
    # Example 5: Verify admittance calculation
    print("\n" + "="*70)
    print("Example 5: Admittance Calculation Verification")
    print("-" * 70)
    
    # For a typical 230 kV line, shunt admittance should be around 3-5 μS/km
    print(f"\nShunt susceptance (imaginary part of Y):")
    print(f"  Self: {np.imag(Y[0,0]) * 1000 * 1e6:.4f} μS/km")
    print(f"  Mutual: {np.imag(Y[0,1]) * 1000 * 1e6:.4f} μS/km")
    print(f"\nTypical range for overhead lines: 3-6 μS/km")
    
    print("\n" + "="*70)
    print("All examples completed successfully!")
    print("="*70)
