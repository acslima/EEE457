"""
lcparam_improved.py - Enhanced Python port of LineCableParam2020.wl

Improved version with better performance, documentation, and error handling.
Features vectorized operations, type hints, and comprehensive validation.
"""

from __future__ import annotations

import numpy as np
from numpy import pi, sqrt, log, exp, diag, eye, kron
from numpy.linalg import inv, eig
from scipy.special import iv, kv
from typing import Iterable, Tuple, Optional, Union
import warnings

# Physical constants
MU0 = 4.0e-7 * pi
EPS0 = 8.854e-12

class TransmissionLineError(Exception):
    """Custom exception for transmission line parameter errors."""
    pass

def _validate_positive(value: float, name: str) -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")

def _validate_non_negative(value: float, name: str) -> None:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")

def zint_tubo(
    omega: complex, 
    rhoc: float, 
    rf: float, 
    rint: float, 
    mur: float = 1.0, 
    mu: float = MU0
) -> complex:
    """
    Internal impedance of tubular conductor.
    
    Parameters:
    -----------
    omega : complex
        Angular frequency [rad/s]
    rhoc : float
        Effective resistivity [Ω·m]
    rf : float
        Outer radius [m]
    rint : float
        Inner radius [m] (0 for solid conductor)
    mur : float
        Relative permeability
    mu : float
        Magnetic permeability [H/m]
    
    Returns:
    --------
    complex
        Internal impedance per unit length [Ω/m]
    """
    _validate_positive(rhoc, "rhoc")
    _validate_positive(rf, "rf")
    _validate_non_negative(rint, "rint")
    
    if rint >= rf:
        raise ValueError("Inner radius must be less than outer radius")
    
    eta_c = sqrt(1j * omega * mur * mu / rhoc)
    ri = max(rint, 1e-12)  # Avoid numerical issues
    
    # Handle very small arguments for Bessel functions
    if abs(eta_c * ri) < 1e-8 or abs(eta_c * rf) < 1e-8:
        warnings.warn("Small argument approximation used for Bessel functions")
        return (rhoc / (pi * (rf**2 - ri**2))) * (1 + 1j * omega * mur * mu * (rf**2 + ri**2) / (4 * rhoc))
    
    den = kv(1, eta_c * ri) * iv(1, eta_c * rf) - kv(1, eta_c * rf) * iv(1, eta_c * ri)
    num = kv(1, eta_c * ri) * iv(0, eta_c * rf) + kv(0, eta_c * rf) * iv(1, eta_c * ri)
    
    return (rhoc * eta_c) * num / (2 * pi * rf * den)

def zin(
    omega: complex, 
    rhopr: float, 
    rpr: float, 
    mur: float = 90.0, 
    mu: float = MU0
) -> complex:
    """
    Internal impedance of solid cylindrical conductor.
    
    Parameters:
    -----------
    omega : complex
        Angular frequency [rad/s]
    rhopr : float
        Effective resistivity [Ω·m]
    rpr : float
        Radius [m]
    mur : float
        Relative permeability
    mu : float
        Magnetic permeability [H/m]
    
    Returns:
    --------
    complex
        Internal impedance per unit length [Ω/m]
    """
    _validate_positive(rhopr, "rhopr")
    _validate_positive(rpr, "rpr")
    
    etapr = sqrt(1j * omega * mu * mur / rhopr)
    
    # Small argument approximation
    if abs(etapr * rpr) < 1e-8:
        return (rhopr / (pi * rpr**2)) * (1 + 1j * omega * mur * mu * rpr**2 / (4 * rhopr))
    
    return (etapr * rhopr) * iv(0, etapr * rpr) / (2 * pi * rpr * iv(1, etapr * rpr))

def _coaxial_impedance_element(
    omega: complex, 
    r_outer: float, 
    r_inner: float, 
    rho: Optional[float] = None,
    mur: float = 1.0, 
    mu: float = MU0
) -> complex:
    """Helper function for coaxial impedance elements."""
    if rho is None:
        # Dielectric impedance
        return 1j * omega * mur * mu / (2 * pi) * log(r_outer / r_inner)
    else:
        # Sheath impedance
        eta = sqrt(1j * omega * mur * mu / rho)
        den = iv(1, eta * r_outer) * kv(1, eta * r_inner) - iv(1, eta * r_inner) * kv(1, eta * r_outer)
        return (rho * eta) / (2 * pi * r_inner * den) * (
            iv(0, eta * r_inner) * kv(1, eta * r_outer) + kv(0, eta * r_inner) * iv(1, eta * r_outer)
        )

def z_solo(
    omega: complex, 
    r: float, 
    h1: float, 
    h2: float, 
    sigma_solo: complex, 
    mu: float = MU0
) -> complex:
    """
    Carson-type complex ground return impedance.
    
    Parameters:
    -----------
    omega : complex
        Angular frequency [rad/s]
    r : float
        Horizontal separation [m]
    h1 : float
        Height of conductor 1 [m]
    h2 : float
        Height of conductor 2 [m]
    sigma_solo : complex
        Soil conductivity [S/m]
    mu : float
        Magnetic permeability [H/m]
    
    Returns:
    --------
    complex
        Ground return impedance [Ω/m]
    """
    _validate_non_negative(r, "r")
    _validate_positive(h1, "h1")
    _validate_positive(h2, "h2")
    
    eta_solo = sqrt(1j * omega * mu * sigma_solo)
    a = h1 + h2
    b = h1 - h2
    r1 = sqrt(r**2 + b**2)
    r2 = sqrt(r**2 + a**2)
    
    term1 = kv(0, eta_solo * r1)
    term2 = kv(2, eta_solo * r2)
    term3 = 2.0 * exp(-a * eta_solo) * (1.0 + a * eta_solo) / (eta_solo**2 * r2**2)
    
    return 1j * omega * mu / (2 * pi) * (term1 + ((a**2 - r**2) / r2**2) * (term2 - term3))

def y_ci(
    omega: complex, 
    r_inner: float, 
    r_outer: float, 
    eps_r: float, 
    eps0_local: float = EPS0
) -> complex:
    """
    Shunt admittance between coaxial cylinders.
    
    Parameters:
    -----------
    omega : complex
        Angular frequency [rad/s]
    r_inner : float
        Inner radius [m]
    r_outer : float
        Outer radius [m]
    eps_r : float
        Relative permittivity
    eps0_local : float
        Vacuum permittivity [F/m]
    
    Returns:
    --------
    complex
        Shunt admittance per unit length [S/m]
    """
    _validate_positive(r_inner, "r_inner")
    _validate_positive(r_outer, "r_outer")
    _validate_positive(eps_r, "eps_r")
    
    if r_outer <= r_inner:
        raise ValueError("Outer radius must be greater than inner radius")
    
    return 1j * 2 * pi * omega * (eps_r * eps0_local) / log(r_outer / r_inner)

def kron_reduction(
    matrix: np.ndarray, 
    nc: int, 
    npr: int
) -> np.ndarray:
    """
    Kron reduction for impedance matrices.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Full matrix to reduce
    nc : int
        Total number of conductors
    npr : int
        Number of ground wires to eliminate
    
    Returns:
    --------
    np.ndarray
        Reduced matrix
    """
    if npr == 0:
        return matrix
    
    if nc <= npr:
        raise ValueError("Number of ground wires must be less than total conductors")
    
    y_full = inv(matrix)
    return y_full[:nc - npr, :nc - npr]

def bundle_reduction(
    matrix: np.ndarray, 
    nb: int, 
    nf: int
) -> np.ndarray:
    """
    Reduce bundled conductors to equivalent single conductors.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Full admittance matrix
    nb : int
        Number of subconductors per bundle
    nf : int
        Number of final equivalent conductors
    
    Returns:
    --------
    np.ndarray
        Reduced matrix
    """
    if matrix.shape[0] != nb * nf or matrix.shape[1] != nb * nf:
        raise ValueError("Matrix dimensions must match nb * nf")
    
    reduced = np.zeros((nf, nf), dtype=np.complex128)
    
    for m in range(nf):
        for n in range(nf):
            i0, i1 = nb * m, nb * (m + 1)
            j0, j1 = nb * n, nb * (n + 1)
            reduced[m, n] = np.sum(matrix[i0:i1, j0:j1])
    
    return reduced

def cZYlt(
    omega: complex,
    x: Iterable[float],
    y: Iterable[float],
    sigmas: complex,
    rdc: float,
    rf: float,
    rint: float,
    npr: int = 0,
    rdcpr: Optional[float] = None,
    rpr: Optional[float] = None,
    nb: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-unit-length (Z, Y) for overhead transmission lines.
    
    Parameters:
    -----------
    omega : complex
        Angular frequency [rad/s]
    x, y : Iterable[float]
        Conductor coordinates [m]
    sigmas : complex
        Soil conductivity [S/m]
    rdc : float
        DC resistance of phase conductor [Ω/m]
    rf : float
        Outer radius of phase conductor [m]
    rint : float
        Inner radius of phase conductor [m] (0 for solid)
    npr : int
        Number of ground wires
    rdcpr : float, optional
        DC resistance of ground wire [Ω/m]
    rpr : float, optional
        Radius of ground wire [m]
    nb : int
        Bundle size (subconductors per phase)
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Impedance and admittance matrices [Ω/m, S/m]
    """
    # Input validation
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y must have the same shape")
    
    nc = x_arr.size
    phases = nc - npr
    
    if phases <= 0:
        raise ValueError("Number of phases must be positive")
    
    if npr > 0 and (rdcpr is None or rpr is None):
        raise ValueError("rdcpr and rpr must be provided when npr > 0")
    
    # Calculate effective resistivities
    rhoc = rdc * pi * (rf**2 - max(rint, 0)**2)
    
    if npr > 0:
        rhopr = rdcpr * pi * rpr**2
    
    # Internal impedance matrix
    z_phase = zint_tubo(omega, rhoc, rf, rint) if rint > 0 else zin(omega, rhoc, rf)
    
    z_diag = [z_phase] * phases
    if npr > 0:
        z_gw = zin(omega, rhopr, rpr, mur=90.0)
        z_diag.extend([z_gw] * npr)
    
    Z_int = np.diag(z_diag).astype(np.complex128)
    
    # External impedance matrix (vectorized)
    p = sqrt(1.0 / (1j * omega * MU0 * sigmas))
    x_diff = x_arr[:, None] - x_arr[None, :]
    y_sum = y_arr[:, None] + y_arr[None, :]
    y_diff = y_arr[:, None] - y_arr[None, :]
    
    radii = np.full(nc, rf)
    if npr > 0:
        radii[phases:] = rpr
    
    # Off-diagonal elements
    off_diag_mask = ~np.eye(nc, dtype=bool)
    numerator = x_diff**2 + (2 * p + y_sum)**2
    denominator = x_diff**2 + y_diff**2
    Z_ext_off = 1j * omega * MU0 / (2 * pi) * 0.5 * log(numerator / denominator)
    
    # Diagonal elements
    Z_ext_diag = 1j * omega * MU0 / (2 * pi) * log(2.0 * (y_arr + p) / radii)
    
    Z_ext = np.where(off_diag_mask, Z_ext_off, np.diag(Z_ext_diag))
    
    # Full impedance matrix
    Z_full = Z_int + Z_ext
    
    # Reduction
    if npr > 0:
        Z_reduced = inv(kron_reduction(Z_full, nc, npr))
        if nb > 1:
            nf = phases // nb
            Z_reduced = inv(bundle_reduction(Z_reduced, nb, nf))
    else:
        Z_reduced = Z_full
    
    # Potential coefficient matrix
    num = x_diff**2 + y_sum**2
    den = x_diff**2 + y_diff**2
    P = 0.5 * log(num / den)
    np.fill_diagonal(P, log(2.0 * y_arr / radii))
    
    # Admittance matrix
    if npr > 0:
        P_reduced = kron_reduction(P, nc, npr)
        if nb > 1:
            nf = phases // nb
            P_reduced = bundle_reduction(P_reduced, nb, nf)
        Y_reduced = 1j * omega * 2 * pi * EPS0 * P_reduced
    else:
        Y_reduced = 1j * omega * 2 * pi * EPS0 * inv(P)
    
    # Add small conductance for numerical stability
    Y_reduced += 1e-12 * np.eye(Y_reduced.shape[0], dtype=np.complex128)
    
    return Z_reduced, Y_reduced

def ynLT(
    Z: np.ndarray, 
    Y: np.ndarray, 
    length: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nodal admittance matrices for a transmission line.
    
    Parameters:
    -----------
    Z : np.ndarray
        Series impedance matrix [Ω/m]
    Y : np.ndarray
        Shunt admittance matrix [S/m]
    length : float
        Line length [m]
    
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Y11 and Y12 nodal admittance matrices
    """
    _validate_positive(length, "length")
    
    Z_total = Z * length
    Y_total = Y * length
    
    # Eigenvalue decomposition
    gamma_squared = Z_total @ Y_total
    eigenvalues, eigenvectors = eig(gamma_squared)
    gamma = sqrt(eigenvalues)
    
    # Hyperbolic functions
    exp_neg_gamma_l = exp(-gamma * length)
    coth_gamma_l = (1 + exp_neg_gamma_l**2) / (1 - exp_neg_gamma_l**2)
    csch_gamma_l = 2 * exp_neg_gamma_l / (1 - exp_neg_gamma_l**2)
    
    # Transform to original basis
    T = eigenvectors
    T_inv = inv(T)
    
    Y11 = inv(Z_total) @ T @ diag(gamma * coth_gamma_l) @ T_inv
    Y12 = -inv(Z_total) @ T @ diag(gamma * csch_gamma_l) @ T_inv
    
    return Y11, Y12