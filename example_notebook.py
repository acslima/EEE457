# Example Jupyter Notebook - Transmission Line Analysis
# Save this as: transmission_line_analysis.ipynb

# Cell 1: Setup and Imports
# ============================================================================
%load_ext autoreload
%autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import line_cable_params as lcp

# Display settings
np.set_printoptions(precision=4, suppress=True)
plt.rcParams['figure.figsize'] = (12, 8)
print("✓ Setup complete!")

# Cell 2: Example 1 - Simple 3-Phase Overhead Line
# ============================================================================
print("="*60)
print("EXAMPLE 1: Simple 3-Phase Overhead Line at 60 Hz")
print("="*60)

# Parameters
freq = 60.0  # Hz
omega = 2 * np.pi * freq

# Conductor positions (meters)
x_coords = np.array([0.0, 2.5, 5.0])  # Horizontal spacing
y_coords = np.array([10.0, 10.0, 10.0])  # Height above ground

# Line parameters
sigma_ground = 0.001  # Ground conductivity (S/m)
rdc = 0.1  # DC resistance (Ohm/km)
conductor_radius = 0.01  # meters (1 cm)
inner_radius = 0.0  # Solid conductor

# Calculate Z and Y matrices
Z, Y = lcp.czyl_simple(omega, x_coords, y_coords, sigma_ground, 
                       rdc, conductor_radius, inner_radius)

print("\nSeries Impedance Matrix (Ohm/km):")
print(Z * 1000)

print("\nShunt Admittance Matrix (S/km):")
print(Y * 1000)

print("\nSelf Impedance (diagonal):", np.diag(Z) * 1000, "Ohm/km")
print("Mutual Impedance (off-diagonal):", Z[0,1] * 1000, "Ohm/km")

# Cell 3: Example 2 - Overhead Line with Ground Wires
# ============================================================================
print("\n" + "="*60)
print("EXAMPLE 2: Overhead Line with Ground Wires")
print("="*60)

# 3 phase conductors + 2 ground wires
x_all = np.array([0.0, 2.5, 5.0, 1.25, 3.75])  # meters
y_all = np.array([10.0, 10.0, 10.0, 12.0, 12.0])  # meters

# Ground wire parameters
npr = 2  # Number of ground wires
rdcpr = 0.5  # Ground wire DC resistance (Ohm/km)
rpr = 0.005  # Ground wire radius (5 mm)

Z_gw, Y_gw = lcp.czyl_overhead(omega, x_all, y_all, sigma_ground, 
                                rdc, conductor_radius, inner_radius,
                                npr, rdcpr, rpr)

print("\nWith Ground Wires - Series Impedance (Ohm/km):")
print(Z_gw * 1000)

print("\nReduction in mutual impedance:")
print(f"Without GW: {abs(Z[0,1] * 1000):.4f} Ohm/km")
print(f"With GW: {abs(Z_gw[0,1] * 1000):.4f} Ohm/km")
print(f"Reduction: {(1 - abs(Z_gw[0,1])/abs(Z[0,1]))*100:.1f}%")

# Cell 4: Example 3 - Frequency Sweep Analysis
# ============================================================================
print("\n" + "="*60)
print("EXAMPLE 3: Frequency Sweep Analysis")
print("="*60)

# Frequency range: 1 Hz to 10 kHz
freqs = np.logspace(0, 4, 100)
n_phases = 3

# Pre-allocate arrays
Z_self = np.zeros(len(freqs))
Z_mutual = np.zeros(len(freqs))
Y_self = np.zeros(len(freqs))

# Calculate for each frequency
for i, f in enumerate(freqs):
    omega_i = 2 * np.pi * f
    Z_i, Y_i = lcp.czyl_simple(omega_i, x_coords, y_coords, 
                               sigma_ground, rdc, conductor_radius, inner_radius)
    Z_self[i] = np.abs(Z_i[0, 0]) * 1000  # Convert to per km
    Z_mutual[i] = np.abs(Z_i[0, 1]) * 1000
    Y_self[i] = np.abs(Y_i[0, 0]) * 1000

print(f"\nCalculated impedance for {len(freqs)} frequency points")
print(f"Frequency range: {freqs[0]:.1f} Hz to {freqs[-1]:.1f} Hz")

# Cell 5: Plotting Frequency Response
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Impedance magnitude vs frequency
axes[0, 0].loglog(freqs, Z_self, 'b-', linewidth=2, label='Self Impedance')
axes[0, 0].loglog(freqs, Z_mutual, 'r--', linewidth=2, label='Mutual Impedance')
axes[0, 0].set_xlabel('Frequency (Hz)', fontsize=12)
axes[0, 0].set_ylabel('Impedance Magnitude (Ω/km)', fontsize=12)
axes[0, 0].set_title('Series Impedance vs Frequency', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, which='both', alpha=0.3)
axes[0, 0].legend(fontsize=11)

# Plot 2: Impedance phase vs frequency
Z_phase = np.zeros(len(freqs))
for i, f in enumerate(freqs):
    omega_i = 2 * np.pi * f
    Z_i, _ = lcp.czyl_simple(omega_i, x_coords, y_coords, 
                            sigma_ground, rdc, conductor_radius, inner_radius)
    Z_phase[i] = np.angle(Z_i[0, 0], deg=True)

axes[0, 1].semilogx(freqs, Z_phase, 'g-', linewidth=2)
axes[0, 1].set_xlabel('Frequency (Hz)', fontsize=12)
axes[0, 1].set_ylabel('Phase Angle (degrees)', fontsize=12)
axes[0, 1].set_title('Impedance Phase Angle', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Admittance magnitude vs frequency
axes[1, 0].loglog(freqs, Y_self, 'm-', linewidth=2)
axes[1, 0].set_xlabel('Frequency (Hz)', fontsize=12)
axes[1, 0].set_ylabel('Admittance Magnitude (S/km)', fontsize=12)
axes[1, 0].set_title('Shunt Admittance vs Frequency', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, which='both', alpha=0.3)

# Plot 4: Resistance and Reactance components
R_series = np.zeros(len(freqs))
X_series = np.zeros(len(freqs))
for i, f in enumerate(freqs):
    omega_i = 2 * np.pi * f
    Z_i, _ = lcp.czyl_simple(omega_i, x_coords, y_coords, 
                            sigma_ground, rdc, conductor_radius, inner_radius)
    R_series[i] = np.real(Z_i[0, 0]) * 1000
    X_series[i] = np.imag(Z_i[0, 0]) * 1000

axes[1, 1].loglog(freqs, R_series, 'b-', linewidth=2, label='Resistance')
axes[1, 1].loglog(freqs, X_series, 'r--', linewidth=2, label='Reactance')
axes[1, 1].set_xlabel('Frequency (Hz)', fontsize=12)
axes[1, 1].set_ylabel('Impedance Component (Ω/km)', fontsize=12)
axes[1, 1].set_title('R and X Components', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, which='both', alpha=0.3)
axes[1, 1].legend(fontsize=11)

plt.tight_layout()
plt.show()

print("✓ Frequency sweep plots generated")

# Cell 6: Example 4 - Nodal Admittance for Power Flow
# ============================================================================
print("\n" + "="*60)
print("EXAMPLE 4: Nodal Admittance Matrix")
print("="*60)

# Line length
line_length = 50.0  # km

# Get per-unit-length parameters at 60 Hz
omega = 2 * np.pi * 60.0
Z_pu, Y_pu = lcp.czyl_simple(omega, x_coords, y_coords, sigma_ground, 
                             rdc, conductor_radius, inner_radius)

# Convert to per km
Z_per_km = Z_pu / 1000
Y_per_km = Y_pu / 1000

# Calculate nodal admittance
y11, y12 = lcp.yn_lt(Z_per_km, Y_per_km, line_length)

print(f"\nLine length: {line_length} km")
print(f"\nNodal Admittance Y11 (self):")
print(y11)
print(f"\nNodal Admittance Y12 (mutual):")
print(y12)

# Verify reciprocity
print(f"\nReciprocity check (Y12 should equal Y21):")
print(f"Max difference: {np.max(np.abs(y12 - y12.T)):.2e}")

# Cell 7: Example 5 - Shielded Underground Cable
# ============================================================================
print("\n" + "="*60)
print("EXAMPLE 5: Shielded Underground Cable")
print("="*60)

# Cable positions
x_cable = np.array([0.0, 1.0, 2.0])  # Horizontal spacing (m)
h_cable = np.array([1.5, 1.5, 1.5])  # Burial depth (m)

# Cable geometry (meters)
# [conductor radius, inner insulation outer radius, 
#  shield inner radius, shield outer radius]
r_cable = np.array([0.01, 0.025, 0.026, 0.03])

# Material properties
rho_conductor = 1.72e-8  # Copper resistivity (Ohm·m)
rho_shield = 2.5e-8      # Aluminum shield (Ohm·m)
epsilon_ins1 = 2.3       # XLPE inner insulation
epsilon_ins2 = 2.3       # XLPE outer insulation

Z_cable, Y_cable = lcp.czysc_shielded_cable(
    omega, x_cable, h_cable, r_cable, sigma_ground,
    rho_conductor, epsilon_ins1, rho_shield, epsilon_ins2
)

print(f"\nCable System: 3 single-core shielded cables")
print(f"Burial depth: {h_cable[0]} m")
print(f"Cable spacing: {x_cable[1] - x_cable[0]} m")

print(f"\nImpedance Matrix Shape: {Z_cable.shape}")
print(f"(Includes both conductor and shield for each phase)")

# Extract phase conductor impedances (every other row/column)
Z_phase_only = Z_cable[::2, ::2]
print(f"\nPhase Conductor Impedance (Ohm/km):")
print(Z_phase_only * 1000)

print("\n" + "="*60)
print("Analysis Complete!")
print("="*60)
