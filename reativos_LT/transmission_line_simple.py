"""
Transmission Line Analysis Tool - Simplified Version
Based on Wolfram Mathematica code conversion
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TransmissionLine:
    """Transmission line parameters"""
    z1: complex  # Series impedance per unit length (Ω/km)
    y1: complex  # Shunt admittance per unit length (S/km)
    length: float  # Line length (km)
    vb: float  # Base voltage (kV)
    
    def __post_init__(self):
        """Calculate derived parameters"""
        self.gamma = np.sqrt(self.z1 * self.y1)
        self.Zc = np.sqrt(self.z1 / self.y1)
        self.Sb = abs(self.vb**2 / self.Zc)
        self.ib = self.Sb / (3 * self.vb)
        
        # ABCD parameters
        self.a = np.cosh(self.gamma * self.length)
        self.b = self.Zc * np.sinh(self.gamma * self.length)
        self.c = (1 / self.Zc) * np.sinh(self.gamma * self.length)
        self.d = self.a
        
        self.ABCD = np.array([[self.a, self.b], 
                              [self.c, self.d]])
    
    def analyze(self, vr: complex, load_factor: float) -> dict:
        """
        Analyze transmission line for a given load
        
        Parameters:
        -----------
        vr : complex
            Receiving end voltage (kV)
        load_factor : float
            Loading factor (1.0 = nominal, 0.5 = half load, 2.0 = double load)
            
        Returns:
        --------
        dict with all calculated parameters
        """
        # Receiving end current
        ir = vr / (load_factor * self.Zc)
        
        # Apply ABCD parameters
        receiving_vector = np.array([vr, ir])
        sending_vector = self.ABCD @ receiving_vector
        vs, is_val = sending_vector
        
        # Powers at receiving end
        Sr_total = 3 * vr * np.conj(ir)
        Pr = Sr_total.real
        Qr = Sr_total.imag
        
        # Powers at sending end
        Ss_total = 3 * vs * np.conj(is_val)
        Ps = Ss_total.real
        Qs = Ss_total.imag
        
        return {
            'load_factor': load_factor,
            'vr': vr, 'ir': ir,
            'vs': vs, 'is': is_val,
            'vr_pu': abs(vr) / self.vb,
            'ir_pu': abs(ir) / self.ib,
            'vs_pu': abs(vs) / self.vb,
            'is_pu': abs(is_val) / self.ib,
            'vr_ang': np.angle(vr, deg=True),
            'ir_ang': np.angle(ir, deg=True),
            'vs_ang': np.angle(vs, deg=True),
            'is_ang': np.angle(is_val, deg=True),
            'Sr': abs(Sr_total), 'Pr': Pr, 'Qr': Qr,
            'Ss': abs(Ss_total), 'Ps': Ps, 'Qs': Qs,
            'efficiency': (Pr / Ps * 100) if Ps != 0 else 0,
            'losses_p': Ps - Pr,
            'losses_q': Qs - Qr,
            'voltage_reg': ((abs(vs) - abs(vr)) / abs(vr)) * 100
        }
    
    def print_info(self):
        """Print transmission line parameters"""
        print("=" * 70)
        print("TRANSMISSION LINE PARAMETERS")
        print("=" * 70)
        print(f"Length: {self.length} km")
        print(f"Series Impedance z1: {abs(self.z1):.6f} ∠{np.angle(self.z1, deg=True):.2f}° Ω/km")
        print(f"Shunt Admittance y1: {abs(self.y1):.6e} ∠{np.angle(self.y1, deg=True):.2f}° S/km")
        print(f"Characteristic Impedance Zc: {abs(self.Zc):.2f} ∠{np.angle(self.Zc, deg=True):.2f}° Ω")
        print(f"Propagation Constant γ: {abs(self.gamma):.6f} ∠{np.angle(self.gamma, deg=True):.2f}° rad/km")
        print(f"\nBase Values:")
        print(f"  Voltage: {self.vb:.2f} kV")
        print(f"  Current: {self.ib:.4f} kA")
        print(f"  Power: {self.Sb:.2f} MVA")
        print("=" * 70)
        print()

def print_scenario_results(result: dict, scenario_name: str):
    """Print detailed results for a scenario"""
    print(f"\n{'=' * 70}")
    print(f"{scenario_name} (Load Factor = {result['load_factor']:.2f})")
    print(f"{'=' * 70}")
    
    print(f"\n{'RECEIVING END':^70}")
    print(f"  Voltage:  {result['vr_pu']:.4f} pu  ∠{result['vr_ang']:>7.2f}°  "
          f"({abs(result['vr']):>7.2f} kV)")
    print(f"  Current:  {result['ir_pu']:.4f} pu  ∠{result['ir_ang']:>7.2f}°  "
          f"({abs(result['ir']):>7.4f} kA)")
    print(f"  Apparent Power: {result['Sr']:>8.2f} MVA")
    print(f"  Active Power:   {result['Pr']:>8.2f} MW")
    print(f"  Reactive Power: {result['Qr']:>8.2f} MVAr")
    
    print(f"\n{'SENDING END':^70}")
    print(f"  Voltage:  {result['vs_pu']:.4f} pu  ∠{result['vs_ang']:>7.2f}°  "
          f"({abs(result['vs']):>7.2f} kV)")
    print(f"  Current:  {result['is_pu']:.4f} pu  ∠{result['is_ang']:>7.2f}°  "
          f"({abs(result['is']):>7.4f} kA)")
    print(f"  Apparent Power: {result['Ss']:>8.2f} MVA")
    print(f"  Active Power:   {result['Ps']:>8.2f} MW")
    print(f"  Reactive Power: {result['Qs']:>8.2f} MVAr")
    
    print(f"\n{'PERFORMANCE METRICS':^70}")
    print(f"  Efficiency:         {result['efficiency']:>6.2f} %")
    print(f"  Active Losses:      {result['losses_p']:>7.2f} MW")
    print(f"  Reactive Losses:    {result['losses_q']:>7.2f} MVAr")
    print(f"  Voltage Regulation: {result['voltage_reg']:>6.2f} %")

def create_comparison_plots(line: TransmissionLine, results: List[dict]):
    """Create comparison plots for multiple scenarios"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Transmission Line Analysis - {line.length} km', 
                 fontsize=16, fontweight='bold')
    
    scenarios = [r['load_factor'] for r in results]
    
    # Plot 1: Voltage comparison
    ax = axes[0, 0]
    ax.plot(scenarios, [r['vr_pu'] for r in results], 'o-', label='Receiving', linewidth=2)
    ax.plot(scenarios, [r['vs_pu'] for r in results], 's-', label='Sending', linewidth=2)
    ax.set_xlabel('Load Factor')
    ax.set_ylabel('Voltage (pu)')
    ax.set_title('Voltage Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Current comparison
    ax = axes[0, 1]
    ax.plot(scenarios, [r['ir_pu'] for r in results], 'o-', label='Receiving', linewidth=2)
    ax.plot(scenarios, [r['is_pu'] for r in results], 's-', label='Sending', linewidth=2)
    ax.set_xlabel('Load Factor')
    ax.set_ylabel('Current (pu)')
    ax.set_title('Current Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Active Power
    ax = axes[0, 2]
    ax.plot(scenarios, [r['Pr'] for r in results], 'o-', label='Receiving', linewidth=2)
    ax.plot(scenarios, [r['Ps'] for r in results], 's-', label='Sending', linewidth=2)
    ax.set_xlabel('Load Factor')
    ax.set_ylabel('Active Power (MW)')
    ax.set_title('Active Power Flow')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Reactive Power
    ax = axes[1, 0]
    ax.plot(scenarios, [r['Qr'] for r in results], 'o-', label='Receiving', linewidth=2)
    ax.plot(scenarios, [r['Qs'] for r in results], 's-', label='Sending', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Load Factor')
    ax.set_ylabel('Reactive Power (MVAr)')
    ax.set_title('Reactive Power Flow')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Efficiency
    ax = axes[1, 1]
    ax.plot(scenarios, [r['efficiency'] for r in results], 'o-', 
            color='green', linewidth=2)
    ax.set_xlabel('Load Factor')
    ax.set_ylabel('Efficiency (%)')
    ax.set_title('Transmission Efficiency')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([95, 101])
    
    # Plot 6: Losses
    ax = axes[1, 2]
    ax.plot(scenarios, [r['losses_p'] for r in results], 'o-', 
            label='Active', linewidth=2)
    ax.plot(scenarios, [r['losses_q'] for r in results], 's-', 
            label='Reactive', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Load Factor')
    ax.set_ylabel('Losses (MW/MVAr)')
    ax.set_title('Power Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    # Define transmission line (matching Mathematica code)
    line = TransmissionLine(
        z1=1j * 0.3179,  # Pure reactive (matching Mathematica)
        y1=1j * 5.1833e-6,
        length=200.0,
        vb=500.0 / np.sqrt(3)
    )
    
    # Print line parameters
    line.print_info()
    
    # Define scenarios (matching Mathematica code)
    scenarios = [
        ("CENÁRIO 1: CARREGAMENTO NOMINAL", 1.0, 1.0),
        ("CENÁRIO 2: CARGA ABAIXO DA NOMINAL", 1.0, 0.5),
        ("CENÁRIO 3: CARGA ACIMA DA NOMINAL", 1.0, 2.0)
    ]
    
    results = []
    for name, vr_factor, load_factor in scenarios:
        vr = vr_factor * line.vb
        result = line.analyze(vr, load_factor)
        results.append(result)
        print_scenario_results(result, name)
    
    # Create comparison plots
    fig = create_comparison_plots(line, results)
    plt.savefig('transmission_line_comparison.png', 
                dpi=300, bbox_inches='tight')
    print(f"\n{'=' * 70}")
    print("✓ Comparison plot saved to: transmission_line_comparison.png")
    print(f"{'=' * 70}\n")
    
    # Additional parametric study
    print("\n" + "=" * 70)
    print("PARAMETRIC STUDY - Load Ratio Sweep")
    print("=" * 70)
    
    load_factors = np.linspace(0.1, 3.0, 30)
    parametric_results = []
    
    for lf in load_factors:
        vr = 1.0 * line.vb
        result = line.analyze(vr, lf)
        parametric_results.append(result)
    
    # Create parametric plots
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Parametric Study - Load Factor Variation', 
                  fontsize=16, fontweight='bold')
    
    ax = axes[0, 0]
    ax.plot(load_factors, [r['vs_pu'] for r in parametric_results], linewidth=2)
    ax.set_xlabel('Load Ratio')
    ax.set_ylabel('Sending Voltage (pu)')
    ax.set_title('Sending Voltage vs Load')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Nominal')
    ax.legend()
    
    ax = axes[0, 1]
    ax.plot(load_factors, [r['Ps'] for r in parametric_results], linewidth=2)
    ax.set_xlabel('Load Ratio')
    ax.set_ylabel('Sending Active Power (MW)')
    ax.set_title('Active Power vs Load')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(load_factors, [r['Qs'] for r in parametric_results], linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Load Ratio')
    ax.set_ylabel('Sending Reactive Power (MVAr)')
    ax.set_title('Reactive Power vs Load')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(load_factors, [r['voltage_reg'] for r in parametric_results], linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Load Ratio')
    ax.set_ylabel('Voltage Regulation (%)')
    ax.set_title('Voltage Regulation vs Load')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parametric_study.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Parametric study plot saved to: parametric_study.png")
    print("=" * 70)
