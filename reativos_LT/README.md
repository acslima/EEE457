# Transmission Line Analysis - Mathematica to Python Conversion

## Overview
This project converts Wolfram Mathematica transmission line analysis code to Python, with interactive dashboards for comparing different loading scenarios.

## Files Generated

### 1. Python Scripts

#### `transmission_line_analysis.py`
**Full-featured script with interactive Plotly dashboard**
- Converts all Mathematica calculations to Python
- Generates interactive HTML dashboard with 9 subplots
- Exports results to CSV
- Uses: plotly, pandas, numpy

**Key Features:**
- Interactive 3x3 dashboard layout
- Voltage, current, and power comparisons
- Efficiency and loss analysis
- Summary table
- Hover-over data inspection

**Usage:**
```bash
python transmission_line_analysis.py
```

**Outputs:**
- `transmission_line_dashboard.html` - Interactive dashboard
- `transmission_line_results.csv` - Numerical results

---

#### `transmission_line_simple.py`
**Simplified, modular version with matplotlib plots**
- Object-oriented design using `TransmissionLine` class
- Easy to modify and extend
- Static matplotlib plots
- Parametric study functionality

**Key Features:**
- Clean class-based architecture
- Detailed console output
- Comparison plots for 3 scenarios
- Parametric load sweep (0.1 to 3.0 load factor)
- Easy to customize

**Usage:**
```bash
python transmission_line_simple.py
```

**Outputs:**
- `transmission_line_comparison.png` - 6-panel comparison
- `parametric_study.png` - Load factor sweep analysis

---

### 2. Output Files

#### `transmission_line_dashboard.html` (4.7 MB)
Interactive Plotly dashboard with:
- Voltage comparison (Receiving vs Sending)
- Current comparison
- Voltage angles
- Active power flow
- Reactive power flow
- Apparent power
- Power losses (Active & Reactive)
- Transmission efficiency
- Summary data table

**How to use:**
- Open in any web browser
- Hover over plots for detailed values
- Use Plotly controls to zoom, pan, save images

---

#### `transmission_line_comparison.png` (546 KB)
Static matplotlib figure with 6 subplots:
1. Voltage profile (pu)
2. Current profile (pu)
3. Active power flow (MW)
4. Reactive power flow (MVAr)
5. Transmission efficiency (%)
6. Power losses (MW/MVAr)

---

#### `parametric_study.png` (380 KB)
Load factor sweep analysis (0.1 to 3.0):
1. Sending voltage vs load
2. Active power vs load
3. Reactive power vs load
4. Voltage regulation vs load

---

#### `transmission_line_results.csv` (618 bytes)
Numerical results for all three scenarios in CSV format.
Columns include: scenario, voltages, currents, powers, angles, efficiency, losses.

---

## Transmission Line Parameters

### Physical Parameters
- **Series Impedance (z1):** 0.3179j Ω/km (pure reactive)
- **Shunt Admittance (y1):** 5.1833×10⁻⁶j S/km (pure susceptive)
- **Line Length:** 200 km

### Calculated Parameters
- **Characteristic Impedance (Zc):** 247.65∠0° Ω
- **Propagation Constant (γ):** 0.001284∠90° rad/km

### Base Values
- **Base Voltage:** 288.68 kV (500/√3 kV)
- **Base Current:** 0.3885 kA
- **Base Power:** 336.49 MVA

---

## Three Loading Scenarios

### Scenario 1: Nominal Loading (Load Factor = 1.0)
- **Receiving End:** Vr = 1.0 pu, Ir = 3.0 pu
- **Sending End:** Vs = 1.0 pu ∠14.71°, Is = 3.0 pu ∠14.71°
- **Power:** Sr = Ss = 1009.48 MVA
- **Efficiency:** 100.00%
- **Key Characteristic:** Perfectly matched load (resistive at Zc)

### Scenario 2: Below Nominal (Load Factor = 0.5)
- **Receiving End:** Vr = 1.0 pu, Ir = 1.5 pu
- **Sending End:** Vs = 0.9755 pu ∠7.48°, Is = 1.6387 pu ∠27.70°
- **Power:** Sr = 504.74 MVA, Ss = 537.90 MVA
- **Reactive:** Capacitive (line charging effect dominant)
- **Voltage Regulation:** -2.45% (Ferranti effect)

### Scenario 3: Above Nominal (Load Factor = 2.0)
- **Receiving End:** Vr = 1.0 pu, Ir = 6.0 pu
- **Sending End:** Vs = 1.0924 pu ∠27.70°, Is = 5.8531 pu ∠7.48°
- **Power:** Sr = 2018.96 MVA, Ss = 2151.61 MVA
- **Reactive:** Inductive (heavy loading)
- **Voltage Regulation:** 9.24%

---

## Key Observations

### 1. Reactive Power Behavior
- **Light Load (0.5):** Line generates reactive power (-185.94 MVAr)
- **Nominal Load (1.0):** Balanced (0.00 MVAr)
- **Heavy Load (2.0):** Line consumes reactive power (+743.78 MVAr)

### 2. Voltage Regulation
- Light loads exhibit Ferranti effect (voltage rise)
- Heavy loads show voltage drop
- Critical for planning reactive compensation

### 3. Efficiency
All scenarios show 100% active power efficiency because:
- Pure reactive impedance (z1 = 0 + j0.3179)
- No series resistance
- In practice, would be 98-99% due to losses

---

## Comparison with Mathematica Code

### Original Mathematica Approach
```mathematica
z1 = I 0.3179;
y1 = I 5.1833 10^-6;
γ = Sqrt[z1 y1];
Zc = Sqrt[z1/y1];
q = {{a, b}, {c, a}};  (* ABCD matrix *)
{vs, is} = q . {vr, ir};  (* Matrix multiplication *)
```

### Python Equivalent
```python
z1 = 1j * 0.3179
y1 = 1j * 5.1833e-6
gamma = np.sqrt(z1 * y1)
Zc = np.sqrt(z1 / y1)
ABCD = np.array([[a, b], [c, d]])
vs, is_val = ABCD @ np.array([vr, ir])
```

### Advantages of Python Version
1. **Reproducibility:** Scripts can be version controlled
2. **Automation:** Easy to run parametric studies
3. **Visualization:** Interactive and static plots
4. **Data Export:** CSV for further analysis
5. **Integration:** Can be integrated into larger workflows
6. **Documentation:** Self-documenting code with docstrings

---

## How to Modify

### Change Line Parameters
Edit in either script:
```python
line = TransmissionLine(
    z1=0.0175 + 1j * 0.3179,  # Add resistance if needed
    y1=1j * 5.1833e-6,
    length=150.0,  # Change length
    vb=345.0 / np.sqrt(3)  # Different voltage level
)
```

### Add More Scenarios
```python
scenarios = [
    ("Custom Scenario", 1.1, 1.5),  # (name, vr_factor, load_factor)
]
```

### Perform Parametric Studies
```python
# Vary conductivity, permittivity, or any parameter
for conductivity in [0.001, 0.01, 0.1]:
    # Update parameters and rerun analysis
    pass
```

---

## Installation Requirements

### Required Packages
```bash
pip install numpy pandas plotly matplotlib
```

Or with conda:
```bash
conda install numpy pandas plotly matplotlib
```

### Tested Environment
- Python 3.8+
- NumPy 1.20+
- Plotly 5.0+
- Matplotlib 3.3+
- Pandas 1.2+

---

## Mathematical Background

### ABCD Parameters for Transmission Line
```
[Vs]   [A  B] [Vr]
[Is] = [C  D] [Ir]
```

Where:
- A = D = cosh(γℓ)
- B = Zc·sinh(γℓ)
- C = (1/Zc)·sinh(γℓ)
- γ = √(zy) = propagation constant
- Zc = √(z/y) = characteristic impedance

### Power Calculations
- **Three-phase Apparent Power:** S = 3·V·I*
- **Active Power:** P = Re(S)
- **Reactive Power:** Q = Im(S)
- **Efficiency:** η = Pr/Ps × 100%

---

## Future Enhancements

Potential additions to the analysis:
1. Add series resistance to z1 for realistic losses
2. Include temperature effects on resistance
3. Add corona losses
4. Transient analysis capabilities
5. Fault analysis
6. Optimal power flow calculations
7. Stability limits (surge impedance loading)
8. Frequency response analysis

---

## References

This code implements standard transmission line analysis using:
- ABCD parameters method
- Hyperbolic functions for distributed parameters
- Per-unit system for normalization

Standard references:
- Grainger & Stevenson: "Power System Analysis"
- Glover et al.: "Power System Analysis and Design"
- Kundur: "Power System Stability and Control"

---

## Contact & Support

For questions about this conversion or to report issues:
- Check the Python scripts for detailed comments
- Review the console output for detailed results
- Examine the interactive dashboard for visual insights

---

## License

This code is provided as-is for educational and research purposes.

---

**Generated:** November 26, 2025
**Python Version:** 3.12+
**Status:** ✓ Verified against Mathematica results
