import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Transmission line parameters
z1 = 1j * 0.3179  # Series impedance per unit length (Ω/km)
y1 = 1j * 5.1833e-6  # Shunt admittance per unit length (S/km)
L = 200.0  # Line length (km)

# Calculate propagation constant and characteristic impedance
gamma = np.sqrt(z1 * y1)
Zc = np.sqrt(z1 / y1)

# ABCD parameters for transmission line
a = np.cosh(gamma * L)
b = Zc * np.sinh(gamma * L)
c = (1 / Zc) * np.sinh(gamma * L)
d = a  # For symmetric transmission line

# ABCD matrix
ABCD = np.array([[a, b], [c, d]])

# Base values
vb = 500.0 / np.sqrt(3)  # Base voltage (kV)
Sb = abs(vb**2 / Zc)  # Base power (MVA) - take absolute value
ib = Sb / (3 * vb)  # Base current (kA)

print(f"=== Transmission Line Parameters ===")
print(f"Line length: {L} km")
print(f"Characteristic Impedance Zc: {abs(Zc):.2f} ∠{np.angle(Zc, deg=True):.2f}° Ω")
print(f"Propagation constant γ: {abs(gamma):.6f} ∠{np.angle(gamma, deg=True):.2f}° rad/km")
print(f"Base voltage: {vb:.2f} kV")
print(f"Base current: {ib:.4f} kA")
print(f"Base power: {Sb:.2f} MVA")
print()

# Function to analyze a scenario
def analyze_scenario(scenario_name, vr, ir_factor):
    """
    Analyze a transmission line scenario
    
    Parameters:
    - scenario_name: Name of the scenario
    - vr: Receiving end voltage (kV)
    - ir_factor: Factor to multiply Zc for current calculation
    """
    # Receiving end quantities
    ir = vr / (ir_factor * Zc)
    
    # Calculate sending end quantities using ABCD parameters
    receiving_vector = np.array([vr, ir])
    sending_vector = ABCD @ receiving_vector
    vs, is_val = sending_vector
    
    # Calculate powers at receiving end
    Sr_total = 3 * vr * np.conj(ir)  # Total three-phase apparent power
    Pr = Sr_total.real  # Active power
    Qr = Sr_total.imag  # Reactive power
    
    # Calculate powers at sending end
    Ss_total = 3 * vs * np.conj(is_val)  # Total three-phase apparent power
    Ps = Ss_total.real  # Active power
    Qs = Ss_total.imag  # Reactive power
    
    results = {
        'scenario': scenario_name,
        # Receiving end
        'vr_mag': float(abs(vr)),
        'vr_ang': float(np.angle(vr, deg=True)),
        'vr_pu': float(abs(vr) / vb),
        'ir_mag': float(abs(ir)),
        'ir_ang': float(np.angle(ir, deg=True)),
        'ir_pu': float(abs(ir) / ib),
        'Sr': float(abs(Sr_total)),
        'Pr': float(Pr),
        'Qr': float(Qr),
        # Sending end
        'vs_mag': float(abs(vs)),
        'vs_ang': float(np.angle(vs, deg=True)),
        'vs_pu': float(abs(vs) / vb),
        'is_mag': float(abs(is_val)),
        'is_ang': float(np.angle(is_val, deg=True)),
        'is_pu': float(abs(is_val) / ib),
        'Ss': float(abs(Ss_total)),
        'Ps': float(Ps),
        'Qs': float(Qs),
        # Efficiency and losses
        'efficiency': float((Pr / Ps * 100) if Ps != 0 else 0),
        'losses_p': float(Ps - Pr),
        'losses_q': float(Qs - Qr)
    }
    
    return results

# Analyze three scenarios
scenarios = []

print("=" * 70)
print("CENÁRIO 1: CARREGAMENTO NOMINAL (Loading Factor = 1.0)")
print("=" * 70)
vr1 = 1.0 * vb
result1 = analyze_scenario("Nominal Loading", vr1, 1.0)
scenarios.append(result1)

print(f"Receiving End:")
print(f"  Voltage Vr = {result1['vr_pu']:.4f} ∠{result1['vr_ang']:.2f}° pu")
print(f"  Current Ir = {result1['ir_pu']:.4f} ∠{result1['ir_ang']:.2f}° pu")
print(f"  Apparent Power = {result1['Sr']:.2f} MVA")
print(f"  Active Power = {result1['Pr']:.2f} MW")
print(f"  Reactive Power = {result1['Qr']:.2f} MVAr")
print(f"\nSending End:")
print(f"  Voltage Vs = {result1['vs_pu']:.4f} ∠{result1['vs_ang']:.2f}° pu")
print(f"  Current Is = {result1['is_pu']:.4f} ∠{result1['is_ang']:.2f}° pu")
print(f"  Apparent Power = {result1['Ss']:.2f} MVA")
print(f"  Active Power = {result1['Ps']:.2f} MW")
print(f"  Reactive Power = {result1['Qs']:.2f} MVAr")
print(f"\nEfficiency: {result1['efficiency']:.2f}%")
print(f"Losses: {result1['losses_p']:.2f} MW, {result1['losses_q']:.2f} MVAr")
print()

print("=" * 70)
print("CENÁRIO 2: CARGA ABAIXO DA NOMINAL (Loading Factor = 0.5)")
print("=" * 70)
vr2 = 1.0 * vb
result2 = analyze_scenario("Below Nominal (50%)", vr2, 2.0)
scenarios.append(result2)

print(f"Receiving End:")
print(f"  Voltage Vr = {result2['vr_pu']:.4f} ∠{result2['vr_ang']:.2f}° pu")
print(f"  Current Ir = {result2['ir_pu']:.4f} ∠{result2['ir_ang']:.2f}° pu")
print(f"  Apparent Power = {result2['Sr']:.2f} MVA")
print(f"  Active Power = {result2['Pr']:.2f} MW")
print(f"  Reactive Power = {result2['Qr']:.2f} MVAr")
print(f"\nSending End:")
print(f"  Voltage Vs = {result2['vs_pu']:.4f} ∠{result2['vs_ang']:.2f}° pu")
print(f"  Current Is = {result2['is_pu']:.4f} ∠{result2['is_ang']:.2f}° pu")
print(f"  Apparent Power = {result2['Ss']:.2f} MVA")
print(f"  Active Power = {result2['Ps']:.2f} MW")
print(f"  Reactive Power = {result2['Qs']:.2f} MVAr")
print(f"\nEfficiency: {result2['efficiency']:.2f}%")
print(f"Losses: {result2['losses_p']:.2f} MW, {result2['losses_q']:.2f} MVAr")
print()

print("=" * 70)
print("CENÁRIO 3: CARGA ACIMA DA NOMINAL (Loading Factor = 2.0)")
print("=" * 70)
vr3 = 1.0 * vb
result3 = analyze_scenario("Above Nominal (200%)", vr3, 0.5)
scenarios.append(result3)

print(f"Receiving End:")
print(f"  Voltage Vr = {result3['vr_pu']:.4f} ∠{result3['vr_ang']:.2f}° pu")
print(f"  Current Ir = {result3['ir_pu']:.4f} ∠{result3['ir_ang']:.2f}° pu")
print(f"  Apparent Power = {result3['Sr']:.2f} MVA")
print(f"  Active Power = {result3['Pr']:.2f} MW")
print(f"  Reactive Power = {result3['Qr']:.2f} MVAr")
print(f"\nSending End:")
print(f"  Voltage Vs = {result3['vs_pu']:.4f} ∠{result3['vs_ang']:.2f}° pu")
print(f"  Current Is = {result3['is_pu']:.4f} ∠{result3['is_ang']:.2f}° pu")
print(f"  Apparent Power = {result3['Ss']:.2f} MVA")
print(f"  Active Power = {result3['Ps']:.2f} MW")
print(f"  Reactive Power = {result3['Qs']:.2f} MVAr")
print(f"\nEfficiency: {result3['efficiency']:.2f}%")
print(f"Losses: {result3['losses_p']:.2f} MW, {result3['losses_q']:.2f} MVAr")
print()

# Create comprehensive dashboard
fig = make_subplots(
    rows=3, cols=3,
    subplot_titles=(
        'Voltage Comparison (pu)', 'Current Comparison (pu)', 'Voltage Angles (degrees)',
        'Active Power (MW)', 'Reactive Power (MVAr)', 'Apparent Power (MVA)',
        'Power Losses', 'Efficiency (%)', 'Voltage & Current Summary'
    ),
    specs=[[{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'bar'}, {'type': 'bar'}],
           [{'type': 'bar'}, {'type': 'bar'}, {'type': 'table'}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.10
)

scenarios_names = [s['scenario'] for s in scenarios]
colors = ['#636EFA', '#EF553B', '#00CC96']

# Row 1, Col 1: Voltage Comparison
for i, loc in enumerate(['Receiving', 'Sending']):
    key = 'vr_pu' if loc == 'Receiving' else 'vs_pu'
    values = [s[key] for s in scenarios]
    fig.add_trace(
        go.Bar(name=f'{loc} End', x=scenarios_names, y=values,
               marker_color=colors[i], showlegend=True),
        row=1, col=1
    )

# Row 1, Col 2: Current Comparison
for i, loc in enumerate(['Receiving', 'Sending']):
    key = 'ir_pu' if loc == 'Receiving' else 'is_pu'
    values = [s[key] for s in scenarios]
    fig.add_trace(
        go.Bar(name=f'{loc} End', x=scenarios_names, y=values,
               marker_color=colors[i], showlegend=False),
        row=1, col=2
    )

# Row 1, Col 3: Voltage Angles
for i, loc in enumerate(['Receiving', 'Sending']):
    key = 'vr_ang' if loc == 'Receiving' else 'vs_ang'
    values = [s[key] for s in scenarios]
    fig.add_trace(
        go.Bar(name=f'{loc} End', x=scenarios_names, y=values,
               marker_color=colors[i], showlegend=False),
        row=1, col=3
    )

# Row 2, Col 1: Active Power
for i, loc in enumerate(['Receiving', 'Sending']):
    key = 'Pr' if loc == 'Receiving' else 'Ps'
    values = [s[key] for s in scenarios]
    fig.add_trace(
        go.Bar(name=f'{loc} End', x=scenarios_names, y=values,
               marker_color=colors[i], showlegend=False),
        row=2, col=1
    )

# Row 2, Col 2: Reactive Power
for i, loc in enumerate(['Receiving', 'Sending']):
    key = 'Qr' if loc == 'Receiving' else 'Qs'
    values = [s[key] for s in scenarios]
    fig.add_trace(
        go.Bar(name=f'{loc} End', x=scenarios_names, y=values,
               marker_color=colors[i], showlegend=False),
        row=2, col=2
    )

# Row 2, Col 3: Apparent Power
for i, loc in enumerate(['Receiving', 'Sending']):
    key = 'Sr' if loc == 'Receiving' else 'Ss'
    values = [s[key] for s in scenarios]
    fig.add_trace(
        go.Bar(name=f'{loc} End', x=scenarios_names, y=values,
               marker_color=colors[i], showlegend=False),
        row=2, col=3
    )

# Row 3, Col 1: Power Losses
loss_p = [s['losses_p'] for s in scenarios]
loss_q = [s['losses_q'] for s in scenarios]
fig.add_trace(
    go.Bar(name='Active Losses', x=scenarios_names, y=loss_p,
           marker_color='#FF6692', showlegend=True),
    row=3, col=1
)
fig.add_trace(
    go.Bar(name='Reactive Losses', x=scenarios_names, y=loss_q,
           marker_color='#B6E880', showlegend=True),
    row=3, col=1
)

# Row 3, Col 2: Efficiency
efficiency = [s['efficiency'] for s in scenarios]
fig.add_trace(
    go.Bar(x=scenarios_names, y=efficiency,
           marker_color='#FECB52', showlegend=False,
           text=[f"{e:.2f}%" for e in efficiency],
           textposition='outside'),
    row=3, col=2
)

# Row 3, Col 3: Summary Table
table_data = []
headers = ['Parameter', 'Scenario 1', 'Scenario 2', 'Scenario 3']
table_data.append(['Vr (pu)', f"{scenarios[0]['vr_pu']:.4f}", 
                   f"{scenarios[1]['vr_pu']:.4f}", f"{scenarios[2]['vr_pu']:.4f}"])
table_data.append(['Vs (pu)', f"{scenarios[0]['vs_pu']:.4f}", 
                   f"{scenarios[1]['vs_pu']:.4f}", f"{scenarios[2]['vs_pu']:.4f}"])
table_data.append(['Ir (pu)', f"{scenarios[0]['ir_pu']:.4f}", 
                   f"{scenarios[1]['ir_pu']:.4f}", f"{scenarios[2]['ir_pu']:.4f}"])
table_data.append(['Is (pu)', f"{scenarios[0]['is_pu']:.4f}", 
                   f"{scenarios[1]['is_pu']:.4f}", f"{scenarios[2]['is_pu']:.4f}"])
table_data.append(['Pr (MW)', f"{scenarios[0]['Pr']:.2f}", 
                   f"{scenarios[1]['Pr']:.2f}", f"{scenarios[2]['Pr']:.2f}"])
table_data.append(['Ps (MW)', f"{scenarios[0]['Ps']:.2f}", 
                   f"{scenarios[1]['Ps']:.2f}", f"{scenarios[2]['Ps']:.2f}"])
table_data.append(['Eff (%)', f"{scenarios[0]['efficiency']:.2f}", 
                   f"{scenarios[1]['efficiency']:.2f}", f"{scenarios[2]['efficiency']:.2f}"])

fig.add_trace(
    go.Table(
        header=dict(values=headers,
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=11, color='black')),
        cells=dict(values=list(zip(*table_data)),
                   fill_color='lavender',
                   align='left',
                   font=dict(size=10))
    ),
    row=3, col=3
)

# Update layout
fig.update_layout(
    title_text=f"Transmission Line Analysis Dashboard - {L} km Line<br>" +
               f"<sub>Zc = {abs(Zc):.2f}∠{np.angle(Zc, deg=True):.2f}° Ω | " +
               f"Vbase = {vb:.2f} kV | Sbase = {Sb:.2f} MVA</sub>",
    title_font_size=16,
    height=1200,
    showlegend=True,
    legend=dict(x=0.02, y=0.98),
    barmode='group'
)

# Update y-axes labels
fig.update_yaxes(title_text="Voltage (pu)", row=1, col=1)
fig.update_yaxes(title_text="Current (pu)", row=1, col=2)
fig.update_yaxes(title_text="Angle (°)", row=1, col=3)
fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
fig.update_yaxes(title_text="Power (MVAr)", row=2, col=2)
fig.update_yaxes(title_text="Power (MVA)", row=2, col=3)
fig.update_yaxes(title_text="Losses (MW/MVAr)", row=3, col=1)
fig.update_yaxes(title_text="Efficiency (%)", row=3, col=2)

# Save the interactive dashboard
output_file = '/mnt/user-data/outputs/transmission_line_dashboard.html'
fig.write_html(output_file)
print(f"\n✓ Interactive dashboard saved to: {output_file}")

# Also create a detailed comparison table
df_comparison = pd.DataFrame(scenarios)
df_comparison = df_comparison.round(4)
csv_file = '/mnt/user-data/outputs/transmission_line_results.csv'
df_comparison.to_csv(csv_file, index=False)
print(f"✓ Results table saved to: {csv_file}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)
