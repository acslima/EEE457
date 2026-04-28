#!/usr/bin/env python3
"""
Seleção Multicritério de Condutores ACSR — LT 138 kV / 80 km
Critério: Maximização da área do polígono de 7 vértices (scores normalizados).

Geometria da torre:
  xc = [3.4, -3.4, 3.4, -3.4, 3.4] m  →  GMD = 7.727 m
  Fases A(+3.4, 21.0 m), B(-3.4, 17.2 m), C(+3.4, 13.4 m)

Os 7 critérios (vértices do polígono):
  ① Perdas resistivas trifásicas     [↓ melhor, score invertido]
  ② Ampacidade (capacidade térmica)  [↑ melhor]
  ③ Regulação de tensão ΔV%          [↓ melhor, score invertido]
  ④ Potência nominal — SIL = V²/Zc   [↑ melhor]
  ⑤ Potência máxima = V²/(XL·ℓ)     [↑ melhor]
  ⑥ Custo do cabo (3 fases)          [↓ melhor, score invertido]
  ⑦ Custo das torres (268 torres)    [↓ melhor, score invertido]
       25% autoportantes  +  75% estaiadas
       C_torre_i = C_base × (m_i / m_Drake)^α

Área do polígono:
  A = ½ Σ_{i=0}^{N-1} r_i · r_{i+1} · sin(2π/N)
  A_max = (N/2) · sin(2π/N)  ≈  2.736  para N = 7
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 150,
    'text.usetex': False,
})

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTES E DADOS FIXOS DA LINHA
# ═══════════════════════════════════════════════════════════════════════════════
f    = 60           # Hz
VLL  = 138e3        # V  (tensão de linha)
Lkm  = 80           # km (extensão)
mu0  = 4 * np.pi * 1e-7   # H/m
eps0 = 8.854e-12          # F/m
GMD  = 7.727        # m   — GMD calculado com xc = [3.4,-3.4,3.4,-3.4,3.4] m

# Torres
VSPAN    = 300                          # m — vão médio entre torres
N_TORRES = int(Lkm * 1000 / VSPAN) + 1 # 268 torres no total
N_AUTO   = round(0.25 * N_TORRES)       # 67 autoportantes  (25 %)
N_ESTAI  = N_TORRES - N_AUTO            # 201 estaiadas     (75 %)
DRAKE_MASS = 2427   # kg/km — massa de referência (Drake) para escalonamento

N_CRIT = 7
A_MAX  = (N_CRIT / 2) * np.sin(2 * np.pi / N_CRIT)  # ≈ 2.736

LABELS = [
    'Perdas\nresistivas', 'Ampacidade', 'Perfil de\ntensão',
    'P nominal\n(SIL)', 'P máxima', 'Custo\ncabo', 'Custo\ntorres',
]
COLORS = [
    '#3266AD', '#1D9E75', '#D85A30', '#D4537E',
    '#534AB7', '#639922', '#BA7517', '#888780', '#E24B4A',
]

# ═══════════════════════════════════════════════════════════════════════════════
# 2. BANCO DE DADOS ACSR
# ═══════════════════════════════════════════════════════════════════════════════
# Campos: nome, bitola (kcmil), encordoamento, diâmetro externo (m),
#         GMR (m), R_AC a 75°C (Ω/km), corrente máxima (A), massa (kg/km)
conductors = [
    dict(name='Linnet',   kcmil=336.4, enc='26/7', od=18.39e-3, gmr=0.00741, rac=0.1059, imax=530,  mass=1026),
    dict(name='Hawk',     kcmil=477,   enc='26/7', od=21.79e-3, gmr=0.00881, rac=0.0747, imax=660,  mass=1454),
    dict(name='Osprey',   kcmil=556.5, enc='26/7', od=23.55e-3, gmr=0.00950, rac=0.0641, imax=730,  mass=1698),
    dict(name='Grosbeak', kcmil=636,   enc='26/7', od=25.15e-3, gmr=0.01013, rac=0.0561, imax=800,  mass=1939),
    dict(name='Drake',    kcmil=795,   enc='26/7', od=28.14e-3, gmr=0.01130, rac=0.0450, imax=907,  mass=2427),
    dict(name='Tern',     kcmil=795,   enc='45/7', od=27.00e-3, gmr=0.01135, rac=0.0448, imax=900,  mass=2361),
    dict(name='Rail',     kcmil=954,   enc='45/7', od=29.59e-3, gmr=0.01245, rac=0.0374, imax=990,  mass=2831),
    dict(name='Cardinal', kcmil=954,   enc='54/7', od=30.38e-3, gmr=0.01214, rac=0.0374, imax=996,  mass=2892),
    dict(name='Bittern',  kcmil=1272,  enc='45/7', od=35.10e-3, gmr=0.01473, rac=0.0282, imax=1156, mass=3775),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PARÂMETROS DE PROJETO (VALORES PADRÃO)
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    P_MW   = 60,      # MW   — potência transmitida
    FP     = 0.92,    #       — fator de potência
    FC     = 0.70,    #       — fator de carga anual
    ec     = 350,     # R$/MWh — preço da energia
    cc_usd = 8.5,     # USD/kg — preço do condutor ACSR
    fx     = 5.8,     # R$/USD — câmbio
    nn     = 30,      # anos   — vida útil
    rr     = 0.08,    # /ano   — taxa de desconto (TMA)
    ta     = 220e3,   # R$     — custo base torre autoportante
    te     = 85e3,    # R$     — custo base torre estaiada
    alpha  = 0.60,    #        — expoente de escala de massa para torres
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. FUNÇÕES DE CÁLCULO
# ═══════════════════════════════════════════════════════════════════════════════

def line_params(c):
    """Parâmetros elétricos distribuídos (L, C, XL, Zc, SIL, Pmax)."""
    r_ext = c['od'] / 2                                    # m
    LH_km = mu0 / (2 * np.pi) * np.log(GMD / c['gmr']) * 1e3   # H/km
    XL    = 2 * np.pi * f * LH_km                          # Ω/km
    CF_km = 2 * np.pi * eps0 / np.log(GMD / r_ext) * 1e3  # F/km
    Zc    = np.sqrt(LH_km / CF_km)                         # Ω
    SIL   = VLL**2 / Zc / 1e6                              # MW
    Pmax  = VLL**2 / (XL * Lkm) / 1e6                     # MW (limite teórico)
    return dict(LH_km=LH_km, XL=XL, CF_km=CF_km, Zc=Zc, SIL=SIL, Pmax=Pmax)


def poly_area(sv):
    """Área do polígono de N_CRIT lados com scores sv ∈ [0,1]."""
    n = len(sv)
    return 0.5 * sum(sv[i] * sv[(i+1) % n] * np.sin(2 * np.pi / n)
                     for i in range(n))


def compute_all(P_MW, FP, FC, ec, cc_usd, fx, nn, rr, ta, te, alpha):
    """
    Calcula métricas brutas e scores normalizados para todos os condutores.
    Retorna: (results, I_carga, annuity)
    """
    I        = P_MW * 1e6 / (np.sqrt(3) * VLL * FP)   # A — corrente de carga
    FC_loss  = 0.3 * FC + 0.7 * FC**2                  # fator de perdas
    annuity  = rr / (1 - (1 + rr)**(-nn))              # CRF (Capital Recovery Factor)
    cf, sf   = FP, np.sqrt(1 - FP**2)                  # cos φ, sen φ

    raw = []
    for c in conductors:
        lp = line_params(c)

        # ① Perdas resistivas (kW)
        perda = 3 * I**2 * c['rac'] * Lkm / 1e3

        # ③ Regulação de tensão — equação de curto-circuito (%)
        dV = I * (c['rac'] * Lkm * cf + lp['XL'] * Lkm * sf) \
             / (VLL / np.sqrt(3)) * 100

        # ⑥ Custo do cabo — 3 fases (R$)
        cabo_R = c['mass'] * Lkm * 3 * (cc_usd * fx)

        # ⑦ Custo das torres — escalonado pela massa relativa (R$)
        fscale   = (c['mass'] / DRAKE_MASS)**alpha
        torres_R = N_AUTO * (ta * fscale) + N_ESTAI * (te * fscale)

        # Custo anual das perdas e parcelas de capital
        # perda em kW → energia em MWh = perda*8760*FC_loss/1000
        cost_perda  = perda * 8760 * FC_loss * ec / 1000   # R$/ano
        cost_cabo   = annuity * cabo_R                  # R$/ano
        cost_torres = annuity * torres_R                # R$/ano
        cost_total  = cost_perda + cost_cabo + cost_torres

        raw.append(dict(
            **c, **lp,
            I=I, perda=perda, dV=dV,
            cabo_R=cabo_R, torres_R=torres_R,
            cost_perda=cost_perda, cost_cabo=cost_cabo,
            cost_torres=cost_torres, cost_total=cost_total,
            ok=(I <= c['imax']),
        ))

    # Normalização min-max por critério
    def mn(k): return min(r[k] for r in raw)
    def mx(k): return max(r[k] for r in raw)

    def norm(v, k, inv=False):
        lo, hi = mn(k), mx(k)
        s = (v - lo) / (hi - lo) if hi > lo else 0.5
        return (1 - s) if inv else s

    results = []
    for r in raw:
        sv = [
            norm(r['perda'],    'perda',    inv=True),   # ① menor é melhor
            norm(r['imax'],     'imax',     inv=False),  # ② maior é melhor
            norm(r['dV'],       'dV',       inv=True),   # ③ menor é melhor
            norm(r['SIL'],      'SIL',      inv=False),  # ④ maior é melhor
            norm(r['Pmax'],     'Pmax',     inv=False),  # ⑤ maior é melhor
            norm(r['cabo_R'],   'cabo_R',   inv=True),   # ⑥ menor é melhor
            norm(r['torres_R'], 'torres_R', inv=True),   # ⑦ menor é melhor
        ]
        results.append(dict(**r, sv=sv, area=poly_area(sv)))

    return results, I, annuity


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FUNÇÕES DE VISUALIZAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

def plot_radar(results, ax, title=''):
    """Radar/spider chart com polígono de 7 critérios."""
    angles = np.linspace(0, 2 * np.pi, N_CRIT, endpoint=False)
    ang_cl = np.append(angles, angles[0])

    ax.set_thetagrids(np.degrees(angles), LABELS, fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=6, color='gray')
    ax.grid(color='gray', alpha=0.25, linewidth=0.5)

    valid = [r for r in results if r['ok']]
    opt   = max(valid, key=lambda r: r['area'])

    for r, col in zip(results, COLORS):
        sv_cl = np.append(r['sv'], r['sv'][0])
        isopt = (r['name'] == opt['name'])
        lw    = 2.5 if isopt else 0.7
        fa    = 0.20 if isopt else 0.03
        la    = 0.90 if isopt else 0.30
        ax.plot(ang_cl, sv_cl, color=col, linewidth=lw, alpha=la,
                label=r['name'] if isopt else '_nolegend_')
        ax.fill(angles, r['sv'], color=col, alpha=fa)

    ax.set_title(title or f'Ótimo multicritério: {opt["name"]}\n'
                 f'({opt["area"] / A_MAX * 100:.1f}% do máximo)',
                 fontsize=9.5, pad=14)


def plot_bars(results, ax):
    """Barras horizontais — área normalizada por condutor."""
    sorted_r = sorted(results, key=lambda r: r['area'], reverse=True)
    valid    = [r for r in results if r['ok']]
    opt      = max(valid, key=lambda r: r['area'])

    names = [r['name'] for r in sorted_r]
    pcts  = [r['area'] / A_MAX * 100 for r in sorted_r]
    cols  = [COLORS[results.index(r)] for r in sorted_r]
    alphs = [0.85 if r['ok'] else 0.30 for r in sorted_r]

    for i, (name, pct, col, alph) in enumerate(zip(names, pcts, cols, alphs)):
        ax.barh(name, pct, color=col, alpha=alph, height=0.6)
        ax.text(pct + 0.4, i, f'{pct:.1f}%', va='center', fontsize=7.5,
                color='black' if alphs[i] > 0.5 else 'gray')

    opt_pct = opt['area'] / A_MAX * 100
    ax.axvline(opt_pct, color='#1D9E75', linestyle='--',
               linewidth=1.2, alpha=0.7, label=f'Ótimo: {opt["name"]}')
    ax.set_xlabel('Área do polígono (% do máximo teórico)', fontsize=9)
    ax.set_xlim(0, 105)
    ax.set_title('Classificação por área do polígono de 7 critérios', fontsize=9.5)
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7.5)


def plot_sensitivity_2param(results_fn):
    """
    Análise de sensibilidade bivariada:
    preço da energia  (ec)  vs  preço do condutor (cc_usd).
    Retorna a figura com 2 subplots.
    """
    param_vals = {
        'ec':    (np.linspace(100, 700, 35), 'Preço da energia (R$/MWh)'),
        'cc_usd':(np.linspace(3, 18,   35), 'Preço do condutor (USD/kg)'),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (pk, (prange, plabel)) in zip(axes, param_vals.items()):
        opt_names, opt_areas = [], []
        for pval in prange:
            kw = dict(**DEFAULTS)
            kw[pk] = pval
            res, *_ = results_fn(**kw)
            valid = [r for r in res if r['ok']]
            opt   = max(valid, key=lambda r: r['area'])
            opt_names.append(opt['name'])
            opt_areas.append(opt['area'] / A_MAX * 100)

        # Área do condutor ótimo
        ax2 = ax.twinx()
        ax.plot(prange, opt_areas, '-', color='#3266AD',
                linewidth=1.5, label='Área ótima (%)')
        ax.set_ylabel('Área (% do máximo)', color='#3266AD', fontsize=8.5)
        ax.tick_params(axis='y', labelcolor='#3266AD')
        ax.set_ylim(0, 100)

        # Identificação visual do condutor ótimo
        prev = None
        cmap = {c['name']: COLORS[i] for i, c in enumerate(conductors)}
        for pval, name in zip(prange, opt_names):
            col = cmap[name]
            ax2.scatter(pval, 0.5, color=col, s=30, zorder=3, alpha=0.8)
            if prev and prev != name:
                ax.axvline(pval, color='gray', linestyle=':', linewidth=0.7)
            prev = name

        ax2.set_ylim(0, 1)
        ax2.set_yticks([])
        ax.set_xlabel(plabel, fontsize=8.5)
        ax.set_title(f'Sensibilidade: {plabel}', fontsize=9.5)
        ax.grid(alpha=0.2)

        # Legenda com condutores que aparecem na faixa
        seen = dict.fromkeys(opt_names)
        patches = [mpatches.Patch(color=cmap[n], label=n) for n in seen]
        ax.legend(handles=patches, fontsize=7, loc='lower right',
                  ncol=2, framealpha=0.8)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAÍDA TEXTO
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(results, I, annuity):
    valid = [r for r in results if r['ok']]
    opt   = max(valid, key=lambda r: r['area'])

    hdr = (f"\n{'Condutor':<10} {'kcmil':>6}  "
           f"{'Perda':>6}  {'ΔV%':>5}  {'SIL':>5}  {'Pmax':>5}  "
           f"{'Cabo':>6}  {'Torres':>6}  {'Área':>7}  {'%Amax':>6}  Obs")
    sep = '─' * len(hdr)
    print(sep)
    print(f"{'LT 138 kV / 80 km — Resultados Multicritério (7 vértices)':^{len(hdr)}}")
    print(f"{'I_carga = ' + f'{I:.1f} A':^{len(hdr)}}")
    print(sep)
    print(hdr)
    print(sep)
    for r in sorted(results, key=lambda x: x['area'], reverse=True):
        obs = '★ ÓTIMO' if r['name'] == opt['name'] else \
              'sobrecarga' if not r['ok'] else ''
        print(f"{r['name']:<10} {r['kcmil']:>6.0f}  "
              f"{r['perda']:>6.0f}  {r['dV']:>5.2f}  "
              f"{r['SIL']:>5.1f}  {r['Pmax']:>5.0f}  "
              f"{r['cabo_R']/1e6:>6.2f}  {r['torres_R']/1e6:>6.2f}  "
              f"{r['area']:>7.4f}  {r['area']/A_MAX*100:>5.1f}%   {obs}")
    print(sep)
    print(f"\nColunas: Perda (kW) | ΔV% | SIL (MW) | Pmax (MW) | "
          f"Cabo (MR$) | Torres (MR$)")
    print(f"Condutor ótimo  : {opt['name']} ({opt['kcmil']:.0f} kcmil {opt['enc']})")
    print(f"Área máxima     : {opt['area']:.4f}  →  "
          f"{opt['area']/A_MAX*100:.1f}% de {A_MAX:.4f}")
    print(f"Custo total/ano : R$ {opt['cost_total']/1e3:.0f} k/ano  "
          f"(cabo: {opt['cost_cabo']/1e3:.0f} | torres: {opt['cost_torres']/1e3:.0f} "
          f"| perdas: {opt['cost_perda']/1e3:.0f}  kR$/ano)")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. EXECUÇÃO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Cálculo com parâmetros padrão ──────────────────────────────────────────
    results, I, annuity = compute_all(**DEFAULTS)
    print_results(results, I, annuity)

    valid = [r for r in results if r['ok']]
    opt   = max(valid, key=lambda r: r['area'])

    # ── Figura 1: Radar + Barras ───────────────────────────────────────────────
    fig1 = plt.figure(figsize=(13, 5.5))
    ax_rad = fig1.add_subplot(1, 2, 1, polar=True)
    ax_bar = fig1.add_subplot(1, 2, 2)

    plot_radar(results, ax=ax_rad,
               title=f'Polígono de 7 critérios\n(ótimo: {opt["name"]})')
    plot_bars(results, ax=ax_bar)

    # Legenda global (condutores)
    legend_elements = [
        Line2D([0], [0], color=COLORS[i], linewidth=2,
               label=f'{c["name"]} ({c["kcmil"]:.0f} kcmil {c["enc"]})',
               alpha=0.9 if results[i]['ok'] else 0.35)
        for i, c in enumerate(conductors)
    ]
    fig1.legend(handles=legend_elements, loc='lower center', ncol=5,
                fontsize=7.5, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig1.suptitle('LT 138 kV / 80 km — Seleção Multicritério ACSR (7 critérios)\n'
                  r'$\mathbf{x}_c = [3.4,\,-3.4,\,3.4,\,-3.4,\,3.4]$ m  '
                  r'$|$  GMD = 7.727 m  $|$  268 torres (25% AP + 75% Est.)',
                  fontsize=10, fontweight='bold')
    plt.tight_layout(rect=[0, 0.10, 1, 0.95])
    fig1.savefig('fig_radar_bars.pdf', bbox_inches='tight')
    fig1.savefig('fig_radar_bars.png', bbox_inches='tight', dpi=150)
    print('\nFigura 1 salva: fig_radar_bars.pdf / .png')

    # ── Figura 2: Análise de sensibilidade ────────────────────────────────────
    fig2 = plot_sensitivity_2param(compute_all)
    fig2.suptitle('Análise de Sensibilidade — Condutor Ótimo vs Parâmetros Econômicos',
                  fontsize=10, fontweight='bold', y=1.02)
    fig2.savefig('fig_sensitivity.pdf', bbox_inches='tight')
    fig2.savefig('fig_sensitivity.png', bbox_inches='tight', dpi=150)
    print('Figura 2 salva: fig_sensitivity.pdf / .png')
