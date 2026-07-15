#!/usr/bin/env python3
"""
Selecao Multicriteria de Condutores ACSR -- LT 138 kV / 80 km
Criterio: Maximizacao da area do poligono de 7 vertices (scores normalizados).

Fonte de dados: Tabela 4-04 -- 138kV CST1 (parametros de sequencia positiva
e zero para feixes simples 1x e duplos 2x, obtidos de planilha de projeto).

Os 7 criterios (vertices do poligono):
  (1) Perdas resistivas trifasicas    [menor -> melhor, score invertido]
  (2) Ampacidade Normal [A]           [maior -> melhor]
  (3) Regulacao de tensao DeltaV%     [menor -> melhor, score invertido]
  (4) Potencia nominal SIL = V^2/Zc  [maior -> melhor]
  (5) Potencia maxima = V^2/(x1*L)   [maior -> melhor]
  (6) Custo do cabo (3 fases)         [menor -> melhor, score invertido]
  (7) Custo das torres (268 torres)   [menor -> melhor, score invertido]
       25% autoportantes + 75% estaiadas
       C_torre_i = C_base * (m_i / m_ref)^alpha * k_bundle

Area do poligono de N=7 vertices igualmente espacados:
  A = 0.5 * sum_{i=0}^{6}  r_i * r_{(i+1) mod 7} * sin(2*pi/7)
  A_max = (7/2) * sin(2*pi/7) ~= 2.736

Parametros calculados a partir dos dados tabelados:
  Zc   = sqrt(x1 / b1_SI)   onde b1_SI = b1[uS/km] * 1e-6  [Ohm]
  SIL  = VLL^2 / Zc                                         [MW]
  Pmax = VLL^2 / (x1 * L)   (limite de estabilidade)        [MW]
  DV%  = I*(r1*L*cos_phi + x1*L*sin_phi)/(VLL/sqrt(3)) * 100

Massa estimada por regressao linear (ACSR conhecidos):
  m_simples [kg/km] ~= 3.056 * MCM + 26.2
  m_feixe   = bundle * m_simples
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# =============================================================================
# 1. CONSTANTES E PARAMETROS FIXOS DA LINHA
# =============================================================================
VLL    = 138e3   # V   -- tensao de linha nominal
Lkm    = 80      # km  -- extensao da linha
VSPAN  = 300     # m   -- vao medio entre torres

N_TORRES = int(Lkm * 1000 / VSPAN) + 1  # 268 torres
N_AUTO   = round(0.25 * N_TORRES)        # 67  autoportantes (25%)
N_ESTAI  = N_TORRES - N_AUTO             # 201 estaiadas     (75%)

# Massa de referencia: Drake 1x estimada por regressao
DRAKE_MCM  = 795
DRAKE_MASS = 3.056 * DRAKE_MCM + 26.2   # ~2456 kg/km

N_CRIT = 7
A_MAX  = (N_CRIT / 2) * np.sin(2 * np.pi / N_CRIT)  # ~2.736

LABELS_CRIT = [
    'Perdas\nresist.', 'Ampacidade', 'Perfil\ntensao',
    'P nom.\n(SIL)', 'P max.', 'Custo\ncabo', 'Custo\ntorres',
]

# Paleta: tons quentes para feixe simples, tons frios para feixe duplo
COLORS_1X = [
    '#D85A30','#D4537E','#BA7517','#639922',
    '#534AB7','#3266AD','#1D9E75','#888780',
    '#E24B4A','#B85C00','#7A5C99','#00887A','#556B2F','#8B4513',
]
COLORS_2X = [
    '#1565C0','#00838F','#283593','#00695C',
    '#1B5E20','#4A148C','#880E4F','#BF360C',
    '#37474F','#4E342E','#006064','#1A237E','#33691E','#827717',
]

# =============================================================================
# 2. BANCO DE DADOS -- TABELA 4-04 -- 138kV CST1
# =============================================================================
# Campos: name, mcm, bundle, In[A], Ie[A],
#         r1[Ohm/km], x1[Ohm/km], b1[uS/km],
#         r0[Ohm/km], x0[Ohm/km], b0[uS/km]

_COLS = ('name','mcm','bundle','In','Ie','r1','x1','b1','r0','x0','b0')

_RAW = (
    # -- Feixe simples (1 cabo / fase) ---------------------------------------
    ('1x Linnet',   336, 1,  435,  580, 0.1914, 0.5014, 3.3018, 0.5379, 1.4813, 2.1619),
    ('1x Ibis',     397, 1,  485,  645, 0.1622, 0.4952, 3.3456, 0.5080, 1.4761, 2.1766),
    ('1x Hawk',     477, 1,  540,  720, 0.1354, 0.4883, 3.3951, 0.4809, 1.4695, 2.1967),
    ('1x Dove',     556, 1,  595,  795, 0.1160, 0.4821, 3.4379, 0.4609, 1.4645, 2.2103),
    ('1x Squab',    605, 1,  625,  840, 0.1069, 0.4790, 3.4601, 0.4503, 1.4638, 2.2107),
    ('1x Grosbeak', 636, 1,  640,  865, 0.1017, 0.4772, 3.4748, 0.4459, 1.4608, 2.2212),
    ('1x Gannet',   666, 1,  660,  890, 0.0971, 0.4753, 3.4891, 0.4416, 1.4583, 2.2290),
    ('1x Starling', 716, 1,  690,  930, 0.0906, 0.4727, 3.5098, 0.4348, 1.4562, 2.2359),
    ('1x Drake',    795, 1,  735,  995, 0.0817, 0.4690, 3.5410, 0.4258, 1.4526, 2.2478),
    ('1x Tern',     795, 1,  730,  985, 0.0819, 0.4734, 3.5103, 0.4181, 1.4699, 2.1877),
    ('1x Ruddy',    900, 1,  785, 1065, 0.0726, 0.4691, 3.5478, 0.4095, 1.4644, 2.2059),
    ('1x Rail',     954, 1,  810, 1100, 0.0686, 0.4665, 3.5645, 0.4037, 1.4650, 2.2024),
    ('1x Ortolan', 1033, 1,  850, 1155, 0.0635, 0.4634, 3.5883, 0.3980, 1.4629, 2.2078),
    ('1x Bluejay', 1113, 1,  890, 1210, 0.0592, 0.4609, 3.6114, 0.3936, 1.4605, 2.2160),
    # -- Feixe duplo (2 cabos / fase) ----------------------------------------
    ('2x Linnet',   336, 2,  865, 1150, 0.0959, 0.3460, 4.7827, 0.4423, 1.3257, 2.7125),
    ('2x Ibis',     397, 2,  960, 1280, 0.0813, 0.3429, 4.8280, 0.4270, 1.3237, 2.7208),
    ('2x Hawk',     477, 2, 1070, 1430, 0.0679, 0.3394, 4.8794, 0.4134, 1.3206, 2.7358),
    ('2x Dove',     556, 2, 1175, 1580, 0.0582, 0.3363, 4.9230, 0.4030, 1.3186, 2.7428),
    ('2x Squab',    605, 2, 1235, 1660, 0.0536, 0.3348, 4.9447, 0.3970, 1.3195, 2.7370),
    ('2x Grosbeak', 636, 2, 1275, 1715, 0.0510, 0.3339, 4.9604, 0.3952, 1.3173, 2.7482),
    ('2x Gannet',   666, 2, 1310, 1765, 0.0487, 0.3329, 4.9752, 0.3932, 1.3158, 2.7557),
    ('2x Starling', 716, 2, 1370, 1845, 0.0455, 0.3317, 4.9962, 0.3897, 1.3150, 2.7596),
    ('2x Drake',    795, 2, 1460, 1970, 0.0410, 0.3298, 5.0278, 0.3851, 1.3132, 2.7686),
    ('2x Tern',     795, 2, 1445, 1950, 0.0411, 0.3320, 4.9906, 0.3773, 1.3284, 2.6851),
    ('2x Ruddy',    900, 2, 1555, 2105, 0.0364, 0.3299, 5.0290, 0.3734, 1.3251, 2.7020),
    ('2x Rail',     954, 2, 1610, 2185, 0.0344, 0.3286, 5.0446, 0.3695, 1.3269, 2.6909),
    ('2x Ortolan', 1033, 2, 1690, 2295, 0.0319, 0.3270, 5.0681, 0.3663, 1.3264, 2.6921),
    ('2x Bluejay', 1113, 2, 1765, 2400, 0.0297, 0.3258, 5.0912, 0.3641, 1.3253, 2.6980),
)

conductors = [dict(zip(_COLS, row)) for row in _RAW]

# =============================================================================
# 3. PARAMETROS DE PROJETO (VALORES PADRAO)
# =============================================================================
DEFAULTS = dict(
    P_MW   = 60,     # MW      -- potencia transmitida
    FP     = 0.92,   #         -- fator de potencia
    FC     = 0.70,   #         -- fator de carga anual
    ec     = 350,    # R$/MWh  -- preco da energia
    cc_usd = 8.5,    # USD/kg  -- preco do condutor ACSR
    fx     = 5.8,    # R$/USD  -- cambio
    nn     = 30,     # anos    -- vida util
    rr     = 0.08,   # /ano    -- taxa de desconto (TMA)
    ta     = 220e3,  # R$      -- custo base torre autoportante
    te     = 85e3,   # R$      -- custo base torre estaiada
    alpha  = 0.60,   #         -- expoente de escala de massa para torres
    k2x    = 0.25,   #         -- sobrecusto relativo de torre feixe duplo
)

# =============================================================================
# 4. FUNCOES DE CALCULO
# =============================================================================

def derived_params(c):
    """Calcula Zc, SIL, Pmax e massa a partir dos parametros tabelados."""
    b1_SI = c['b1'] * 1e-6                         # uS/km -> S/km
    Zc    = np.sqrt(c['x1'] / b1_SI)               # Ohm
    SIL   = VLL**2 / Zc / 1e6                      # MW
    Pmax  = VLL**2 / (c['x1'] * Lkm) / 1e6         # MW
    mass  = c['bundle'] * (3.056 * c['mcm'] + 26.2) # kg/km (feixe completo)
    return dict(Zc=Zc, SIL=SIL, Pmax=Pmax, mass=mass)


def poly_area(sv):
    """Area do poligono de N_CRIT lados com scores sv em [0, 1]."""
    n = len(sv)
    return 0.5 * sum(
        sv[i] * sv[(i + 1) % n] * np.sin(2 * np.pi / n)
        for i in range(n)
    )


def compute_all(P_MW, FP, FC, ec, cc_usd, fx, nn, rr,
                ta, te, alpha, k2x):
    """
    Calcula metricas brutas e scores normalizados para os 28 condutores/feixe.
    Retorna: (results, I_carga, annuity)
    """
    I       = P_MW * 1e6 / (np.sqrt(3) * VLL * FP)  # A
    FC_loss = 0.3 * FC + 0.7 * FC**2                  # fator de perdas
    annuity = rr / (1 - (1 + rr)**(-nn))              # CRF
    cf, sf  = FP, np.sqrt(max(0.0, 1 - FP**2))        # cos phi, sen phi

    raw = []
    for c in conductors:
        dp = derived_params(c)

        # (1) Perdas resistivas -- 3 fases [kW]
        perda = 3 * I**2 * c['r1'] * Lkm / 1e3

        # (3) Regulacao de tensao -- modelo de curto-circuito [%]
        dV = (I * (c['r1'] * Lkm * cf + c['x1'] * Lkm * sf)
              / (VLL / np.sqrt(3)) * 100)

        # (6) Custo do cabo -- 3 fases [R$]
        cabo_R = dp['mass'] * Lkm * 3 * (cc_usd * fx)

        # (7) Custo das torres -- escalado por massa e tipo de feixe [R$]
        fscale   = (dp['mass'] / DRAKE_MASS)**alpha
        k_bundle = (1 + k2x) if c['bundle'] == 2 else 1.0
        torres_R = (N_AUTO * ta + N_ESTAI * te) * fscale * k_bundle

        # Custos anuais
        cost_perda  = perda * 8760 * FC_loss * ec / 1000  # R$/ano
        cost_cabo   = annuity * cabo_R                      # R$/ano
        cost_torres = annuity * torres_R                    # R$/ano
        cost_total  = cost_perda + cost_cabo + cost_torres

        raw.append(dict(
            **c, **dp,
            I=I, perda=perda, dV=dV,
            cabo_R=cabo_R, torres_R=torres_R,
            cost_perda=cost_perda, cost_cabo=cost_cabo,
            cost_torres=cost_torres, cost_total=cost_total,
            ok=(I <= c['In']),
        ))

    # Normalizacao min-max global (sobre os 28 condutores)
    def mn(k): return min(r[k] for r in raw)
    def mx(k): return max(r[k] for r in raw)

    def norm(v, k, inv=False):
        lo, hi = mn(k), mx(k)
        s = (v - lo) / (hi - lo) if hi > lo else 0.5
        return (1 - s) if inv else s

    results = []
    for r in raw:
        sv = [
            norm(r['perda'],    'perda',    inv=True),
            norm(r['In'],       'In',       inv=False),
            norm(r['dV'],       'dV',       inv=True),
            norm(r['SIL'],      'SIL',      inv=False),
            norm(r['Pmax'],     'Pmax',     inv=False),
            norm(r['cabo_R'],   'cabo_R',   inv=True),
            norm(r['torres_R'], 'torres_R', inv=True),
        ]
        results.append(dict(**r, sv=sv, area=poly_area(sv)))

    return results, I, annuity


def _optima(results):
    """Retorna condutores otimos para feixe 1x e 2x (apenas elegiveis)."""
    valid  = [r for r in results if r['ok']]
    valid1 = [r for r in valid if r['bundle'] == 1]
    valid2 = [r for r in valid if r['bundle'] == 2]
    opt1 = max(valid1, key=lambda r: r['area']) if valid1 else None
    opt2 = max(valid2, key=lambda r: r['area']) if valid2 else None
    return opt1, opt2

# =============================================================================
# 5. FUNCOES DE VISUALIZACAO
# =============================================================================

def _colors(results):
    """Mapeia nome do condutor para cor, separando 1x e 2x."""
    cmap = {}
    i1 = i2 = 0
    for r in results:
        if r['name'] not in cmap:
            if r['bundle'] == 1:
                cmap[r['name']] = COLORS_1X[i1 % len(COLORS_1X)]; i1 += 1
            else:
                cmap[r['name']] = COLORS_2X[i2 % len(COLORS_2X)]; i2 += 1
    return cmap


def plot_radar(results, ax, opt1, opt2):
    """Radar com apenas os dois condutores otimos destacados."""
    angles = np.linspace(0, 2 * np.pi, N_CRIT, endpoint=False)
    ang_cl = np.append(angles, angles[0])

    ax.set_thetagrids(np.degrees(angles), LABELS_CRIT, fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=6, color='gray')
    ax.grid(color='gray', alpha=0.25, linewidth=0.5)

    pairs = []
    if opt1: pairs.append((opt1, '#D85A30', f'Otimo 1x: {opt1["name"]}'))
    if opt2: pairs.append((opt2, '#1565C0', f'Otimo 2x: {opt2["name"]}'))
    for r, col, lbl in pairs:
        sv_cl = np.append(r['sv'], r['sv'][0])
        ax.plot(ang_cl, sv_cl, color=col, linewidth=2.5, label=lbl)
        ax.fill(angles, r['sv'], color=col, alpha=0.15)

    ax.legend(loc='upper right', bbox_to_anchor=(1.42, 1.18), fontsize=8)
    ax.set_title('Poligono de 7 criterios\n(otimos 1x e 2x)',
                 fontsize=9.5, pad=14)


def plot_bars(results, ax, opt1, opt2):
    """Barras horizontais para todos os 28 condutores, ordenadas por area."""
    cmap = _colors(results)
    sorted_r = sorted(results, key=lambda r: r['area'], reverse=True)
    opt_names = {r['name'] for r in [opt1, opt2] if r}

    for i, r in enumerate(sorted_r):
        col   = cmap[r['name']]
        alph  = 0.85 if r['ok'] else 0.25
        isopt = r['name'] in opt_names
        pct   = r['area'] / A_MAX * 100
        ax.barh(i, pct, color=col, alpha=alph, height=0.72,
                edgecolor='black' if isopt else col,
                linewidth=1.4 if isopt else 0)
        lbl = r['name'] + (' *' if isopt else '')
        ax.text(-0.5, i, lbl, ha='right', va='center', fontsize=7,
                fontweight='bold' if isopt else 'normal')
        ax.text(pct + 0.3, i, f'{pct:.1f}%', va='center', fontsize=6.5,
                color='black' if r['ok'] else 'gray')

    ax.set_yticks([])
    ax.set_xlim(-20, 102)
    ax.set_xlabel('Area do poligono (% de A_max = 2.736)', fontsize=9)
    ax.set_title('Ranking por area -- 28 condutores/feixe', fontsize=9.5)
    ax.grid(axis='x', alpha=0.25, linewidth=0.5)
    p1 = mpatches.Patch(color='#D85A30', label='Feixe simples (1x)')
    p2 = mpatches.Patch(color='#1565C0', label='Feixe duplo (2x)')
    ax.legend(handles=[p1, p2], fontsize=8, loc='lower right')


def plot_sensitivity(results_fn, axes):
    """
    Sensibilidade do condutor otimo (1x e 2x) a dois parametros:
      axes[0]: potencia transmitida P [MW]
      axes[1]: preco da energia ec  [R$/MWh]
    """
    sweeps = [
        (axes[0], 'P_MW',
         np.linspace(30, 250, 44),
         'Potencia transmitida P (MW)'),
        (axes[1], 'ec',
         np.linspace(100, 700, 44),
         'Preco da energia (R$/MWh)'),
    ]
    for ax, pk, prange, plabel in sweeps:
        names1, names2 = [], []
        areas1, areas2 = [], []
        for pval in prange:
            kw = dict(**DEFAULTS); kw[pk] = pval
            res, *_ = results_fn(**kw)
            o1, o2 = _optima(res)
            names1.append(o1['name'] if o1 else '--')
            names2.append(o2['name'] if o2 else '--')
            areas1.append(o1['area'] / A_MAX * 100 if o1 else 0)
            areas2.append(o2['area'] / A_MAX * 100 if o2 else 0)

        ax.plot(prange, areas1, '-',  color='#D85A30', lw=1.8,
                label='Area otimo 1x')
        ax.plot(prange, areas2, '--', color='#1565C0', lw=1.8,
                label='Area otimo 2x')
        ax.set_ylim(0, 100)
        ax.set_xlabel(plabel, fontsize=9)
        ax.set_ylabel('Area (% do maximo)', fontsize=9)
        ax.set_title(f'Sensibilidade -- {plabel.split("(")[0].strip()}',
                     fontsize=9.5)
        ax.grid(alpha=0.25)

        # Pontos coloridos indicando o condutor otimo em cada valor
        ax2 = ax.twinx()
        cmap1 = {n: COLORS_1X[i % 14]
                 for i, n in enumerate(dict.fromkeys(names1))}
        cmap2 = {n: COLORS_2X[i % 14]
                 for i, n in enumerate(dict.fromkeys(names2))}
        for pval, n1, n2 in zip(prange, names1, names2):
            if n1 != '--':
                ax2.scatter(pval, 0.75, color=cmap1[n1],
                            s=18, zorder=3, alpha=0.9)
            if n2 != '--':
                ax2.scatter(pval, 0.25, color=cmap2[n2],
                            s=18, zorder=3, alpha=0.9, marker='D')
        ax2.set_ylim(0, 1)
        ax2.set_yticks([0.75, 0.25])
        ax2.set_yticklabels(['1x', '2x'], fontsize=7)

        seen1 = dict.fromkeys(n for n in names1 if n != '--')
        seen2 = dict.fromkeys(n for n in names2 if n != '--')
        handles = (
            [mpatches.Patch(color=cmap1[n], label=f'1x: {n}')
             for n in seen1] +
            [mpatches.Patch(color=cmap2[n], label=f'2x: {n}', alpha=0.7)
             for n in seen2]
        )
        ax.legend(handles=handles, fontsize=6.5, ncol=2,
                  loc='lower right', framealpha=0.85)

# =============================================================================
# 6. SAIDA TEXTO
# =============================================================================

def print_results(results, I, annuity):
    opt1, opt2 = _optima(results)
    W   = 104
    sep = '-' * W
    print(sep)
    print('LT 138 kV / 80 km | Tabela 4-04 | '
          'Selecao Multicriteria (7 criterios)'.center(W))
    print(f'I_carga = {I:.1f} A | A_max = {A_MAX:.4f}'.center(W))
    print(sep)
    hdr = (f"{'Feixe/Cabo':<14} {'MCM':>5} {'In':>5} "
           f"{'r1':>6} {'x1':>6} {'b1':>7} {'Zc':>7} "
           f"{'SIL':>6} {'Perda':>7} {'dV%':>5} "
           f"{'Cabo':>6} {'Torres':>7} {'Area':>7} {'%Amax':>6}  Obs")
    print(hdr)
    print(sep)
    for bun, label in [(1, 'Feixe simples -- 1 cabo/fase'),
                       (2, 'Feixe duplo   -- 2 cabos/fase')]:
        print(f'  [{label}]')
        sub = sorted([r for r in results if r['bundle'] == bun],
                     key=lambda r: r['area'], reverse=True)
        for r in sub:
            obs = ''
            if opt1 and r['name'] == opt1['name']: obs = '*** OTIMO 1x'
            if opt2 and r['name'] == opt2['name']: obs = '*** OTIMO 2x'
            if not r['ok']:                         obs = 'SOBRECARGA'
            print(f"  {r['name']:<13} {r['mcm']:>5} {r['In']:>5} "
                  f"{r['r1']:>6.4f} {r['x1']:>6.4f} {r['b1']:>7.4f} "
                  f"{r['Zc']:>7.1f} {r['SIL']:>6.1f} "
                  f"{r['perda']:>7.1f} {r['dV']:>5.2f} "
                  f"{r['cabo_R']/1e6:>6.2f} {r['torres_R']/1e6:>7.2f} "
                  f"{r['area']:>7.4f} {r['area']/A_MAX*100:>5.1f}%  {obs}")
    print(sep)
    print('Colunas: In[A] r1[Ohm/km] x1[Ohm/km] b1[uS/km] Zc[Ohm] '
          'SIL[MW] Perda[kW] dV[%] Cabo[MR$] Torres[MR$]')
    if opt1:
        print(f'\nOtimo 1x : {opt1["name"]:<15} '
              f'Area = {opt1["area"]:.4f} ({opt1["area"]/A_MAX*100:.1f}%) | '
              f'Custo total = R$ {opt1["cost_total"]/1e3:.0f} k/ano')
    if opt2:
        print(f'Otimo 2x : {opt2["name"]:<15} '
              f'Area = {opt2["area"]:.4f} ({opt2["area"]/A_MAX*100:.1f}%) | '
              f'Custo total = R$ {opt2["cost_total"]/1e3:.0f} k/ano')

# =============================================================================
# 7. EXECUCAO PRINCIPAL
# =============================================================================

if __name__ == '__main__':

    # -- Calculo com parametros padrao ----------------------------------------
    results, I, annuity = compute_all(**DEFAULTS)
    print_results(results, I, annuity)

    opt1, opt2 = _optima(results)

    # -- Figura 1: Radar + Barras (28 condutores) -----------------------------
    fig1 = plt.figure(figsize=(15, 7))
    ax_rad = fig1.add_subplot(1, 2, 1, polar=True)
    ax_bar = fig1.add_subplot(1, 2, 2)

    plot_radar(results, ax_rad, opt1, opt2)
    plot_bars(results, ax_bar, opt1, opt2)

    fig1.suptitle(
        'LT 138 kV / 80 km -- Selecao Multicriteria ACSR | Tabela 4-04\n'
        '28 condutores/feixe | 7 criterios | '
        '268 torres (25% AP + 75% Est.) | xc=[3.4,-3.4,3.4,-3.4,3.4] m',
        fontsize=10, fontweight='bold')
    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    fig1.savefig('fig_radar_bars.pdf', bbox_inches='tight')
    fig1.savefig('fig_radar_bars.png', bbox_inches='tight', dpi=150)
    print('\nFigura 1 salva: fig_radar_bars.pdf / .png')

    # -- Figura 2: Sensibilidade (P e ec) -------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    plot_sensitivity(compute_all, axes2)
    fig2.suptitle(
        'Analise de Sensibilidade -- Condutor Otimo vs P e Preco da Energia',
        fontsize=10, fontweight='bold')
    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    fig2.savefig('fig_sensitivity.pdf', bbox_inches='tight')
    fig2.savefig('fig_sensitivity.png', bbox_inches='tight', dpi=150)
    print('Figura 2 salva: fig_sensitivity.pdf / .png')
