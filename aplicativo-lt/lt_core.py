"""
lt_core.py
==========
Núcleo de cálculo (sem interface) para o frontend de parâmetros unitários
de linhas de transmissão aéreas.

Separa a física da camada de apresentação (app.py em Streamlit), de modo que
as rotinas possam ser testadas isoladamente.

EEE 457 - Transmissão de Energia Elétrica
Escola Politécnica / COPPE - UFRJ

As matrizes de impedância (Z) e admitância (Y) por unidade de comprimento são
obtidas com `line_cable_param.czyl_overhead_bundled`, exatamente como no
notebook eee457-06_calculo_param_unitarios.ipynb. A flecha dos condutores é
calculada pela expressão da CATENÁRIA.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import line_cable_param as lcp

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
G = 9.80665                 # aceleração da gravidade (m/s^2)
KGF_TO_N = G                # 1 kgf = 9.80665 N
ALPHA_AL = 0.00403          # coef. de temperatura do alumínio (1/°C)

# Cabo para-raios padrão 3/8" EHS
GW_RADIUS_DEFAULT = 4.57e-3        # raio efetivo (m)
GW_RDC_DEFAULT = 4.190e-3          # resistência CC (Ohm/m)
GW_RTS_DEFAULT = 6980.0            # carga de ruptura - RTS (kgf)
GW_WEIGHT_DEFAULT = 410.0          # peso nominal (kg/km)
GW_TENSION_FRAC_DEFAULT = 25.0     # tração horizontal de trabalho (% da RTS)


# ===========================================================================
# 1. CATENÁRIA
# ===========================================================================
def catenary_sag(span: float, a: float) -> float:
    """Flecha de uma catenária de parâmetro ``a`` para um vão ``span``.

    f = a * (cosh(L / 2a) - 1)

    O argumento é limitado para evitar overflow numérico quando ``a`` é
    muito pequeno (situação visitada apenas durante o cerco da raiz).
    """
    arg = np.minimum(span / (2.0 * a), 700.0)
    return a * (np.cosh(arg) - 1.0)


def catenary_param_from_sag(span: float, sag: float) -> float:
    """Parâmetro da catenária ``a = H/w`` dado o par (vão, flecha).

    Resolve numericamente  f = a (cosh(L/2a) - 1)  para ``a``.
    A função é monotônica decrescente em ``a``, garantindo raiz única.
    """
    if sag <= 0:
        raise ValueError("A flecha deve ser positiva.")
    func = lambda a: catenary_sag(span, a) - sag
    return brentq(func, span / 1.0e4, span * 1.0e6, xtol=1e-9, rtol=1e-12)


def sag_from_tension(span: float, weight_per_m: float, H: float):
    """Flecha pela catenária a partir do peso linear e da tração horizontal.

    Parâmetros
    ----------
    span : vão entre estruturas (m)
    weight_per_m : peso do condutor por unidade de comprimento (N/m)
    H : tração horizontal (N)

    Retorna
    -------
    (flecha [m], parâmetro a [m], tração na extremidade T [N])
    """
    a = H / weight_per_m                       # parâmetro da catenária
    f = catenary_sag(span, a)
    T = H * np.cosh(span / (2.0 * a))          # tração máxima (no ponto de fixação)
    return f, a, T


def catenary_profile(span: float, sag: float, h_attach: float, n: int = 200):
    """Perfil de altura do condutor ao longo do vão (para visualização).

    Retorna (x, h) com x em [-L/2, L/2] e h = altura acima do solo.
    """
    a = catenary_param_from_sag(span, sag)
    x = np.linspace(-span / 2.0, span / 2.0, n)
    h = h_attach - sag + (a * np.cosh(x / a) - a)
    return x, h


def ground_wire_sag(span: float,
                    rts_kgf: float = GW_RTS_DEFAULT,
                    weight_kg_per_km: float = GW_WEIGHT_DEFAULT,
                    tension_frac: float = GW_TENSION_FRAC_DEFAULT):
    """Flecha do cabo para-raios pela catenária (modelo tração + peso).

    Reproduz a expressão:

        Hpr     = (tension_frac/100) * Tpr            [kgf]
        Ccatpr  = Hpr / Wpr                            [km]
        flecha  = 1000 * Ccatpr * (cosh(L/2Ccatpr) - 1)   [m],  L em km

    Para o cabo 3/8" EHS (Tpr = 6980 kgf, Wpr = 410 kg/km, 25 % da RTS)
    resulta flecha = 8.887 m para um vão de 550 m.

    Parâmetros
    ----------
    span : vão entre estruturas (m)
    rts_kgf : carga de ruptura do para-raios - RTS (kgf)
    weight_kg_per_km : peso nominal do para-raios (kg/km)
    tension_frac : tração horizontal de trabalho (% da RTS)

    Retorna
    -------
    (flecha [m], a [m], H [N], T [N])
        a : parâmetro da catenária
        H : tração horizontal
        T : tração máxima (no ponto de fixação)
    """
    H = (tension_frac / 100.0) * rts_kgf * KGF_TO_N    # tração horizontal (N)
    w = (weight_kg_per_km / 1000.0) * G                # peso linear (N/m)
    f, a, T = sag_from_tension(span, w, H)
    return f, a, H, T


# ===========================================================================
# 2. TABELA DE CONDUTORES
# ===========================================================================
# Mapeamento dos rótulos do CSV (condutores-caa.csv) para chaves internas.
COL = {
    "nome":      "Nome comercial",
    "bitola":    "Bitola (MCM)",
    "diam_cond": "Diâmetro do cond. (mm)",
    "diam_alma": "Diâmetro da Alma de Aço (mm)",
    "massa":     "Massa (aprox.) (kg/km)",
    "rts_a":     "Carga de ruptura (Classe A) (kgf)",
    "rdc20":     "Resistência elétrica máxima CC a 20 graus C (Ohm/km)",
    "rac75":     "Resistência elétrica máxima CA @ 60 Hz e 75 graus C (Ohm/km)",
    "gmr":       "Raio médio geométrico (m)",
    "ampac":     "Ampacidade (A)",
}


def load_conductor_table(file_or_path) -> pd.DataFrame:
    """Lê a tabela de condutores CAA (ACSR) do arquivo CSV."""
    df = pd.read_csv(file_or_path, encoding="utf-8-sig")
    faltando = [c for c in COL.values() if c not in df.columns]
    if faltando:
        raise ValueError(
            "Colunas ausentes no CSV: " + ", ".join(faltando))
    return df


def conductor_props(row: pd.Series, temp_c: float = 20.0) -> dict:
    """Extrai e converte as grandezas necessárias para o cálculo elétrico.

    Parâmetros
    ----------
    row : linha da tabela de condutores
    temp_c : temperatura de operação para correção da resistência CC (°C)

    Retorna
    -------
    dict com:
        nome      : nome comercial
        rf        : raio externo do condutor (m)
        rint      : raio interno = raio da alma de aço (m)
        rdc       : resistência CC corrigida para temp_c (Ohm/m)
        rdc20     : resistência CC a 20 °C (Ohm/m)
        rac75     : resistência CA 60 Hz / 75 °C (Ohm/m)
        massa     : massa por comprimento (kg/m)
        peso      : peso por comprimento (N/m)
        rts       : carga de ruptura classe A (N)
        gmr       : raio médio geométrico tabelado (m)
        ampacidade: corrente admissível (A)
    """
    nome = str(row[COL["nome"]])
    d_cond = float(row[COL["diam_cond"]]) * 1e-3      # mm -> m
    d_alma = float(row[COL["diam_alma"]]) * 1e-3      # mm -> m
    rf = d_cond / 2.0
    rint = d_alma / 2.0

    rdc20 = float(row[COL["rdc20"]]) * 1e-3           # Ohm/km -> Ohm/m
    rac75 = float(row[COL["rac75"]]) * 1e-3
    # correção de temperatura da resistência CC (referência 20 °C)
    rdc = rdc20 * (1.0 + ALPHA_AL * (temp_c - 20.0))

    massa = float(row[COL["massa"]]) / 1000.0         # kg/km -> kg/m
    peso = massa * G                                  # N/m
    rts = float(row[COL["rts_a"]]) * KGF_TO_N         # kgf -> N

    return {
        "nome": nome,
        "rf": rf,
        "rint": rint,
        "rdc": rdc,
        "rdc20": rdc20,
        "rac75": rac75,
        "massa": massa,
        "peso": peso,
        "rts": rts,
        "gmr": float(row[COL["gmr"]]),
        "ampacidade": float(row[COL["ampac"]]),
    }


# ===========================================================================
# 3. GEOMETRIA DO FEIXE E DAS ESTRUTURAS
# ===========================================================================
def bundle_offsets(nb: int, spacing: float, angle0_deg: float = 0.0) -> np.ndarray:
    """Deslocamentos (dx, dy) dos subcondutores de um feixe circular.

    Para nb >= 2 os subcondutores são dispostos sobre um polígono regular.
    O espaçamento ``spacing`` é a distância entre subcondutores adjacentes;
    o raio do feixe vale  R = s / (2 sen(pi/nb)).
    Para nb = 2 e angle0 = 0 obtém-se o par horizontal usual (± s/2).
    """
    if nb < 1:
        raise ValueError("nb deve ser >= 1.")
    if nb == 1:
        return np.zeros((1, 2))
    R = spacing / (2.0 * np.sin(np.pi / nb))
    ang = np.deg2rad(angle0_deg) + np.arange(nb) * 2.0 * np.pi / nb
    return np.column_stack([R * np.cos(ang), R * np.sin(ang)])


def elliptical_bundle_offsets(nb: int, a: float, b: float,
                              angle0_deg: float = 0.0) -> np.ndarray:
    """Deslocamentos (dx, dy) dos subcondutores de um feixe ELÍPTICO.

    Os ``nb`` subcondutores são igualmente espaçados no parâmetro angular
    ``t`` sobre a elipse de semi-eixos ``a`` (horizontal) e ``b`` (vertical):

        (x_k, y_k) = ( a cos(t0 + 2π k/nb),  b sin(t0 + 2π k/nb) )

    Se ``a == b``, recai no feixe circular de raio ``a``. Para ``nb = 2`` e
    ``angle0 = 0``, obtém-se um par horizontal em (± a, 0).
    """
    if nb < 1:
        raise ValueError("nb deve ser >= 1.")
    if nb == 1:
        return np.zeros((1, 2))
    if a <= 0 or b <= 0:
        raise ValueError("Semi-eixos a e b devem ser positivos.")
    ang = np.deg2rad(angle0_deg) + np.arange(nb) * 2.0 * np.pi / nb
    return np.column_stack([a * np.cos(ang), b * np.sin(ang)])


def build_coordinates(phase_centers, phase_sag, gw_positions, gw_sag,
                      nb, spacing, angle0_deg=0.0):
    """Monta os vetores x, y de TODOS os condutores para o czyl (feixe CIRCULAR).

    Mantida por compatibilidade. Para geometrias distintas por fase (feixes
    elípticos ou fase central diferente das laterais), use
    ``build_coordinates_per_phase``.

    Ordenação exigida por ``czyl_overhead_bundled``: subcondutores da fase A,
    depois B, depois C e, por fim, os para-raios. A altura média de cada
    condutor é  y = y_fixação - (2/3) * flecha.
    """
    offs = bundle_offsets(nb, spacing, angle0_deg)
    return build_coordinates_per_phase(
        phase_centers, [offs, offs, offs], phase_sag,
        gw_positions, gw_sag)


def build_coordinates_per_phase(phase_centers, phase_offsets, phase_sag,
                                gw_positions, gw_sag):
    """Monta x, y quando cada fase tem seu próprio conjunto de offsets.

    ``phase_offsets`` é uma lista com 3 arrays (nb, 2) — offsets de subcondutor
    das fases A, B e C, nessa ordem. Todos os arrays devem ter o mesmo nb
    (limitação de ``czyl_overhead_bundled``: um único nb global).
    """
    if len(phase_offsets) != len(phase_centers):
        raise ValueError("phase_offsets e phase_centers devem ter o mesmo comprimento.")
    nbs = {len(o) for o in phase_offsets}
    if len(nbs) != 1:
        raise ValueError(
            "Todas as fases devem ter o mesmo número de subcondutores nb "
            f"(recebido: {sorted(nbs)}).")
    xs, ys = [], []
    for (xc, yc), offs in zip(phase_centers, phase_offsets):
        y_med = yc - (2.0 / 3.0) * phase_sag
        for dx, dy in offs:
            xs.append(xc + dx)
            ys.append(y_med + dy)
    for (xg, yg) in gw_positions:
        y_med = yg - (2.0 / 3.0) * gw_sag
        xs.append(xg)
        ys.append(y_med)
    return np.asarray(xs, float), np.asarray(ys, float)


# ===========================================================================
# 4. CÁLCULO DOS PARÂMETROS UNITÁRIOS
# ===========================================================================
def line_parameters(omega, x, y, rho_solo, props, npr,
                     nb, rpr=GW_RADIUS_DEFAULT, rdcpr=GW_RDC_DEFAULT):
    """Calcula Z e Y por unidade de comprimento (matrizes 3x3).

    Encapsula a chamada a ``lcp.czyl_overhead_bundled`` com a mesma
    convenção do notebook. Z e Y são retornados em Ohm/m e S/m.
    """
    Z, Y = lcp.czyl_overhead_bundled(
        omega=omega,
        x=x,
        y=y,
        sigma_s=1.0 / rho_solo,
        rdc=props["rdc"],
        rf=props["rf"],
        rint=props["rint"],
        npr=npr,
        rdcpr=rdcpr,
        rpr=rpr,
        nb=nb,
    )
    return Z, Y


def fortescue_matrices():
    """Matriz de Fortescue A e sua inversa."""
    a = np.exp(2j * np.pi / 3.0)
    A = np.array([[1, 1, 1],
                  [1, a**2, a],
                  [1, a, a**2]], dtype=complex)
    return A, np.linalg.inv(A)


def sequence_analysis(Z, Y, Vn_kv):
    """Componentes de sequência, impedância característica e potência natural.

    Z, Y em por-metro. Retorna um dicionário com grandezas já em /km e
    a potência natural (SIL) em MW para a tensão nominal ``Vn_kv``.
    """
    A, A_inv = fortescue_matrices()
    Z_seq = A_inv @ Z @ A
    Y_seq = A_inv @ Y @ A
    z012 = np.diag(Z_seq)
    y012 = np.diag(Y_seq)

    zc = np.sqrt(z012[1] / y012[1])           # impedância característica (Ohm)
    Pn = Vn_kv**2 / np.real(zc)               # potência natural (MW)

    return {
        "Z_seq": Z_seq,
        "Y_seq": Y_seq,
        "z0": z012[0] * 1000.0,               # Ohm/km
        "z1": z012[1] * 1000.0,
        "z2": z012[2] * 1000.0,
        "y0": y012[0] * 1000.0,               # S/km
        "y1": y012[1] * 1000.0,
        "y2": y012[2] * 1000.0,
        "zc": zc,
        "Pn": Pn,
    }


def multi_circuit_sequence_analysis(Z, Y, Vn_kv, ncirc):
    """Análise de sequência para ``ncirc`` circuitos trifásicos em paralelo.

    ``Z`` e ``Y`` são as matrizes de fase completas (3·ncirc × 3·ncirc), em
    por-metro, na ordenação C1-A, C1-B, C1-C, C2-A, ... A transformação de
    Fortescue é aplicada em blocos:  T = I_ncirc ⊗ A,
    Ẑ = T⁻¹ Z T  e  Ŷ = T⁻¹ Y T.

    Para cada circuito ``i``, as impedâncias/admitâncias próprias de sequência
    são os elementos diagonais do bloco 3×3 correspondente de Ẑ (Ŷ).

    O acoplamento entre circuitos é resumido nas matrizes ncirc × ncirc
    ``M0`` e ``M1`` (impedância) e ``N0`` e ``N1`` (admitância), cujo elemento
    (i, j) é o termo de sequência zero (resp. positiva) que liga o circuito i
    ao circuito j.

    EQUIVALENTE PARALELO — circuitos ligados aos MESMOS barramentos e no
    mesmo nível de tensão: a queda de tensão de sequência é comum a todos e a
    corrente total é a soma das correntes de cada circuito. Assim,

        z_eq = 1 / (1ᵀ M⁻¹ 1)      (série)
        y_eq = 1ᵀ N 1              (shunt)

    com 1 = vetor de uns. Para ncirc = 1 recupera-se ``sequence_analysis``.

    Retorna um dicionário com grandezas por circuito (/km), matrizes de
    acoplamento (/km), equivalente paralelo, Z_c equivalente e SIL (MW).
    """
    Z = np.asarray(Z, dtype=complex)
    Y = np.asarray(Y, dtype=complex)
    if Z.shape != (3 * ncirc, 3 * ncirc):
        raise ValueError(
            f"Z deve ser {3*ncirc}x{3*ncirc} para {ncirc} circuito(s); "
            f"recebido {Z.shape}.")

    A, A_inv = fortescue_matrices()
    T = np.kron(np.eye(ncirc), A)
    T_inv = np.kron(np.eye(ncirc), A_inv)
    Z_seq = T_inv @ Z @ T
    Y_seq = T_inv @ Y @ T

    # --- grandezas próprias de cada circuito (blocos diagonais) ---
    per_circuit = []
    for i in range(ncirc):
        zb = np.diag(Z_seq[3 * i:3 * i + 3, 3 * i:3 * i + 3])
        yb = np.diag(Y_seq[3 * i:3 * i + 3, 3 * i:3 * i + 3])
        zc_i = np.sqrt(zb[1] / yb[1])
        per_circuit.append({
            "z0": zb[0] * 1000.0, "z1": zb[1] * 1000.0, "z2": zb[2] * 1000.0,
            "y0": yb[0] * 1000.0, "y1": yb[1] * 1000.0, "y2": yb[2] * 1000.0,
            "zc": zc_i,
            "Pn": Vn_kv**2 / np.real(zc_i),
        })

    # --- matrizes de acoplamento de sequência entre circuitos ---
    def _coupling(Mseq, k):
        return np.array([[Mseq[3 * i + k, 3 * j + k] for j in range(ncirc)]
                         for i in range(ncirc)], dtype=complex)

    M0 = _coupling(Z_seq, 0)      # sequência zero  (Ohm/m)
    M1 = _coupling(Z_seq, 1)      # sequência positiva
    N0 = _coupling(Y_seq, 0)      # (S/m)
    N1 = _coupling(Y_seq, 1)

    # --- equivalente dos circuitos em paralelo (mesmos barramentos) ---
    ones = np.ones(ncirc)
    z1_eq = 1.0 / (ones @ np.linalg.inv(M1) @ ones)
    z0_eq = 1.0 / (ones @ np.linalg.inv(M0) @ ones)
    y1_eq = ones @ N1 @ ones
    y0_eq = ones @ N0 @ ones

    zc_eq = np.sqrt(z1_eq / y1_eq)
    Pn_eq = Vn_kv**2 / np.real(zc_eq)

    return {
        "Z_seq": Z_seq,
        "Y_seq": Y_seq,
        "per_circuit": per_circuit,
        "M0": M0 * 1000.0,        # Ohm/km
        "M1": M1 * 1000.0,
        "N0": N0 * 1000.0,        # S/km
        "N1": N1 * 1000.0,
        "z0_eq": z0_eq * 1000.0,  # Ohm/km
        "z1_eq": z1_eq * 1000.0,
        "y0_eq": y0_eq * 1000.0,  # S/km
        "y1_eq": y1_eq * 1000.0,
        "zc": zc_eq,
        "Pn": Pn_eq,
    }
