"""
app.py  -  Frontend para cálculo de parâmetros unitários de linhas aéreas
=========================================================================

EEE 457 - Transmissão de Energia Elétrica
Escola Politécnica / COPPE - Universidade Federal do Rio de Janeiro

O aplicativo lê a tabela de condutores CAA (ACSR) em CSV, extrai as grandezas
necessárias (geometria, resistência e peso), calcula a flecha dos condutores
pela expressão da CATENÁRIA e monta as matrizes de impedância (Z) e admitância
(Y) por unidade de comprimento, seguindo a metodologia do notebook
eee457-06_calculo_param_unitarios.ipynb (rotina `czyl_overhead_bundled`).

Suporta de 1 a 3 CIRCUITOS TRIFÁSICOS EM PARALELO, no mesmo nível de tensão,
compartilhando o mesmo corredor (mesma torre ou torres lado a lado). Todos os
circuitos usam o mesmo condutor e o mesmo número de subcondutores por fase —
limitação da rotina `czyl_overhead_bundled` (um único rdc/rf/rint e um único
nb globais).

Execução:
    uv run streamlit run app.py
    -- ou --
    streamlit run app.py

Os arquivos `line_cable_param.py` e `condutores-caa.csv` devem estar na mesma
pasta deste script.
"""

from __future__ import annotations

import io
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

try:
    import lt_core as core
except ImportError as exc:  # pragma: no cover
    st.error(f"Falha ao importar o núcleo de cálculo (lt_core.py): {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Aparência
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Parâmetros unitários - LT aérea",
                   page_icon="\u26a1", layout="wide")

# Paleta Okabe-Ito (segura para daltonismo)
OKABE = {
    "preto":   "#000000", "laranja":  "#E69F00", "azul_ceu": "#56B4E9",
    "verde":   "#009E73", "amarelo":  "#F0E442", "azul":     "#0072B2",
    "vermelho":"#D55E00", "roxo":     "#CC79A7",
}
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PADRAO = os.path.join(DIR, "condutores-caa.csv")

FASES = ["A", "B", "C"]
CORES_FASE = [OKABE["azul"], OKABE["vermelho"], OKABE["verde"]]
MARCADOR_CIRC = ["o", "s", "D"]          # um marcador por circuito


# ---------------------------------------------------------------------------
# Geometrias padrão por número de circuitos
# ---------------------------------------------------------------------------
def geometria_padrao(ncirc: int):
    """Coordenadas de fixação padrão das fases e dos para-raios.

    - 1 circuito : configuração horizontal do notebook (345 kV).
    - 2 circuitos: torre de circuito duplo, fases em disposição vertical.
    - 3 circuitos: três torres lado a lado no mesmo corredor (25 m entre eixos).
    """
    if ncirc == 1:
        fases = [("C1", "A", -8.5, 28.4),
                 ("C1", "B",  0.0, 29.25),
                 ("C1", "C",  8.5, 28.4)]
        gw = [(-6.25, 35.9), (6.25, 35.9), (-15.0, 35.9), (15.0, 35.9)]
    elif ncirc == 2:
        fases = [("C1", "A", -6.0, 33.0),
                 ("C1", "B", -6.5, 26.5),
                 ("C1", "C", -6.0, 20.0),
                 ("C2", "A",  6.0, 33.0),
                 ("C2", "B",  6.5, 26.5),
                 ("C2", "C",  6.0, 20.0)]
        gw = [(-3.5, 38.5), (3.5, 38.5), (-8.0, 38.5), (8.0, 38.5)]
    else:
        fases = []
        for c, x0 in enumerate((-25.0, 0.0, 25.0)):
            fases += [(f"C{c+1}", "A", x0 - 8.5, 28.4),
                      (f"C{c+1}", "B", x0,       29.25),
                      (f"C{c+1}", "C", x0 + 8.5, 28.4)]
        gw = [(-6.25, 35.9), (6.25, 35.9), (-31.25, 35.9), (31.25, 35.9)]
    return fases, gw


# ---------------------------------------------------------------------------
# Funções auxiliares de apresentação
# ---------------------------------------------------------------------------
def fmt_complex(z: complex, casas: int = 5) -> str:
    """Formata um número complexo como 'a + bj'."""
    sinal = "+" if z.imag >= 0 else "-"
    return f"{z.real:.{casas}f} {sinal} {abs(z.imag):.{casas}f}j"


def matriz_complexa_df(M: np.ndarray, rotulos, casas: int = 5) -> pd.DataFrame:
    """Converte uma matriz complexa em DataFrame de strings formatadas."""
    dados = [[fmt_complex(M[i, j], casas) for j in range(M.shape[1])]
             for i in range(M.shape[0])]
    return pd.DataFrame(dados, index=rotulos, columns=rotulos)


def matriz_para_csv(M: np.ndarray) -> str:
    """Serializa matriz complexa em CSV (partes real e imaginária)."""
    buf = io.StringIO()
    n = M.shape[0]
    cab = []
    for j in range(n):
        cab += [f"Re_{j+1}", f"Im_{j+1}"]
    buf.write(",".join(cab) + "\n")
    for i in range(n):
        linha = []
        for j in range(n):
            linha += [f"{M[i, j].real:.10e}", f"{M[i, j].imag:.10e}"]
        buf.write(",".join(linha) + "\n")
    return buf.getvalue()


def rotulo_fase(k: int, ncirc: int) -> str:
    """Rótulo da fase global k (0..3·ncirc-1): 'A' ou 'C2-A'."""
    c, f = divmod(k, 3)
    return FASES[f] if ncirc == 1 else f"C{c+1}-{FASES[f]}"


def grafico_estrutura(x, y, centers, gw_positions, nb, npr, ncirc,
                      phase_offsets=None, draw_bundle_outline=True):
    """Desenha a silhueta da torre/corredor com os condutores nas alturas médias.

    Cores identificam a FASE (A azul, B vermelho, C verde) e marcadores
    identificam o CIRCUITO (C1 círculo, C2 quadrado, C3 losango).

    Se ``phase_offsets`` for informado (lista com 3·ncirc arrays (nb,2)),
    o contorno (círculo ou elipse) de cada feixe é traçado como guia visual —
    apenas quando ``draw_bundle_outline`` é True (para geometrias paramétricas).
    """
    largura = 6.2 if ncirc < 3 else 8.6
    fig, ax = plt.subplots(figsize=(largura, 5.0))
    nf = 3 * ncirc

    # subcondutores de fase (alturas médias)
    for k in range(nf):
        circ, fase = divmod(k, 3)
        ini, fim = k * nb, (k + 1) * nb
        ax.scatter(x[ini:fim], y[ini:fim], s=55, color=CORES_FASE[fase],
                   marker=MARCADOR_CIRC[circ], zorder=3,
                   label=rotulo_fase(k, ncirc))
        cx, cy = centers[k]
        # centro do feixe na altura MEDIA (não na fixação) para ficar sobre os pontos
        cy_med = np.mean(y[ini:fim]) if nb >= 1 else cy
        ax.scatter([cx], [cy_med], s=30, marker="x",
                   color=CORES_FASE[fase], alpha=0.6)
        ax.annotate(rotulo_fase(k, ncirc),
                    (np.mean(x[ini:fim]), np.mean(y[ini:fim])),
                    textcoords="offset points", xytext=(10, 8),
                    fontsize=9 if ncirc > 1 else 11, color=CORES_FASE[fase])
        # contorno do feixe
        if nb >= 2 and phase_offsets is not None and draw_bundle_outline:
            offs = np.asarray(phase_offsets[k])
            dx_max = np.max(np.abs(offs[:, 0]))
            dy_max = np.max(np.abs(offs[:, 1]))
            if dx_max > 1e-9 or dy_max > 1e-9:
                t = np.linspace(0, 2 * np.pi, 200)
                a = dx_max if dx_max > 1e-9 else dy_max
                b = dy_max if dy_max > 1e-9 else dx_max
                ax.plot(cx + a * np.cos(t), cy_med + b * np.sin(t),
                        color=CORES_FASE[fase], ls=":", lw=0.9,
                        alpha=0.6, zorder=1)

    # para-raios
    if npr > 0:
        xg = x[nf * nb:]
        yg = y[nf * nb:]
        ax.scatter(xg, yg, s=70, marker="^", color=OKABE["preto"],
                   zorder=3, label="Para-raios")
        for (gx, gy) in gw_positions:
            ax.scatter([gx], [gy], s=30, marker="x",
                       color=OKABE["preto"], alpha=0.5)

    # solo
    xlo = min(x.min(), -1) - 3
    xhi = max(x.max(), 1) + 3
    ax.axhline(0.0, color=OKABE["laranja"], lw=2.0)
    ax.fill_between([xlo, xhi], -3, 0, color=OKABE["laranja"], alpha=0.15)
    ax.text(xhi, 0.4, "solo", ha="right", fontsize=9, color=OKABE["laranja"])

    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-3, max(y.max(), 1) + 6)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("altura (m)")
    titulo = "Geometria da estrutura" if ncirc == 1 else \
        f"Geometria do corredor \u2014 {ncirc} circuitos em paralelo"
    ax.set_title(titulo + "  (x = fixação,  $\\bullet$ = altura média)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower center", ncol=3 if ncirc > 1 else 2,
              fontsize=7 if ncirc > 1 else 8, framealpha=0.9)
    fig.tight_layout()
    return fig


def grafico_catenaria(span, sag_fase, h_fase, sag_gw, h_gw, npr):
    """Desenha o perfil de catenária de um condutor de fase e de um para-raios."""
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    xf, hf = core.catenary_profile(span, sag_fase, h_fase)
    ax.plot(xf, hf, color=OKABE["azul"], lw=2.0, label="Condutor de fase")
    ax.scatter([-span / 2, span / 2], [h_fase, h_fase],
               color=OKABE["azul"], zorder=3)
    ax.annotate(f"flecha = {sag_fase:.2f} m",
                (0, h_fase - sag_fase), textcoords="offset points",
                xytext=(0, -16), ha="center", fontsize=9, color=OKABE["azul"])

    if npr > 0 and sag_gw > 0:
        xg, hg = core.catenary_profile(span, sag_gw, h_gw)
        ax.plot(xg, hg, color=OKABE["preto"], lw=1.8, ls="--",
                label="Para-raios")
        ax.scatter([-span / 2, span / 2], [h_gw, h_gw],
                   color=OKABE["preto"], zorder=3)

    ax.axhline(0.0, color=OKABE["laranja"], lw=2.0)
    ax.set_xlabel("posição ao longo do vão (m)")
    ax.set_ylabel("altura (m)")
    ax.set_title("Perfil de catenária")
    ax.set_ylim(-2, max(h_fase, h_gw) + 5)
    ax.legend(loc="lower center", fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# ===========================================================================
# CABEÇALHO
# ===========================================================================
st.title("\u26a1 Parâmetros unitários de linhas de transmissão aéreas")
st.caption(
    "EEE 457 - Transmissão de Energia Elétrica  |  Escola Politécnica / COPPE - UFRJ  "
    "\u2014  matrizes Z e Y por unidade de comprimento com flecha pela catenária  "
    "\u2014  até 3 circuitos em paralelo no mesmo nível de tensão"
)

# ===========================================================================
# BARRA LATERAL - ENTRADAS
# ===========================================================================
st.sidebar.header("1. Tabela de condutores")
arq = st.sidebar.file_uploader("Arquivo CSV de condutores CAA", type=["csv"])
if arq is not None:
    df_cond = core.load_conductor_table(arq)
    fonte = arq.name
elif os.path.exists(CSV_PADRAO):
    df_cond = core.load_conductor_table(CSV_PADRAO)
    fonte = "condutores-caa.csv (padrão)"
else:
    st.sidebar.error("Envie o arquivo condutores-caa.csv para continuar.")
    st.stop()
st.sidebar.caption(f"Fonte: {fonte}  \u2014  {len(df_cond)} condutores")

nomes = df_cond[core.COL["nome"]].astype(str).tolist()
nome_sel = st.sidebar.selectbox(
    "Condutor de fase", nomes,
    index=nomes.index("TERN") if "TERN" in nomes else 0)
linha_cond = df_cond[df_cond[core.COL["nome"]] == nome_sel].iloc[0]

st.sidebar.header("2. Circuitos em paralelo")
ncirc = st.sidebar.radio(
    "Número de circuitos trifásicos", [1, 2, 3], horizontal=True,
    help="Circuitos operando em PARALELO (mesmos barramentos) e no mesmo "
         "nível de tensão, compartilhando o corredor. Todos usam o mesmo "
         "condutor e o mesmo nb — limitação da rotina czyl_overhead_bundled.")
if ncirc > 1:
    st.sidebar.caption(
        f"Ordenação das fases: C1-A, C1-B, C1-C, ..., C{ncirc}-C. "
        "Para arranjos com transposição de barras (ex.: ABC/CBA no circuito "
        "duplo), basta editar as coordenadas na seção 6 de acordo.")

st.sidebar.header("3. Condições de operação")
freq = st.sidebar.number_input("Frequência (Hz)", 1.0, 1000.0, 60.0, 1.0)
rho_solo = st.sidebar.number_input(
    "Resistividade do solo (\u03a9\u00b7m)", 1.0, 1.0e5, 1000.0, 50.0)
temp_op = st.sidebar.number_input(
    "Temperatura do condutor (\u00b0C)", -20.0, 150.0, 20.0, 5.0,
    help="Corrige a resistência CC a partir do valor tabelado a 20 °C. "
         "O efeito pelicular é incluído pelo modelo de impedância interna.")
Vn = st.sidebar.number_input("Tensão nominal (kV)", 1.0, 1500.0, 345.0, 1.0,
                             help="Comum a todos os circuitos em paralelo.")

st.sidebar.header("4. Vão e tração (catenária)")
span = st.sidebar.number_input("Vão entre estruturas (m)", 50.0, 2000.0, 400.0, 10.0)
modo_flecha = st.sidebar.radio(
    "Flecha dos condutores de fase",
    ["Catenária (vão + tração)", "Flecha direta (m)"])
if modo_flecha.startswith("Catenária"):
    frac_rts = st.sidebar.slider(
        "Tração horizontal (% da carga de ruptura)", 5.0, 50.0, 20.0, 0.5,
        help="Tração de trabalho (EDS). Valores típicos: 18 % a 25 % da RTS.")
    flecha_direta = None
else:
    frac_rts = None
    flecha_direta = st.sidebar.number_input(
        "Flecha de fase (m)", 0.5, 60.0, 19.1, 0.1)

st.sidebar.header("5. Feixe de subcondutores")
nb = st.sidebar.number_input(
    "Subcondutores por fase (nb)", 1, 8, 2, 1,
    help="Mesmo nb em todas as fases de todos os circuitos.")

fases_padrao_lista, gw_padrao_lista = geometria_padrao(ncirc)

# variaveis de saida (offsets por fase, alem de descritores para o grafico)
tipo_feixe = "circular"
espac = 0.0
ang0 = 0.0
# semi-eixos (a, b) por grupo de fase
a_lat = b_lat = a_cent = b_cent = 0.0
tab_manual = None

if nb > 1:
    tipo_feixe = st.sidebar.radio(
        "Geometria do feixe",
        ["Circular (mesma em todas as fases)",
         "Elíptico (fase central distinta das laterais)",
         "Manual (coordenadas de cada subcondutor)"],
        help="Circular: polígono regular com espaçamento uniforme. "
             "Elíptico: subcondutores sobre uma elipse de semi-eixos (a, b), "
             "com a fase central (2ª de cada circuito) podendo diferir das "
             "laterais (1ª e 3ª). "
             "Manual: você informa diretamente as coordenadas absolutas de "
             "cada subcondutor de cada circuito.")

    if tipo_feixe.startswith("Circular"):
        ang0 = st.sidebar.number_input(
            "Ângulo de orientação do feixe (\u00b0)", 0.0, 360.0, 0.0, 15.0)
        espac = st.sidebar.number_input(
            "Espaçamento entre subcondutores adjacentes (m)",
            0.05, 2.0, 0.4572, 0.0001, format="%.4f")
    elif tipo_feixe.startswith("Elíptico"):
        ang0 = st.sidebar.number_input(
            "Ângulo de orientação do feixe (\u00b0)", 0.0, 360.0, 0.0, 15.0)
        st.sidebar.markdown(
            "**Semi-eixos das fases laterais (1ª e 3ª de cada circuito)**")
        a_lat = st.sidebar.number_input(
            "a\u2097 - horizontal (m)", 0.01, 2.0, 0.2286, 0.001, format="%.4f",
            key="a_lat")
        b_lat = st.sidebar.number_input(
            "b\u2097 - vertical (m)", 0.01, 2.0, 0.2286, 0.001, format="%.4f",
            key="b_lat")
        st.sidebar.markdown(
            "**Semi-eixos da fase central (2ª de cada circuito)**")
        a_cent = st.sidebar.number_input(
            "a\u1D9C - horizontal (m)", 0.01, 2.0, 0.3000, 0.001, format="%.4f",
            key="a_cent")
        b_cent = st.sidebar.number_input(
            "b\u1D9C - vertical (m)", 0.01, 2.0, 0.1500, 0.001, format="%.4f",
            key="b_cent")
        st.sidebar.caption(
            "Feixe circular como caso particular: informe a = b "
            "(igual ao raio do polígono usual).")
    else:  # Manual
        st.sidebar.caption(
            "Informe as coordenadas **(x, y)** de cada subcondutor no **ponto "
            "de fixação** da torre. A altura média usada no cálculo é obtida "
            "de  y_média = y_fixação \u2212 (2/3)·flecha. Neste modo a seção 6 "
            "é ignorada.")

        # Ponto de partida: geometria padrão + feixe circular (s = 0.4572 m)
        offs_default = core.bundle_offsets(nb, 0.4572, 0.0)
        linhas0 = []
        for (circ, fase, xc, yc) in fases_padrao_lista:
            for k in range(nb):
                linhas0.append({
                    "Circuito": circ,
                    "Fase": fase,
                    "Sub": f"sub {k+1}",
                    "x (m)": round(xc + offs_default[k, 0], 4),
                    "y fixação (m)": round(yc + offs_default[k, 1], 4),
                })
        tab_manual = st.sidebar.data_editor(
            pd.DataFrame(linhas0), hide_index=True,
            disabled=["Circuito", "Fase", "Sub"],
            key=f"manual_n{ncirc}_nb{nb}", num_rows="fixed")

modo_manual = tipo_feixe.startswith("Manual")

fases_padrao = pd.DataFrame(
    fases_padrao_lista,
    columns=["Circuito", "Fase", "x (m)", "y fixação (m)"])
if not modo_manual:
    st.sidebar.header("6. Geometria das fases")
    st.sidebar.caption(
        "Coordenadas do CENTRO de cada feixe (ponto de fixação), "
        "por circuito e fase.")
    fases_edit = st.sidebar.data_editor(
        fases_padrao, hide_index=True, disabled=["Circuito", "Fase"],
        key=f"fases_n{ncirc}")
else:
    # No modo Manual, a seção 6 é ignorada — usamos o default só para não quebrar
    # eventuais referências.
    fases_edit = fases_padrao

st.sidebar.header("7. Cabos para-raios")
npr = st.sidebar.selectbox("Número de para-raios", [0, 1, 2, 3, 4], index=2)
if npr > 0:
    st.sidebar.caption("Considerados 3/8\" EHS.")
    gw_padrao_full = pd.DataFrame({
        "Para-raios": [f"PR-{j+1}" for j in range(len(gw_padrao_lista))],
        "x (m)": [g[0] for g in gw_padrao_lista],
        "y fixação (m)": [g[1] for g in gw_padrao_lista],
    })
    gw_edit = st.sidebar.data_editor(
        gw_padrao_full.iloc[:npr].reset_index(drop=True),
        hide_index=True, disabled=["Para-raios"], key=f"gw{npr}_n{ncirc}")
    rpr = st.sidebar.number_input(
        "Raio do para-raios (m)", 1e-3, 2e-2, core.GW_RADIUS_DEFAULT,
        1e-4, format="%.5f")
    rdcpr = st.sidebar.number_input(
        "Resistência CC do para-raios (\u03a9/m)", 1e-4, 5e-2,
        core.GW_RDC_DEFAULT, 1e-4, format="%.5f")
    modo_flecha_gw = st.sidebar.radio(
        "Flecha dos para-raios",
        ["Catenária (tração + peso)", "Flecha direta (m)"])
    if modo_flecha_gw.startswith("Catenária"):
        Tpr = st.sidebar.number_input(
            "Carga de ruptura do para-raios  T_pr (kgf)",
            500.0, 50000.0, core.GW_RTS_DEFAULT, 10.0)
        Wpr = st.sidebar.number_input(
            "Peso do para-raios  W_pr (kg/km)",
            50.0, 3000.0, core.GW_WEIGHT_DEFAULT, 5.0)
        frac_rts_gw = st.sidebar.slider(
            "Tração horizontal do para-raios (% da RTS)",
            5.0, 50.0, core.GW_TENSION_FRAC_DEFAULT, 0.5,
            help="H_pr = (fração) \u00b7 T_pr.  Valor usual: 25 % da RTS.")
        flecha_gw_direta = None
    else:
        Tpr, Wpr, frac_rts_gw = (core.GW_RTS_DEFAULT, core.GW_WEIGHT_DEFAULT,
                                 core.GW_TENSION_FRAC_DEFAULT)
        flecha_gw_direta = st.sidebar.number_input(
            "Flecha dos para-raios (m)", 0.5, 60.0, 8.887, 0.1)
else:
    gw_edit = pd.DataFrame(columns=["Para-raios", "x (m)", "y fixação (m)"])
    rpr, rdcpr = core.GW_RADIUS_DEFAULT, core.GW_RDC_DEFAULT
    Tpr, Wpr, frac_rts_gw = (core.GW_RTS_DEFAULT, core.GW_WEIGHT_DEFAULT,
                             core.GW_TENSION_FRAC_DEFAULT)
    flecha_gw_direta = None
    modo_flecha_gw = ""

# ===========================================================================
# CÁLCULO
# ===========================================================================
props = core.conductor_props(linha_cond, temp_c=temp_op)
omega = 2.0 * np.pi * freq
nf = 3 * ncirc                      # número total de fases

# --- flecha dos condutores de fase (catenária) ---
if flecha_direta is not None:
    flecha_fase = flecha_direta
    a_fase = core.catenary_param_from_sag(span, flecha_fase)
    H_fase = a_fase * props["peso"]
    T_fase = H_fase * np.cosh(span / (2.0 * a_fase))
else:
    H_fase = (frac_rts / 100.0) * props["rts"]
    flecha_fase, a_fase, T_fase = core.sag_from_tension(span, props["peso"], H_fase)

# --- flecha dos para-raios (catenária: modelo tração + peso) ---
gw_info = None
if npr > 0:
    if flecha_gw_direta is not None:
        flecha_gw = flecha_gw_direta
        a_gw = core.catenary_param_from_sag(span, flecha_gw)
    else:
        flecha_gw, a_gw, H_gw, T_gw = core.ground_wire_sag(
            span, rts_kgf=Tpr, weight_kg_per_km=Wpr, tension_frac=frac_rts_gw)
        gw_info = {"a": a_gw, "H": H_gw, "T": T_gw}
else:
    flecha_gw = 0.0

# --- coordenadas de todos os condutores ---
gw_positions = [(float(r["x (m)"]), float(r["y fixação (m)"]))
                for _, r in gw_edit.iterrows()] if npr > 0 else []

if modo_manual:
    # Coordenadas absolutas de cada subcondutor (fixação); flecha aplicada aqui.
    centers = []
    phase_offsets = []
    for (circ, fase, _, _) in fases_padrao_lista:
        sel = tab_manual[(tab_manual["Circuito"] == circ)
                         & (tab_manual["Fase"] == fase)]
        fix = sel[["x (m)", "y fixação (m)"]].to_numpy(dtype=float)
        centro = fix.mean(axis=0)          # centroide (apenas para o gráfico)
        centers.append(tuple(centro))
        phase_offsets.append(fix - centro)

    x, y = core.build_coordinates_per_phase(
        centers, phase_offsets, flecha_fase, gw_positions, flecha_gw)
else:
    centers = [(float(r["x (m)"]), float(r["y fixação (m)"]))
               for _, r in fases_edit.iterrows()]

    # offsets de subcondutor por fase, na ordem C1-A, C1-B, C1-C, C2-A, ...
    if nb == 1:
        off_all = np.zeros((1, 2))
        phase_offsets = [off_all] * nf
    elif tipo_feixe.startswith("Circular"):
        off_all = core.bundle_offsets(nb, espac, ang0)
        phase_offsets = [off_all] * nf
    else:  # Elíptico: laterais = 1ª e 3ª fase de cada circuito, central = 2ª
        off_lat_arr = core.elliptical_bundle_offsets(nb, a_lat, b_lat, ang0)
        off_cent_arr = core.elliptical_bundle_offsets(nb, a_cent, b_cent, ang0)
        phase_offsets = [off_lat_arr, off_cent_arr, off_lat_arr] * ncirc

    x, y = core.build_coordinates_per_phase(
        centers, phase_offsets, flecha_fase, gw_positions, flecha_gw)

# --- validação: condutores coincidentes ou sobrepostos ---
erro_calc = None
if len(x) > 1:
    pts = np.column_stack([x, y])
    dist = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(dist, np.inf)
    if dist.min() < 2.0 * props["rf"]:
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        erro_calc = (
            f"Condutores sobrepostos ou coincidentes (separação mínima "
            f"{dist.min():.4f} m entre os condutores {i+1} e {j+1}). "
            "Revise as coordenadas — com múltiplos circuitos, verifique se as "
            "posições de circuitos diferentes não colidem.")

# --- matrizes Z e Y ---
if erro_calc is None:
    try:
        Z, Y = core.line_parameters(omega, x, y, rho_solo, props, npr=npr,
                                    nb=nb, rpr=rpr, rdcpr=rdcpr)
        seq = core.multi_circuit_sequence_analysis(Z, Y, Vn, ncirc)
    except Exception as exc:  # pragma: no cover
        erro_calc = str(exc)

# ===========================================================================
# PAINEL PRINCIPAL - RESULTADOS
# ===========================================================================
col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("Condutor selecionado")
    desc_circ = "" if ncirc == 1 else f"  \u2014  {ncirc} circuitos em paralelo"
    st.markdown(f"**{props['nome']}**  \u2014  feixe de {nb} "
                f"subcondutor(es){desc_circ}")
    dados_cond = pd.DataFrame({
        "Grandeza": [
            "Raio externo  r_f",
            "Raio interno (alma de aço)  r_int",
            "Resist. CC a 20 \u00b0C",
            f"Resist. CC a {temp_op:.0f} \u00b0C (usada)",
            "Resist. CA 60 Hz / 75 \u00b0C (tabela)",
            "Peso linear",
            "Carga de ruptura (classe A)",
            "Ampacidade",
        ],
        "Valor": [
            f"{props['rf']*1e3:.3f} mm",
            f"{props['rint']*1e3:.3f} mm",
            f"{props['rdc20']*1e3:.5f} \u03a9/km",
            f"{props['rdc']*1e3:.5f} \u03a9/km",
            f"{props['rac75']*1e3:.5f} \u03a9/km",
            f"{props['peso']:.3f} N/m  ({props['massa']*1000:.1f} kg/km)",
            f"{props['rts']/core.KGF_TO_N:.0f} kgf  ({props['rts']/1e3:.1f} kN)",
            f"{props['ampacidade']:.0f} A",
        ],
    })
    st.dataframe(dados_cond, hide_index=True, width='stretch')

with col_b:
    st.subheader("Flecha pela catenária")
    linhas_flecha = [
        ("Vão", f"{span:.1f} m"),
        ("Flecha dos condutores de fase", f"{flecha_fase:.3f} m"),
        ("Parâmetro da catenária  a = H/w", f"{a_fase:.1f} m"),
        ("Tração horizontal  H", f"{H_fase/1e3:.2f} kN "
            f"({H_fase/props['rts']*100:.1f} % da RTS)"),
        ("Tração máxima  T (na fixação)", f"{T_fase/1e3:.2f} kN"),
    ]
    if npr > 0:
        linhas_flecha.append(
            ("Flecha dos para-raios (3/8\" EHS)", f"{flecha_gw:.3f} m"))
        if gw_info is not None:
            linhas_flecha.append(
                ("Parâmetro da catenária do para-raios  C_pr",
                 f"{gw_info['a']:.1f} m"))
            linhas_flecha.append(
                ("Tração horizontal do para-raios  H_pr",
                 f"{gw_info['H']/core.KGF_TO_N:.0f} kgf "
                 f"({gw_info['H']/1e3:.2f} kN)"))
    st.dataframe(
        pd.DataFrame(linhas_flecha, columns=["Grandeza", "Valor"]),
        hide_index=True, width='stretch')
    st.caption("Altura média de cada condutor = altura de fixação "
               "\u2212 (2/3)\u00b7flecha.  Para-raios: C_pr = H_pr / W_pr, "
               "com H_pr = (\u0025 RTS)\u00b7T_pr.  A mesma flecha de fase é "
               "aplicada a todos os circuitos (mesmo condutor e mesma tração).")

st.divider()

if erro_calc is not None:
    st.error(f"Erro no cálculo das matrizes: {erro_calc}")
    st.stop()

# --- gráficos ---
g1, g2 = st.columns([1, 1])
with g1:
    st.pyplot(grafico_estrutura(x, y, centers, gw_positions, nb, npr, ncirc,
                                phase_offsets=phase_offsets,
                                draw_bundle_outline=not modo_manual))
with g2:
    h_fase_med = np.mean([c[1] for c in centers])
    h_gw_med = np.mean([g[1] for g in gw_positions]) if npr > 0 else 0.0
    st.pyplot(grafico_catenaria(span, flecha_fase, h_fase_med,
                                flecha_gw, h_gw_med, npr))

st.divider()

# --- matrizes Z e Y ---
st.subheader("Matrizes de parâmetros unitários (após reduções de Kron e de feixe)")
rot = [rotulo_fase(k, ncirc) for k in range(nf)]
if ncirc > 1:
    st.caption(
        f"Matrizes {nf}\u00d7{nf} de fase: blocos 3\u00d73 diagonais são os "
        "parâmetros próprios de cada circuito; blocos fora da diagonal são o "
        "acoplamento eletromagnético entre circuitos do mesmo corredor.")

st.markdown("**Matriz de impedância série  Z  (\u03a9/km)**")
st.dataframe(matriz_complexa_df(Z * 1000.0, rot, 5), width='stretch')

st.markdown("**Matriz de admitância shunt  Y  (\u03bcS/km)**")
st.dataframe(matriz_complexa_df(Y * 1000.0 * 1e6, rot, 4), width='stretch')

d1, d2 = st.columns(2)
d1.download_button("Baixar Z (\u03a9/m) em CSV",
                   matriz_para_csv(Z), "matriz_Z_ohm_por_m.csv", "text/csv")
d2.download_button("Baixar Y (S/m) em CSV",
                   matriz_para_csv(Y), "matriz_Y_S_por_m.csv", "text/csv")

st.divider()

# --- componentes de sequência ---
st.subheader("Componentes de sequência")
st.caption("Transformação de Fortescue aplicada em blocos "
           "(T = I ⊗ A) sobre as matrizes de fase completas."
           if ncirc > 1 else
           "Transformação de Fortescue sobre as matrizes de fase.")

# tabela por circuito
per = seq["per_circuit"]
sc1, sc2 = st.columns(2)
with sc1:
    st.markdown("**Impedâncias de sequência próprias  Z\u2080\u2081\u2082  (\u03a9/km)**")
    st.dataframe(pd.DataFrame({
        "Circuito": [f"C{i+1}" for i in range(ncirc)],
        "Z\u2080 (\u03a9/km)": [fmt_complex(p["z0"]) for p in per],
        "Z\u2081 (\u03a9/km)": [fmt_complex(p["z1"]) for p in per],
        "Z\u2082 (\u03a9/km)": [fmt_complex(p["z2"]) for p in per],
    }), hide_index=True, width='stretch')
with sc2:
    st.markdown("**Admitâncias de sequência próprias  Y\u2080\u2081\u2082  (\u03bcS/km)**")
    st.dataframe(pd.DataFrame({
        "Circuito": [f"C{i+1}" for i in range(ncirc)],
        "Y\u2080 (\u03bcS/km)": [fmt_complex(p["y0"] * 1e6, 4) for p in per],
        "Y\u2081 (\u03bcS/km)": [fmt_complex(p["y1"] * 1e6, 4) for p in per],
        "Y\u2082 (\u03bcS/km)": [fmt_complex(p["y2"] * 1e6, 4) for p in per],
    }), hide_index=True, width='stretch')

# acoplamento entre circuitos
if ncirc > 1:
    rot_c = [f"C{i+1}" for i in range(ncirc)]
    st.markdown("**Acoplamento de sequência entre circuitos**")
    ac1, ac2 = st.columns(2)
    with ac1:
        st.markdown("Sequência ZERO — matriz M\u2080 (\u03a9/km)")
        st.dataframe(matriz_complexa_df(seq["M0"], rot_c, 5), width='stretch')
    with ac2:
        st.markdown("Sequência POSITIVA — matriz M\u2081 (\u03a9/km)")
        st.dataframe(matriz_complexa_df(seq["M1"], rot_c, 5), width='stretch')
    st.caption(
        "O elemento (i, j) é o termo de sequência que liga o circuito i ao "
        "circuito j. O acoplamento de sequência zero entre circuitos do mesmo "
        "corredor é tipicamente significativo (retorno comum pelo solo); o de "
        "sequência positiva é pequeno, mas não nulo, quando as fases não são "
        "transpostas.")

# destaque: equivalente dos circuitos em paralelo
titulo_eq = ("Sequência positiva \u2014 destaque"
             if ncirc == 1 else
             f"Equivalente dos {ncirc} circuitos em PARALELO \u2014 destaque")
st.markdown(f"**{titulo_eq}**")
if ncirc > 1:
    st.caption(
        "Circuitos ligados aos mesmos barramentos:  "
        "z_eq = 1/(1\u1d40 M\u2081\u207b\u00b9 1)  (série)  e  "
        "y_eq = 1\u1d40 N\u2081 1  (shunt), incluindo o acoplamento mútuo "
        "entre circuitos.")

# linha 1: Z1 equivalente
z1 = seq["z1_eq"]
mod_z1 = abs(z1)
ang_z1 = np.degrees(np.angle(z1))
r1, r2, r3, r4 = st.columns(4)
r1.metric("R\u2081  (\u03a9/km)", f"{z1.real:.5f}")
r2.metric("X\u2081  (\u03a9/km)", f"{z1.imag:.5f}")
r3.metric("|Z\u2081|  (\u03a9/km)", f"{mod_z1:.5f}")
r4.metric("\u2220 Z\u2081  (\u00b0)", f"{ang_z1:.2f}")

# linha 2: Y1 equivalente (em uS/km, como no notebook)
y1_us = seq["y1_eq"] * 1e6
mod_y1 = abs(y1_us)
ang_y1 = np.degrees(np.angle(y1_us))
q1, q2, q3, q4 = st.columns(4)
q1.metric("G\u2081  (\u03bcS/km)", f"{y1_us.real:.4f}")
q2.metric("B\u2081  (\u03bcS/km)", f"{y1_us.imag:.4f}")
q3.metric("|Y\u2081|  (\u03bcS/km)", f"{mod_y1:.4f}")
q4.metric("\u2220 Y\u2081  (\u00b0)", f"{ang_y1:.2f}")

# linha 3: desempenho do conjunto (Zc, Pn, indutância, velocidade)
omega_rad = 2.0 * np.pi * freq
L_km = z1.imag / omega_rad              # H/km (equivalente)
C_km = seq["y1_eq"].imag / omega_rad    # F/km (equivalente)
v_prop = 1.0 / np.sqrt(max(L_km * C_km, 1e-30)) if L_km > 0 and C_km > 0 else 0.0
n1, n2, n3, n4 = st.columns(4)
n1.metric("Z\u2080  carac. (\u03a9)", f"{np.real(seq['zc']):.2f}")
n2.metric(f"P\u2099 @ {Vn:.0f} kV (MW)", f"{seq['Pn']:.1f}")
n3.metric("L\u2081 (mH/km)", f"{L_km*1e3:.4f}")
n4.metric("v de propagação (km/s)", f"{v_prop:.0f}")

if ncirc > 1:
    st.caption(
        "Potência natural por circuito (isolado): " +
        ",  ".join(f"C{i+1}: {p['Pn']:.1f} MW" for i, p in enumerate(per)) +
        f".  Soma: {sum(p['Pn'] for p in per):.1f} MW — próxima, mas não "
        "idêntica, ao P\u2099 equivalente, por causa do acoplamento mútuo.")

st.caption(
    "Z\u2081 e Y\u2081 acima referem-se ao " +
    ("circuito único. " if ncirc == 1 else "equivalente paralelo. ") +
    "Z\u2080 = \u221a(z\u2081/y\u2081), P\u2099 = V\u2099\u00b2/Re(Z\u2080). "
    "A capacitância e a indutância unitárias, bem como a velocidade de "
    "propagação  v = 1/\u221a(L\u2081C\u2081),  são deduzidas de Im(z\u2081)/\u03c9 "
    "e Im(y\u2081)/\u03c9."
)

with st.expander("Detalhes \u2014 matrizes de sequência completas"):
    rot_seq_full = [f"C{k//3+1}-{s}" if ncirc > 1 else s
                    for k in range(nf) for s in ([str(k % 3)])]
    st.markdown("**Z de sequência (\u03a9/km)**")
    st.dataframe(matriz_complexa_df(seq["Z_seq"] * 1000.0, rot_seq_full, 5),
                 width='stretch')
    st.markdown("**Y de sequência (\u03bcS/km)**")
    st.dataframe(matriz_complexa_df(seq["Y_seq"] * 1000.0 * 1e6,
                                    rot_seq_full, 4),
                 width='stretch')
    if ncirc > 1:
        st.caption("Ordenação: C1-0, C1-1, C1-2, C2-0, ... "
                   "(0 = zero, 1 = positiva, 2 = negativa).")

with st.expander("Coordenadas de todos os condutores (entrada do czyl)"):
    n_sub = nf * nb
    rotulos = ([f"{rotulo_fase(k // nb, ncirc)} - sub {k % nb + 1}"
                for k in range(n_sub)]
               + [f"Para-raios {j+1}" for j in range(npr)])
    st.dataframe(pd.DataFrame({
        "Condutor": rotulos,
        "x (m)": np.round(x, 4),
        "y altura média (m)": np.round(y, 4),
    }), hide_index=True, width='stretch')

st.divider()
st.caption(
    "Metodologia: impedância externa por Carson, impedância interna de condutor "
    "tubular (funções de Bessel), redução de Kron para eliminar os para-raios e "
    "redução de feixe \u2014 rotina `czyl_overhead_bundled` de line_cable_param.py. "
    "Com múltiplos circuitos, a mesma rotina é aplicada ao conjunto completo de "
    "condutores do corredor (3\u00b7n circuitos + para-raios), capturando "
    "naturalmente o acoplamento eletromagnético entre circuitos."
)
