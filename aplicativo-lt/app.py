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


def grafico_estrutura(x, y, centers, gw_positions, nb, npr):
    """Desenha a silhueta da torre com os condutores nas alturas médias."""
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    nf = 3
    rot_fase = ["A", "B", "C"]
    cores = [OKABE["azul"], OKABE["vermelho"], OKABE["verde"]]

    # subcondutores de fase (alturas médias)
    for k in range(nf):
        ini, fim = k * nb, (k + 1) * nb
        ax.scatter(x[ini:fim], y[ini:fim], s=55, color=cores[k],
                   zorder=3, label=f"Fase {rot_fase[k]}")
        cx, cy = centers[k]
        ax.scatter([cx], [cy], s=30, marker="x", color=cores[k], alpha=0.6)
        ax.annotate(rot_fase[k], (np.mean(x[ini:fim]), np.mean(y[ini:fim])),
                    textcoords="offset points", xytext=(8, 6),
                    fontsize=11, color=cores[k])

    # para-raios
    if npr > 0:
        xg = x[nf * nb:]
        yg = y[nf * nb:]
        ax.scatter(xg, yg, s=70, marker="^", color=OKABE["preto"],
                   zorder=3, label="Para-raios")
        for j, (gx, gy) in enumerate(gw_positions):
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
    ax.set_title("Geometria da estrutura (x = fixação,  $\\bullet$ = altura média)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower center", ncol=2, fontsize=8, framealpha=0.9)
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
    "\u2014  matrizes Z e Y por unidade de comprimento com flecha pela catenária"
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

st.sidebar.header("2. Condições de operação")
freq = st.sidebar.number_input("Frequência (Hz)", 1.0, 1000.0, 60.0, 1.0)
rho_solo = st.sidebar.number_input(
    "Resistividade do solo (\u03a9\u00b7m)", 1.0, 1.0e5, 1000.0, 50.0)
temp_op = st.sidebar.number_input(
    "Temperatura do condutor (\u00b0C)", -20.0, 150.0, 20.0, 5.0,
    help="Corrige a resistência CC a partir do valor tabelado a 20 °C. "
         "O efeito pelicular é incluído pelo modelo de impedância interna.")
Vn = st.sidebar.number_input("Tensão nominal (kV)", 1.0, 1500.0, 345.0, 1.0)

st.sidebar.header("3. Vão e tração (catenária)")
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

st.sidebar.header("4. Feixe de condutores")
nb = st.sidebar.number_input("Subcondutores por fase (nb)", 1, 8, 2, 1)
if nb > 1:
    espac = st.sidebar.number_input(
        "Espaçamento entre subcondutores (m)", 0.05, 2.0, 0.4572, 0.0001,
        format="%.4f")
    ang0 = st.sidebar.number_input(
        "Ângulo de orientação do feixe (\u00b0)", 0.0, 360.0, 0.0, 15.0)
else:
    espac, ang0 = 0.0, 0.0

st.sidebar.header("5. Geometria das fases")
st.sidebar.caption("Coordenadas do CENTRO de cada feixe (ponto de fixação).")
fases_padrao = pd.DataFrame({
    "Fase": ["A", "B", "C"],
    "x (m)": [-8.5, 0.0, 8.5],
    "y fixação (m)": [28.4, 29.25, 28.4],
})
fases_edit = st.sidebar.data_editor(
    fases_padrao, hide_index=True, disabled=["Fase"], key="fases")

st.sidebar.header("6. Cabos para-raios")
npr = st.sidebar.selectbox("Número de para-raios", [0, 1, 2], index=2)
if npr > 0:
    st.sidebar.caption("Considerados 3/8\" EHS.")
    gw_padrao_full = pd.DataFrame({
        "Para-raios": ["PR-1", "PR-2"],
        "x (m)": [-6.25, 6.25],
        "y fixação (m)": [35.9, 35.9],
    })
    gw_edit = st.sidebar.data_editor(
        gw_padrao_full.iloc[:npr].reset_index(drop=True),
        hide_index=True, disabled=["Para-raios"], key=f"gw{npr}")
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
centers = [(float(r["x (m)"]), float(r["y fixação (m)"]))
           for _, r in fases_edit.iterrows()]
gw_positions = [(float(r["x (m)"]), float(r["y fixação (m)"]))
                for _, r in gw_edit.iterrows()] if npr > 0 else []

x, y = core.build_coordinates(centers, flecha_fase, gw_positions, flecha_gw,
                              nb=nb, spacing=espac, angle0_deg=ang0)

# --- matrizes Z e Y ---
erro_calc = None
try:
    Z, Y = core.line_parameters(omega, x, y, rho_solo, props, npr=npr,
                                nb=nb, rpr=rpr, rdcpr=rdcpr)
    seq = core.sequence_analysis(Z, Y, Vn)
except Exception as exc:  # pragma: no cover
    erro_calc = str(exc)

# ===========================================================================
# PAINEL PRINCIPAL - RESULTADOS
# ===========================================================================
col_a, col_b = st.columns([1, 1])

with col_a:
    st.subheader("Condutor selecionado")
    st.markdown(f"**{props['nome']}**  \u2014  feixe de {nb} subcondutor(es)")
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
               "com H_pr = (\u0025 RTS)\u00b7T_pr.")

st.divider()

if erro_calc is not None:
    st.error(f"Erro no cálculo das matrizes: {erro_calc}")
    st.stop()

# --- gráficos ---
g1, g2 = st.columns([1, 1])
with g1:
    st.pyplot(grafico_estrutura(x, y, centers, gw_positions, nb, npr))
with g2:
    h_fase_med = np.mean([c[1] for c in centers])
    h_gw_med = np.mean([g[1] for g in gw_positions]) if npr > 0 else 0.0
    st.pyplot(grafico_catenaria(span, flecha_fase, h_fase_med,
                                flecha_gw, h_gw_med, npr))

st.divider()

# --- matrizes Z e Y ---
st.subheader("Matrizes de parâmetros unitários (após reduções de Kron e de feixe)")
rot = ["A", "B", "C"]

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
st.subheader("Componentes de sequência e desempenho do circuito")
rot_seq = ["0 (zero)", "1 (positiva)", "2 (negativa)"]

sc1, sc2 = st.columns(2)
with sc1:
    st.markdown("**Impedâncias de sequência  Z\u2080\u2081\u2082  (\u03a9/km)**")
    st.dataframe(pd.DataFrame(
        {"Sequência": rot_seq,
         "Z (\u03a9/km)": [fmt_complex(seq["z0"]), fmt_complex(seq["z1"]),
                           fmt_complex(seq["z2"])]},
    ), hide_index=True, width='stretch')
with sc2:
    st.markdown("**Admitâncias de sequência  Y\u2080\u2081\u2082  (\u03bcS/km)**")
    st.dataframe(pd.DataFrame(
        {"Sequência": rot_seq,
         "Y (\u03bcS/km)": [fmt_complex(seq["y0"] * 1e6, 4),
                            fmt_complex(seq["y1"] * 1e6, 4),
                            fmt_complex(seq["y2"] * 1e6, 4)]},
    ), hide_index=True, width='stretch')

m1, m2, m3, m4 = st.columns(4)
m1.metric("R\u2081  (\u03a9/km)", f"{seq['z1'].real:.5f}")
m2.metric("X\u2081  (\u03a9/km)", f"{seq['z1'].imag:.5f}")
m3.metric("Z\u2080  carac. (\u03a9)", f"{np.real(seq['zc']):.2f}")
m4.metric(f"Pot. natural @ {Vn:.0f} kV (MW)", f"{seq['Pn']:.1f}")

st.caption(
    "Impedância característica  Z\u2080 = \u221a(z\u2081/y\u2081)  e potência "
    "natural (SIL)  P\u2099 = V\u2099\u00b2 / Re(Z\u2080), ambas para a sequência "
    "positiva, conforme o notebook de referência."
)

with st.expander("Detalhes \u2014 matrizes de sequência completas"):
    st.markdown("**Z de sequência (\u03a9/km)**")
    st.dataframe(matriz_complexa_df(seq["Z_seq"] * 1000.0,
                                    ["0", "1", "2"], 5),
                 width='stretch')
    st.markdown("**Y de sequência (\u03bcS/km)**")
    st.dataframe(matriz_complexa_df(seq["Y_seq"] * 1000.0 * 1e6,
                                    ["0", "1", "2"], 4),
                 width='stretch')

with st.expander("Coordenadas de todos os condutores (entrada do czyl)"):
    n_sub = 3 * nb
    rotulos = ([f"Fase {['A','B','C'][k//nb]} - sub {k%nb+1}" for k in range(n_sub)]
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
    "redução de feixe \u2014 rotina `czyl_overhead_bundled` de line_cable_param.py."
)
