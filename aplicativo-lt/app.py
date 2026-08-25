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
import json
import os
from datetime import datetime

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
CONFIG_PADRAO = os.path.join(DIR, "ultima_config.json")


def _json_default(o):
    """Converte tipos numpy/pandas para tipos nativos ao serializar em JSON."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _df_para_registros(df: pd.DataFrame) -> list:
    """Serializa um DataFrame como lista de registros (dicionarios)."""
    return df.to_dict(orient="records")


def salvar_ultima_config(caminho: str, config: dict) -> None:
    """Grava a ultima configuracao executada em um arquivo JSON."""
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=_json_default)


def carregar_ultima_config(caminho: str) -> dict:
    """Le a ultima configuracao salva; retorna {} se ausente ou invalida."""
    if not os.path.exists(caminho):
        return {}
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _cfg(caminho: str, padrao):
    """Busca um valor aninhado na config inicial (ex.: 'operacao.frequencia_hz').

    Retorna ``padrao`` quando a chave nao existe ou o valor salvo e None.
    """
    no = st.session_state.get("cfg_inicial") or {}
    for chave in caminho.split("."):
        if isinstance(no, dict) and chave in no:
            no = no[chave]
        else:
            return padrao
    return padrao if no is None else no


def _indice(opcoes: list, valor, padrao: int = 0) -> int:
    """Indice de ``valor`` em ``opcoes`` (para restaurar radios/selectbox)."""
    return opcoes.index(valor) if valor in opcoes else padrao


def _df_from_registros(registros, colunas, padrao: pd.DataFrame) -> pd.DataFrame:
    """Reconstroi um DataFrame a partir de registros salvos em JSON."""
    if not registros:
        return padrao
    try:
        df = pd.DataFrame(registros)
        if not set(colunas).issubset(df.columns):
            return padrao
        return df[colunas].reset_index(drop=True)
    except (ValueError, KeyError):
        return padrao


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


def grafico_estrutura(x, y, centers, gw_positions, nb, npr,
                      phase_offsets=None, draw_bundle_outline=True):
    """Desenha a silhueta da torre com os condutores nas alturas médias.

    Se ``phase_offsets`` for informado (lista com 3 arrays (nb,2) para A, B, C),
    o contorno (círculo ou elipse) de cada feixe é traçado como guia visual —
    apenas quando ``draw_bundle_outline`` é True (para geometrias paramétricas).
    """
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
        # centro do feixe na altura MEDIA (não na fixação) para ficar sobre os pontos
        cy_med = np.mean(y[ini:fim]) if nb >= 1 else cy
        ax.scatter([cx], [cy_med], s=30, marker="x",
                   color=cores[k], alpha=0.6)
        ax.annotate(rot_fase[k], (np.mean(x[ini:fim]), np.mean(y[ini:fim])),
                    textcoords="offset points", xytext=(10, 8),
                    fontsize=11, color=cores[k])
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
                        color=cores[k], ls=":", lw=0.9, alpha=0.6, zorder=1)

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
# Carrega UMA vez por sessao a ultima configuracao salva, para servir de valor
# inicial dos widgets. O arquivo e reescrito a cada execucao, por isso so lemos
# no inicio da sessao (evita "resetar" o que o usuario altera durante o uso).
if "cfg_inicial" not in st.session_state:
    st.session_state.cfg_inicial = carregar_ultima_config(CONFIG_PADRAO)
    st.session_state.cfg_carregada = bool(st.session_state.cfg_inicial)

if st.session_state.get("cfg_carregada"):
    _ts_carregado = _cfg("timestamp", "")
    st.sidebar.success(
        "\u21a9\ufe0f Configuração inicial carregada de `ultima_config.json`"
        + (f" ({_ts_carregado})." if _ts_carregado else "."))
    if st.sidebar.button("Restaurar padrões de fábrica"):
        try:
            if os.path.exists(CONFIG_PADRAO):
                os.remove(CONFIG_PADRAO)
        except OSError:
            pass
        st.session_state.clear()
        st.rerun()

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
_cond_salvo = _cfg("condutor_fase", "TERN")
if _cond_salvo in nomes:
    _idx_cond = nomes.index(_cond_salvo)
else:
    _idx_cond = nomes.index("TERN") if "TERN" in nomes else 0
nome_sel = st.sidebar.selectbox("Condutor de fase", nomes, index=_idx_cond)
linha_cond = df_cond[df_cond[core.COL["nome"]] == nome_sel].iloc[0]

st.sidebar.header("2. Condições de operação")
freq = st.sidebar.number_input(
    "Frequência (Hz)", 1.0, 1000.0,
    float(_cfg("operacao.frequencia_hz", 60.0)), 1.0)
rho_solo = st.sidebar.number_input(
    "Resistividade do solo (\u03a9\u00b7m)", 1.0, 1.0e5,
    float(_cfg("operacao.resistividade_solo_ohm_m", 1000.0)), 50.0)
temp_op = st.sidebar.number_input(
    "Temperatura do condutor (\u00b0C)", -20.0, 150.0,
    float(_cfg("operacao.temperatura_condutor_c", 20.0)), 5.0,
    help="Corrige a resistência CC a partir do valor tabelado a 20 °C. "
         "O efeito pelicular é incluído pelo modelo de impedância interna.")
Vn = st.sidebar.number_input(
    "Tensão nominal (kV)", 1.0, 1500.0,
    float(_cfg("operacao.tensao_nominal_kv", 345.0)), 1.0)

st.sidebar.header("3. Vão e tração (catenária)")
span = st.sidebar.number_input(
    "Vão entre estruturas (m)", 50.0, 2000.0,
    float(_cfg("vao_tracao.vao_m", 400.0)), 10.0)
_opc_flecha = ["Catenária (vão + tração)", "Flecha direta (m)"]
modo_flecha = st.sidebar.radio(
    "Flecha dos condutores de fase", _opc_flecha,
    index=_indice(_opc_flecha, _cfg("vao_tracao.modo_flecha", _opc_flecha[0])))
if modo_flecha.startswith("Catenária"):
    frac_rts = st.sidebar.slider(
        "Tração horizontal (% da carga de ruptura)", 5.0, 50.0,
        float(_cfg("vao_tracao.fracao_rts_pct", 20.0)), 0.5,
        help="Tração de trabalho (EDS). Valores típicos: 18 % a 25 % da RTS.")
    flecha_direta = None
else:
    frac_rts = None
    flecha_direta = st.sidebar.number_input(
        "Flecha de fase (m)", 0.5, 60.0,
        float(_cfg("vao_tracao.flecha_direta_m", 19.1)), 0.1)

st.sidebar.header("4. Feixe de subcondutores")
nb = st.sidebar.number_input(
    "Subcondutores por fase (nb)", 1, 8, int(_cfg("feixe.nb", 2)), 1)

# variaveis de saida (offsets por fase, alem de descritores para o grafico)
tipo_feixe = "circular"
espac = 0.0
ang0 = 0.0
# semi-eixos (a, b) por grupo de fase
a_lat = b_lat = a_cent = b_cent = 0.0

if nb > 1:
    _opc_feixe = ["Circular (mesma em todas as fases)",
                  "Elíptico (fase central distinta das laterais)",
                  "Manual (coordenadas de cada subcondutor)"]
    tipo_feixe = st.sidebar.radio(
        "Geometria do feixe", _opc_feixe,
        index=_indice(_opc_feixe, _cfg("feixe.tipo_feixe", _opc_feixe[0])),
        help="Circular: polígono regular com espaçamento uniforme. "
             "Elíptico: subcondutores sobre uma elipse de semi-eixos (a, b), "
             "com fase central podendo diferir das laterais. "
             "Manual: você informa diretamente os deslocamentos (dx, dy) "
             "de cada subcondutor em relação ao centro do feixe de cada fase.")

    if tipo_feixe.startswith("Circular"):
        ang0 = st.sidebar.number_input(
            "Ângulo de orientação do feixe (\u00b0)", 0.0, 360.0,
            float(_cfg("feixe.angulo_orientacao_graus", 0.0)), 15.0)
        espac = st.sidebar.number_input(
            "Espaçamento entre subcondutores adjacentes (m)",
            0.05, 2.0, float(_cfg("feixe.espacamento_m", 0.4572)), 0.0001,
            format="%.4f")
    elif tipo_feixe.startswith("Elíptico"):
        ang0 = st.sidebar.number_input(
            "Ângulo de orientação do feixe (\u00b0)", 0.0, 360.0,
            float(_cfg("feixe.angulo_orientacao_graus", 0.0)), 15.0)
        st.sidebar.markdown("**Semi-eixos das fases laterais (A e C)**")
        a_lat = st.sidebar.number_input(
            "a\u2097 - horizontal (m)", 0.01, 2.0,
            float(_cfg("feixe.semi_eixos_laterais.a", 0.2286)), 0.001,
            format="%.4f", key="a_lat")
        b_lat = st.sidebar.number_input(
            "b\u2097 - vertical (m)", 0.01, 2.0,
            float(_cfg("feixe.semi_eixos_laterais.b", 0.2286)), 0.001,
            format="%.4f", key="b_lat")
        st.sidebar.markdown("**Semi-eixos da fase central (B)**")
        a_cent = st.sidebar.number_input(
            "a\u1D9C - horizontal (m)", 0.01, 2.0,
            float(_cfg("feixe.semi_eixos_central.a", 0.3000)), 0.001,
            format="%.4f", key="a_cent")
        b_cent = st.sidebar.number_input(
            "b\u1D9C - vertical (m)", 0.01, 2.0,
            float(_cfg("feixe.semi_eixos_central.b", 0.1500)), 0.001,
            format="%.4f", key="b_cent")
        st.sidebar.caption(
            "Feixe circular como caso particular: informe a = b "
            "(igual ao raio do polígono usual).")
    else:  # Manual
        st.sidebar.caption(
            "Informe as coordenadas **(x, y)** de cada subcondutor no **ponto "
            "de fixação** da torre. A altura média usada no cálculo é obtida "
            "de  y_média = y_fixação \u2212 (2/3)·flecha. Neste modo a seção 5 "
            "é ignorada.")

        # Ponto de partida: layout do notebook (feixe circular nb=2 s=0.4572)
        offs_default = core.bundle_offsets(nb, 0.4572, 0.0)
        centros_default = [(-8.5, 28.4), (0.0, 29.25), (8.5, 28.4)]
        subcond_labels = [f"sub {k+1}" for k in range(nb)]
        _man_salvo = _cfg("feixe.subcondutores_manuais", {})
        _cols_man = ["Sub", "x (m)", "y fixação (m)"]

        def _tabela_coords(fase_label: str, centro, key: str) -> pd.DataFrame:
            xc, yc = centro
            df0 = pd.DataFrame({
                "Sub": subcond_labels,
                "x (m)":         np.round(xc + offs_default[:, 0], 4),
                "y fixação (m)": np.round(yc + offs_default[:, 1], 4),
            })
            # Restaura coordenadas salvas apenas quando o n de subcondutores casa.
            reg = _man_salvo.get(fase_label) if isinstance(_man_salvo, dict) else None
            if reg and len(reg) == nb:
                df0 = _df_from_registros(reg, _cols_man, df0)
            st.sidebar.markdown(f"**Fase {fase_label}**")
            return st.sidebar.data_editor(
                df0, hide_index=True, disabled=["Sub"], key=key,
                num_rows="fixed")

        tab_A = _tabela_coords("A", centros_default[0], key=f"manA_nb{nb}")
        tab_B = _tabela_coords("B", centros_default[1], key=f"manB_nb{nb}")
        tab_C = _tabela_coords("C", centros_default[2], key=f"manC_nb{nb}")

modo_manual = tipo_feixe.startswith("Manual")

fases_padrao = pd.DataFrame({
    "Fase": ["A", "B", "C"],
    "x (m)": [-8.5, 0.0, 8.5],
    "y fixação (m)": [28.4, 29.25, 28.4],
})
fases_inicial = _df_from_registros(
    _cfg("geometria_fases", None),
    ["Fase", "x (m)", "y fixação (m)"], fases_padrao)
if not modo_manual:
    st.sidebar.header("5. Geometria das fases")
    st.sidebar.caption("Coordenadas do CENTRO de cada feixe (ponto de fixação).")
    fases_edit = st.sidebar.data_editor(
        fases_inicial, hide_index=True, disabled=["Fase"], key="fases")
else:
    # No modo Manual, a seção 5 é ignorada — usamos o default só para não quebrar
    # eventuais referências.
    fases_edit = fases_padrao

st.sidebar.header("6. Cabos para-raios")
_opc_npr = [0, 1, 2]
npr = st.sidebar.selectbox(
    "Número de para-raios", _opc_npr,
    index=_indice(_opc_npr, int(_cfg("para_raios.numero", 2)), 2))
if npr > 0:
    st.sidebar.caption("Considerados 3/8\" EHS.")
    gw_padrao_full = pd.DataFrame({
        "Para-raios": ["PR-1", "PR-2"],
        "x (m)": [-6.25, 6.25],
        "y fixação (m)": [35.9, 35.9],
    })
    gw_inicial = gw_padrao_full.iloc[:npr].reset_index(drop=True)
    _gw_salvo = _cfg("para_raios.posicoes", None)
    gw_inicial = _df_from_registros(
        _gw_salvo, ["Para-raios", "x (m)", "y fixação (m)"], gw_inicial)
    if len(gw_inicial) != npr:  # config salva com outro n de para-raios
        gw_inicial = gw_padrao_full.iloc[:npr].reset_index(drop=True)
    gw_edit = st.sidebar.data_editor(
        gw_inicial, hide_index=True, disabled=["Para-raios"], key=f"gw{npr}")
    rpr = st.sidebar.number_input(
        "Raio do para-raios (m)", 1e-3, 2e-2,
        float(_cfg("para_raios.raio_m", core.GW_RADIUS_DEFAULT)),
        1e-4, format="%.5f")
    rdcpr = st.sidebar.number_input(
        "Resistência CC do para-raios (\u03a9/m)", 1e-4, 5e-2,
        float(_cfg("para_raios.resistencia_cc_ohm_m", core.GW_RDC_DEFAULT)),
        1e-4, format="%.5f")
    _opc_flecha_gw = ["Catenária (tração + peso)", "Flecha direta (m)"]
    modo_flecha_gw = st.sidebar.radio(
        "Flecha dos para-raios", _opc_flecha_gw,
        index=_indice(_opc_flecha_gw,
                      _cfg("para_raios.modo_flecha", _opc_flecha_gw[0])))
    if modo_flecha_gw.startswith("Catenária"):
        Tpr = st.sidebar.number_input(
            "Carga de ruptura do para-raios  T_pr (kgf)",
            500.0, 50000.0,
            float(_cfg("para_raios.carga_ruptura_kgf", core.GW_RTS_DEFAULT)), 10.0)
        Wpr = st.sidebar.number_input(
            "Peso do para-raios  W_pr (kg/km)",
            50.0, 3000.0,
            float(_cfg("para_raios.peso_kg_km", core.GW_WEIGHT_DEFAULT)), 5.0)
        frac_rts_gw = st.sidebar.slider(
            "Tração horizontal do para-raios (% da RTS)",
            5.0, 50.0,
            float(_cfg("para_raios.fracao_rts_pct", core.GW_TENSION_FRAC_DEFAULT)),
            0.5, help="H_pr = (fração) \u00b7 T_pr.  Valor usual: 25 % da RTS.")
        flecha_gw_direta = None
    else:
        Tpr, Wpr, frac_rts_gw = (core.GW_RTS_DEFAULT, core.GW_WEIGHT_DEFAULT,
                                 core.GW_TENSION_FRAC_DEFAULT)
        flecha_gw_direta = st.sidebar.number_input(
            "Flecha dos para-raios (m)", 0.5, 60.0,
            float(_cfg("para_raios.flecha_direta_m", 8.887)), 0.1)
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
gw_positions = [(float(r["x (m)"]), float(r["y fixação (m)"]))
                for _, r in gw_edit.iterrows()] if npr > 0 else []

if modo_manual:
    # Coordenadas absolutas de cada subcondutor (fixação); flecha aplicada aqui.
    def _extrair(tab):
        return tab[["x (m)", "y fixação (m)"]].to_numpy(dtype=float)

    fix_A = _extrair(tab_A)
    fix_B = _extrair(tab_B)
    fix_C = _extrair(tab_C)

    # Centro de cada feixe = centroide dos subcondutores (apenas para o gráfico)
    centers = [tuple(fix_A.mean(axis=0)),
               tuple(fix_B.mean(axis=0)),
               tuple(fix_C.mean(axis=0))]

    # Offsets relativos ao centroide, para renderização e para uso uniforme
    off_A = fix_A - np.asarray(centers[0])
    off_B = fix_B - np.asarray(centers[1])
    off_C = fix_C - np.asarray(centers[2])
    phase_offsets = [off_A, off_B, off_C]

    x, y = core.build_coordinates_per_phase(
        centers, phase_offsets, flecha_fase, gw_positions, flecha_gw)
else:
    centers = [(float(r["x (m)"]), float(r["y fixação (m)"]))
               for _, r in fases_edit.iterrows()]

    # offsets de subcondutor por fase (A, B, C)
    if nb == 1:
        off_all = np.zeros((1, 2))
        phase_offsets = [off_all, off_all, off_all]
    elif tipo_feixe.startswith("Circular"):
        off_all = core.bundle_offsets(nb, espac, ang0)
        phase_offsets = [off_all, off_all, off_all]
    else:  # Elíptico
        off_lat_arr = core.elliptical_bundle_offsets(nb, a_lat, b_lat, ang0)
        off_cent_arr = core.elliptical_bundle_offsets(nb, a_cent, b_cent, ang0)
        phase_offsets = [off_lat_arr, off_cent_arr, off_lat_arr]

    x, y = core.build_coordinates_per_phase(
        centers, phase_offsets, flecha_fase, gw_positions, flecha_gw)

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

# ===========================================================================
# PERSISTÊNCIA - última configuração executada (ultima_config.json)
# ===========================================================================
config_atual = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "fonte_condutores": fonte,
    "condutor_fase": nome_sel,
    "operacao": {
        "frequencia_hz": freq,
        "resistividade_solo_ohm_m": rho_solo,
        "temperatura_condutor_c": temp_op,
        "tensao_nominal_kv": Vn,
    },
    "vao_tracao": {
        "vao_m": span,
        "modo_flecha": modo_flecha,
        "fracao_rts_pct": frac_rts,
        "flecha_direta_m": flecha_direta,
    },
    "feixe": {
        "nb": nb,
        "tipo_feixe": tipo_feixe,
        "modo_manual": modo_manual,
        "angulo_orientacao_graus": ang0,
        "espacamento_m": espac,
        "semi_eixos_laterais": {"a": a_lat, "b": b_lat},
        "semi_eixos_central": {"a": a_cent, "b": b_cent},
    },
    "geometria_fases": _df_para_registros(fases_edit),
    "para_raios": {
        "numero": npr,
        "raio_m": rpr,
        "resistencia_cc_ohm_m": rdcpr,
        "modo_flecha": modo_flecha_gw,
        "carga_ruptura_kgf": Tpr,
        "peso_kg_km": Wpr,
        "fracao_rts_pct": frac_rts_gw,
        "flecha_direta_m": flecha_gw_direta,
        "posicoes": _df_para_registros(gw_edit) if npr > 0 else [],
    },
    "resultados": {
        "flecha_fase_m": flecha_fase,
        "flecha_para_raios_m": flecha_gw,
        "z1_ohm_por_km": {"re": (seq["z1"] * 1000.0).real,
                          "im": (seq["z1"] * 1000.0).imag},
        "y1_uS_por_km": {"re": (seq["y1"] * 1e6 * 1000.0).real,
                         "im": (seq["y1"] * 1e6 * 1000.0).imag},
    },
}
if modo_manual:
    config_atual["feixe"]["subcondutores_manuais"] = {
        "A": _df_para_registros(tab_A),
        "B": _df_para_registros(tab_B),
        "C": _df_para_registros(tab_C),
    }

try:
    salvar_ultima_config(CONFIG_PADRAO, config_atual)
    st.sidebar.caption(
        f"\U0001F4BE Configuração salva em `ultima_config.json` "
        f"({config_atual['timestamp']}).")
except OSError as exc:  # pragma: no cover
    st.sidebar.warning(f"Não foi possível salvar a configuração: {exc}")

st.sidebar.download_button(
    "Baixar última configuração (JSON)",
    json.dumps(config_atual, ensure_ascii=False, indent=2, default=_json_default),
    "ultima_config.json", "application/json")

# --- gráficos ---
g1, g2 = st.columns([1, 1])
with g1:
    st.pyplot(grafico_estrutura(x, y, centers, gw_positions, nb, npr,
                                phase_offsets=phase_offsets,
                                draw_bundle_outline=not tipo_feixe.startswith("Manual")))
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

# destaque para Z1, Y1 e desempenho do circuito
st.markdown("**Sequência positiva \u2014 destaque**")

# linha 1: Z1
z1 = seq["z1"]
mod_z1 = abs(z1)
ang_z1 = np.degrees(np.angle(z1))
r1, r2, r3, r4 = st.columns(4)
r1.metric("R\u2081  (\u03a9/km)", f"{z1.real:.5f}")
r2.metric("X\u2081  (\u03a9/km)", f"{z1.imag:.5f}")
r3.metric("|Z\u2081|  (\u03a9/km)", f"{mod_z1:.5f}")
r4.metric("\u2220 Z\u2081  (\u00b0)", f"{ang_z1:.2f}")

# linha 2: Y1 (em uS/km, como no notebook)
y1_us = seq["y1"] * 1e6
mod_y1 = abs(y1_us)
ang_y1 = np.degrees(np.angle(y1_us))
q1, q2, q3, q4 = st.columns(4)
q1.metric("G\u2081  (\u03bcS/km)", f"{y1_us.real:.4f}")
q2.metric("B\u2081  (\u03bcS/km)", f"{y1_us.imag:.4f}")
q3.metric("|Y\u2081|  (\u03bcS/km)", f"{mod_y1:.4f}")
q4.metric("\u2220 Y\u2081  (\u00b0)", f"{ang_y1:.2f}")

# linha 3: desempenho do circuito (Zc, Pn, capacitância aparente, velocidade)
# Zc (Ohm) - impedancia caracteristica; velocidade v = 1/sqrt(LC)
# a partir de z1 = R + jωL e y1 = G + jωC:
omega_rad = 2.0 * np.pi * freq
L_km = z1.imag / omega_rad      # H/km
C_km = seq["y1"].imag / omega_rad  # F/km
v_prop = 1.0 / np.sqrt(max(L_km * C_km, 1e-30)) if L_km > 0 and C_km > 0 else 0.0
n1, n2, n3, n4 = st.columns(4)
n1.metric("Z\u2080  carac. (\u03a9)", f"{np.real(seq['zc']):.2f}")
n2.metric(f"P\u2099 @ {Vn:.0f} kV (MW)", f"{seq['Pn']:.1f}")
n3.metric("L\u2081 (mH/km)", f"{L_km*1e3:.4f}")
n4.metric("v de propagação (km/s)", f"{v_prop:.0f}")

st.caption(
    "Z\u2081 e Y\u2081 são os autovalores da sequência positiva das matrizes "
    "de fase. Z\u2080 = \u221a(z\u2081/y\u2081), P\u2099 = V\u2099\u00b2/Re(Z\u2080). "
    "A capacitância e a indutância unitárias, bem como a velocidade de "
    "propagação  v = 1/\u221a(L\u2081C\u2081),  são deduzidas de Im(z\u2081)/\u03c9 "
    "e Im(y\u2081)/\u03c9."
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
