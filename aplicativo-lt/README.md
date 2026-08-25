# Parâmetros unitários de linhas de transmissão aéreas

Frontend em Python (Streamlit) para o cálculo das matrizes de impedância (**Z**)
e admitância (**Y**) por unidade de comprimento de linhas aéreas, a partir da
tabela de condutores CAA (ACSR). Suporta de **1 a 3 circuitos trifásicos em
paralelo**, no mesmo nível de tensão, compartilhando o corredor (mesma torre ou
torres lado a lado). Desenvolvido para a disciplina
**EEE 457 — Transmissão de Energia Elétrica** (Escola Politécnica / COPPE — UFRJ).

A metodologia de cálculo reproduz a do notebook
`eee457-06_calculo_param_unitarios.ipynb` — rotina `czyl_overhead_bundled`.

## Arquivos

| Arquivo | Função |
|---|---|
| `app.py` | Interface Streamlit (frontend) |
| `lt_core.py` | Núcleo de cálculo: catenária, leitura da tabela, geometria, Z/Y |
| `line_cable_param.py` | Rotinas eletromagnéticas (módulo original, sem alterações) |
| `condutores-caa.csv` | Tabela de 66 condutores CAA |
| `requirements.txt` | Dependências |

Os quatro primeiros arquivos devem permanecer na **mesma pasta**.

## Execução

Com `uv`:

```bash
uv run --with-requirements requirements.txt streamlit run app.py
```

Ou em um ambiente já preparado:

```bash
pip install -r requirements.txt
streamlit run app.py
```

O navegador abre em `http://localhost:8501`.

## Circuitos em paralelo (novo)

Na seção **2 — Circuitos em paralelo** da barra lateral escolhe-se o número de
circuitos (1, 2 ou 3). Convenções e comportamento:

- **Ordenação das fases**: C1-A, C1-B, C1-C, C2-A, …, C3-C. Para arranjos com
  transposição de barras (ex.: ABC/CBA no circuito duplo), basta editar as
  coordenadas da seção 6 de acordo — a fase "A" de cada circuito é, por
  definição, a primeira linha daquele circuito na tabela.
- **Geometrias padrão**: 1 circuito → configuração horizontal do notebook;
  2 circuitos → torre de circuito duplo com fases em disposição vertical;
  3 circuitos → três torres lado a lado (25 m entre eixos). Tudo editável.
- **Limitações herdadas de `czyl_overhead_bundled`**: todos os circuitos usam
  o **mesmo condutor**, o **mesmo `nb`** e a **mesma tensão nominal**. A mesma
  flecha de fase é aplicada a todos os circuitos (mesmo condutor e tração).
- **Cálculo**: a rotina `czyl_overhead_bundled` é aplicada uma única vez ao
  conjunto completo de condutores do corredor (3·n fases + até 4 para-raios),
  de modo que o **acoplamento eletromagnético entre circuitos** (retorno pelo
  solo e capacitâncias mútuas) é capturado naturalmente nas matrizes
  (3n × 3n) de fase.
- **Sequências**: a transformação de Fortescue é aplicada em blocos
  (T = I ⊗ A). O app mostra as impedâncias/admitâncias de sequência
  **próprias de cada circuito**, as matrizes de **acoplamento entre circuitos**
  M₀ (sequência zero — tipicamente significativa) e M₁ (positiva) e o
  **equivalente dos circuitos em paralelo** (mesmos barramentos):
  `z_eq = 1/(1ᵀ M₁⁻¹ 1)` (série) e `y_eq = 1ᵀ N₁ 1` (shunt). Z_c, Pₙ (SIL),
  L₁ e a velocidade de propagação são calculados a partir do equivalente.
  Com 1 circuito, tudo se reduz exatamente ao comportamento anterior.

## O que o aplicativo faz

1. **Lê a tabela de condutores** (CSV) e extrai, do condutor escolhido, as
   grandezas necessárias: diâmetro externo (raio `r_f`), diâmetro da alma de
   aço (raio interno `r_int`), resistência CC e massa.
2. **Calcula a flecha pela catenária**: `f = a·(cosh(L/2a) − 1)`, com
   `a = H/w`. O peso linear `w` vem da tabela; a tração horizontal `H` é dada
   como percentual da carga de ruptura (RTS). Há também a opção de informar a
   flecha diretamente.
3. **Para-raios**: considerados 3/8" EHS. A flecha é obtida pela catenária a
   partir da tração e do peso do próprio cabo:
   `H_pr = (% RTS)·T_pr`, `C_pr = H_pr/W_pr`,
   `flecha = C_pr·(cosh(L/2C_pr) − 1)`. Os valores padrão são
   `T_pr = 6980 kgf`, `W_pr = 410 kg/km` e tração de **25 % da RTS** — o que
   resulta em **8,887 m para um vão de 550 m**. A flecha também pode ser
   informada diretamente.
4. **Feixe de subcondutores** — três modos:
   - **Circular**: os subcondutores são vértices de um polígono regular; basta
     informar o espaçamento entre subcondutores adjacentes.
   - **Elíptico**: os subcondutores caem sobre uma elipse de semi-eixos
     `(a, b)` — `a` horizontal e `b` vertical —, com posições
     `(x_k, y_k) = (a·cos(t_k), b·sin(t_k))` e `t_k = t₀ + 2πk/nb`. As **fases
     laterais** (A e C) usam um par `(aₗ, bₗ)` e a **fase central** (B) usa um
     par independente `(aᶜ, bᶜ)`. Fazendo `a = b` recupera-se o feixe circular
     de raio `a`.
   - **Manual**: você entra diretamente com uma tabela por fase (A, B, C)
     contendo as coordenadas **(x, y)** absolutas de cada subcondutor no
     **ponto de fixação** da torre. A flecha continua sendo aplicada
     automaticamente (`y_média = y_fixação − (2/3)·flecha`). Nesse modo a
     seção 5 (centro dos feixes) é ignorada — útil quando as coordenadas
     vêm de desenhos de fabricante e não seguem uma parametrização simples.

   Em todos os modos o número de subcondutores por fase (`nb`) é o mesmo nas
   três fases — limitação da rotina `czyl_overhead_bundled`.
5. **Monta as coordenadas** de todos os subcondutores. A altura média de cada
   condutor é `y = y_fixação − (2/3)·flecha`.
6. **Calcula Z e Y** com `czyl_overhead_bundled` (impedância externa de Carson,
   impedância interna de condutor tubular, redução de Kron dos para-raios e
   redução de feixe).
7. **Apresenta** as matrizes em Ω/km e μS/km, as componentes de sequência e,
   em destaque, os valores unitários da **sequência positiva**:
   - **Z₁**: R₁, X₁, |Z₁| e ∠Z₁
   - **Y₁**: G₁, B₁, |Y₁| e ∠Y₁ (em μS/km)
   - **Desempenho do circuito**: impedância característica `Z_c = √(z₁/y₁)`,
     potência natural `Pₙ = Vₙ²/Re(Z_c)`, indutância unitária `L₁ = X₁/ω` e
     velocidade de propagação `v = 1/√(L₁C₁)`.

   Também mostra os gráficos da estrutura (com o contorno da elipse ou do
   círculo de cada feixe) e do perfil de catenária.

## Reproduzindo o notebook

Para recuperar a primeira configuração do notebook (circuito 345 kV, dois
condutores Tern), use: condutor **TERN**, `nb = 2`, espaçamento `0,4572 m`,
geometria das fases já preenchida por padrão, **flecha de fase direta = 19,1 m**
e **flecha dos para-raios direta = 15,24 m**.
