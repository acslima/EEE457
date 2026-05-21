**Definição dos 5 critérios:**

| Vértice | Definição | Fórmula | Direção |
|---|---|---|---|
| Perdas | Perdas resistivas 3φ | $3I^2 R_{\text{AC}} \ell$ | menor → melhor (score invertido) |
| Ampacidade | Capacidade térmica | $I_{\max}$ | maior → melhor |
| Perfil V | Regulação de tensão (curto-circuito) | $\Delta V\% = \frac{I(R\cos\phi + X\sin\phi)}{V/\sqrt{3}} \times 100$ | menor → melhor (invertido) |
| P nominal | Potência natural (SIL) | $V^2 / Z_c$ | maior → melhor |
| P máxima | Limite de estabilidade | $V^2 / (X_L \ell)$ | maior → melhor |

**Observação importante:** todos os 5 critérios elétricos favorecem condutores maiores — perdas caem com menor $R_{\text{AC}}$, capacidade térmica sobe com $I_{\max}$, e os parâmetros $Z_c$, $X_L$ também evoluem favoravelmente. O critério de polígono máximo resulta portanto em **Bittern como ótimo** pelos critérios puramente elétricos.

A análise de **custo econômico** (sessão anterior) é o contrapeso natural que cria a trade-off real: o Bittern vence eletricamente mas o Grosbeak/Tern vencem economicamente. Se desejar, posso incorporar o custo como 6º vértice do polígono, criando o conflito projetual explícito.