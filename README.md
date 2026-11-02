# Problema 1 — Regressão Linear para previsão de produção de energia

## 1. Correlação entre as variáveis independentes e `PE`

Resultados obtidos:

| Variável | Correlação com PE |
|---|---:|
| AT | -0,9481 |
| V | -0,8698 |
| AP | 0,5184 |
| RH | 0,3898 |

**Quais variáveis parecem ter maior influência na produção?** `AT` apresenta a maior correlação em valor absoluto (-0,9481), seguida de `V` (-0,8698). As duas têm forte relação negativa com `PE`: quando seus valores aumentam, a produção tende a diminuir. `AP` apresenta relação positiva moderada, enquanto `RH` apresenta relação positiva mais fraca.

A correlação mostra associação linear, não prova causalidade. Portanto, é mais correto dizer que `AT` e `V` são as variáveis com associação linear mais forte com a produção neste conjunto.

## 2. Correlação entre as variáveis independentes

O par com maior correlação é `AT` e `V`, com **0,8441**. As correlações de `AT` com `RH` (-0,5425) e `AP` (-0,5075) também são moderadas. `AP` e `RH` apresentam correlação baixa (0,0996).

**Existe relação forte entre elas?** Sim. A correlação positiva de 0,8441 entre `AT` e `V` é forte e pode indicar multicolinearidade, pois as duas entradas carregam informações parcialmente semelhantes. Isso pode dificultar a interpretação e a estabilidade dos coeficientes da regressão. A correlação, sozinha, é um indício; o VIF seria uma verificação complementar.

## 3. Modelo de regressão linear múltipla

O notebook dividiu os dados em 80% para treino (7.654 registros) e 20% para teste (1.914 registros). Não foram encontrados valores ausentes nas quatro variáveis de entrada. O modelo usa `AT`, `V`, `AP` e `RH` para prever `PE`.

## 4. Avaliação do modelo

Resultados no conjunto de teste:

- **R² = 0,9301**
- **RMSE = 4,5026 MW**
- **MAE = 3,5959 MW**

**O modelo é adequado?** Os resultados indicam que sim, para este conjunto de dados. O R² de aproximadamente 0,93 mostra que o modelo explica grande parte da variação de `PE`. O RMSE e o MAE indicam erros médios de poucos megawatts, relativamente pequenos diante da faixa observada de produção (420,26 a 495,76 MW).

# Problema 2 — Regressão Logística para previsão de doenças cardíacas

## 1. Treinamento e número de iterações

A base contém 303 pacientes. O alvo original foi convertido em duas classes: `0` para sem doença e `1` para com doença. Após a binarização, há 164 registros da classe 0 e 139 da classe 1, uma distribuição relativamente equilibrada.

Foram identificados valores ausentes em `ca` (4) e `thal` (2); os demais atributos não apresentaram ausências. O pré-processamento trata os valores ausentes e padroniza as entradas.

**Quantas iterações foram necessárias?** O modelo convergiu em **5 iterações**, conforme `n_iter_`. Como o limite foi definido em 1000, a convergência ocorreu antes de atingir o máximo permitido.

## 2. Avaliação do modelo

No conjunto de teste, com 61 pacientes, os resultados foram:

| Métrica | Resultado |
|---|---:|
| Acurácia | 0,8689 (86,89%) |
| Precisão | 0,8125 (81,25%) |
| Recall | 0,9286 (92,86%) |
| F1-score | 0,8667 (86,67%) |
| AUC | 0,9513 |

**Interpretação:** a acurácia de 86,89% indica que o modelo classificou corretamente a maioria dos pacientes do teste. A precisão de 81,25% significa que, entre os pacientes previstos como tendo doença, cerca de 81% realmente estavam na classe positiva. O recall de 92,86% indica que o modelo identificou a maior parte dos pacientes que tinham doença. A AUC de 0,9513 indica boa capacidade de separar as duas classes ao variar o limiar de decisão.

**O desempenho é bom?** Sim, os números são promissores para esse conjunto, especialmente o recall e a AUC. Porém, o conjunto de teste tem apenas 61 casos e o modelo não deve ser considerado uma ferramenta clínica pronta. Em saúde, falsos negativos são importantes porque correspondem a pacientes com doença classificados como sem doença. O modelo serve para fins acadêmicos e não substitui avaliação médica.
# Problema 3 — Classificação da qualidade de vinhos com k-NN

## 1. Preparação e balanceamento das classes

Após converter `quality` em três categorias, a distribuição foi:

| Classe | Quantidade |
|---|---:|
| `medio` | 837 |
| `ruim` | 744 |
| `bom` | 18 |

A classe `bom` é extremamente minoritária: são apenas 18 exemplos, enquanto as outras classes têm centenas. Isso torna difícil para o modelo aprender o padrão da classe `bom` e faz com que a acurácia geral possa esconder um desempenho ruim nessa classe.

**Por que normalizar para k-NN?** O k-NN calcula distâncias entre amostras. Como os atributos químicos usam escalas diferentes, variáveis com valores numericamente maiores poderiam dominar a distância. O `StandardScaler` coloca as entradas em escalas comparáveis e evita que isso aconteça. No notebook, a padronização e a imputação estão dentro de um pipeline.

## 2. Divisão treino/teste

A divisão foi feita em 80% para treino e 20% para teste, com estratificação para preservar, tanto quanto possível, a proporção das classes. Entretanto, a classe `bom` é tão pequena que o teste contém somente **4 exemplos** dessa classe, conforme o relatório de classificação. Por isso, a avaliação dessa classe é bastante instável.

## 3. Escolha do valor de k

A validação cruzada de 5 folds apresentou os seguintes resultados:

| k | Acurácia média |
|---:|---:|
| 1 | 0,7248 |
| 3 | 0,6888 |
| 5 | 0,7013 |
| 7 | 0,7060 |
| 11 | 0,7115 |

O melhor resultado médio foi obtido com **k = 1**, com acurácia de 0,7248 (72,48%). Isso foi o critério usado para selecionar o modelo final.

**O que ocorre com k muito baixo ou muito alto?** Com `k` muito baixo, a decisão depende de poucos vizinhos e fica sensível a ruídos e exemplos individuais, podendo ocorrer overfitting. Com `k` muito alto, muitos vizinhos influenciam a decisão, o que pode suavizar demais as fronteiras e favorecer as classes mais numerosas, causando underfitting. Aqui, `k=1` teve a maior acurácia média, mas isso não garante bom reconhecimento da classe rara `bom`.

## 4. Avaliação no teste

A acurácia no teste foi **0,7656 (76,56%)**. As métricas por classe foram:

| Classe | Precisão | Recall | F1-score | Suporte no teste |
|---|---:|---:|---:|---:|
| `bom` | 0,00 | 0,00 | 0,00 | 4 |
| `medio` | 0,76 | 0,80 | 0,78 | 167 |
| `ruim` | 0,77 | 0,75 | 0,76 | 149 |

**Entre quais classes o modelo erra com mais frequência?** O relatório mostra que o modelo não identificou nenhum dos 4 vinhos da classe `bom` corretamente. Os erros entre `ruim` e `medio` também podem ocorrer, mas a matriz de confusão é necessária para indicar com precisão quantos exemplos foram trocados em cada direção. O problema mais evidente é a classe `bom`, cuja precisão, recall e F1-score são todos zero.

**Conclusão:** a acurácia geral de 76,56% parece razoável, mas é enganosa se analisada sozinha. O modelo funciona melhor para as classes `medio` e `ruim`, mas não reconheceu a classe `bom`. Para melhorar, seria importante obter mais exemplos de vinhos bons ou aplicar técnicas de tratamento do desbalanceamento apenas nos dados de treino e avaliar com métricas sensíveis a cada classe.

# Problema 5 — Segmentação de clientes com k-Means

## 1. Estatísticas, assimetria, transformação logarítmica e padronização

As seis categorias de gastos apresentam grande dispersão e valores máximos muito altos. Por exemplo, `Fresh` varia de 3 a 112.151, `Milk` de 55 a 73.498 e `Grocery` de 3 a 92.780. Em várias categorias, a média é bem maior que a mediana, o que é compatível com uma distribuição assimétrica à direita: poucos clientes gastam muito mais do que a maioria.

**Por que aplicar `log(1+x)`?** A transformação reduz a distância numérica entre gastos muito altos e os demais, diminuindo a influência de valores extremos e tornando a distribuição menos assimétrica. O `+1` permite transformar também valores iguais a zero.

**Por que padronizar por Z-score?** O k-Means atribui os pontos aos centróides usando distância euclidiana. Sem padronização, as categorias com valores ou variâncias maiores poderiam dominar o cálculo da distância. A padronização coloca as seis categorias em escalas comparáveis. A sequência usada foi transformar os gastos com log e depois padronizá-los.

## 2. Correlação entre as categorias

O par `Grocery`–`Detergents_Paper` apresentou correlação de **0,9246**, uma associação positiva muito forte. Isso significa que clientes que gastam mais em mercearia tendem também a gastar mais em detergentes e papel. `Milk` também tem correlação positiva com `Grocery` (0,7283) e com `Detergents_Paper` (0,6618).

Essa relação sugere que essas categorias podem ajudar a distinguir perfis de clientes com gastos altos nesses produtos. Ainda assim, correlação não significa que uma categoria cause o gasto na outra.

## 3. Por que não separar treino e teste?

O k-Means é um método não supervisionado: não existe uma classe correta conhecida que o modelo deva prever durante o treinamento. O objetivo é encontrar estrutura ou grupos nos próprios dados, não estimar o erro de previsão de rótulos conhecidos como nos problemas supervisionados.

Isso não significa que a validação seja dispensável. É necessário avaliar a qualidade dos grupos com critérios internos, como silhueta e inércia, e com critérios externos quando houver rótulos disponíveis para comparação. `Channel` e `Region` foram reservados para essa validação e não devem entrar no agrupamento.

## 4. Escolha de k: cotovelo e silhueta

Foram testados valores de `k` entre 2 e 10. O maior coeficiente médio de silhueta foi obtido com **k = 2**. A silhueta compara quão próximo um ponto está de seu próprio grupo e quão distante está dos outros; valores maiores indicam, em geral, grupos mais separados e compactos.

**Por que a inércia diminui quando k aumenta?** Com mais grupos, cada ponto pode ficar mais próximo do centróide que lhe foi atribuído, então a soma dos quadrados das distâncias dentro dos grupos tende a diminuir ou permanecer igual. Por isso, a inércia sozinha não escolhe o melhor `k`: ela favorece valores maiores.

**O que ocorre quando k é igual ao número de amostras?** Em princípio, cada cliente pode formar seu próprio grupo e a inércia fica zero, pois cada ponto coincide com o centróide de seu grupo. Essa solução não é útil para segmentação comercial, porque não resume os clientes em perfis acionáveis.

**Escolha final:** os resultados indicam `k=2` pelo critério da silhueta. A decisão também deve considerar o gráfico do cotovelo e se dois perfis são úteis para a área comercial.

## 5. Inicialização random versus k-means++

Os resultados das dez execuções foram muito semelhantes:

- `random`: inércia média 1844,08 e desvio padrão 0,03;
- `k-means++`: inércia média 1844,09 e desvio padrão 0,03.

**Qual método foi melhor?** Neste experimento, nenhum mostrou vantagem clara: as médias e a dispersão foram praticamente iguais, e `random` teve uma média de inércia ligeiramente menor. Essa diferença é mínima.

O `k-means++` busca escolher os centróides iniciais de maneira mais espalhada, reduzindo a chance de iniciar com centróides muito próximos e ajudando a evitar soluções ruins. Porém, ele não garante encontrar o mínimo global. Neste conjunto e com estas sementes, os dois métodos chegaram a resultados quase idênticos.

## 6. Interpretação dos centróides e nomes comerciais

Os centróides foram revertidos para a escala original de gastos. Os valores médios aproximados foram:

| Categoria | Grupo 0 | Grupo 1 |
|---|---:|---:|
| Fresh | 8.580,6 | 4.013,9 |
| Milk | 1.753,7 | 8.061,6 |
| Grocery | 2.274,4 | 12.051,1 |
| Frozen | 2.063,8 | 953,5 |
| Detergents_Paper | 287,4 | 4.035,6 |
| Delicassen | 649,4 | 1.022,1 |

**Interpretação comercial:**
- **Grupo 0 — perfil de frescos e compras mais diversificadas:** tem gastos mais altos em `Fresh` e `Frozen`, mas baixos em `Milk`, `Grocery` e, principalmente, `Detergents_Paper`.
- **Grupo 1 — perfil de mercearia, laticínios e limpeza:** destaca-se em `Milk`, `Grocery` e `Detergents_Paper`, com gasto menor em `Fresh` e `Frozen`.

Os nomes são interpretações dos centróides, não rótulos fornecidos originalmente pela base.

## 7. Validação externa com Channel e Region

Tabela de grupo por canal:

| Grupo | Channel 1 | Channel 2 |
|---|---:|---:|
| 0 | 244 | 8 |
| 1 | 54 | 134 |

A tabela mostra associação entre os grupos e o canal. O Grupo 0 contém principalmente clientes do `Channel 1`, enquanto o Grupo 1 contém uma maioria de clientes do `Channel 2`. O **Adjusted Rand Index (ARI) foi 0,5135**, indicando concordância moderada entre os agrupamentos e o canal. Portanto, o algoritmo recuperou parte da estrutura do canal sem usar essa variável no agrupamento, mas não a reproduziu perfeitamente.

**Quais categorias mais separam Horeca de Retail?** Os centróides sugerem que `Grocery`, `Milk` e `Detergents_Paper` são as categorias que mais diferenciam os dois perfis. Essa interpretação é compatível com o Grupo 1 apresentar gastos bem mais altos nessas três categorias.

Tabela de grupo por região:

| Grupo | Região 1 | Região 2 | Região 3 |
|---|---:|---:|---:|
| 0 | 48 | 28 | 176 |
| 1 | 29 | 19 | 140 |

As duas classes de agrupamento contêm muitos clientes da Região 3 e proporções regionais relativamente parecidas. Assim, os resultados não indicam uma separação geográfica forte entre os grupos.

## 8. PCA e limitações do k-Means

As duas primeiras componentes principais explicaram, juntas, **71,27% da variância** (44,08% + 27,19%). A projeção em duas dimensões resume boa parte, mas não toda, da variação dos dados; portanto, o gráfico é uma visualização aproximada.

Duas limitações importantes do k-Means são:

1. **Forma dos grupos:** tende a funcionar melhor com grupos compactos, aproximadamente esféricos e de escala semelhante. Pode ter dificuldade quando os perfis têm formas ou densidades muito diferentes.
2. **Sensibilidade a outliers:** como usa médias e distâncias quadráticas, clientes com gastos muito extremos podem deslocar os centróides. A transformação logarítmica ajuda a reduzir esse efeito, mas não elimina todas as limitações.

**Conclusão geral:** a solução com dois grupos tem interpretação comercial razoável e apresenta concordância moderada com `Channel` (ARI = 0,5135), mas não deve ser interpretada como uma reconstrução perfeita do canal ou da região.
