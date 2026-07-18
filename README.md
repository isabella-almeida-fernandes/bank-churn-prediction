# Previsão de Churn Bancário: Pipeline de Machine Learning

Este projeto foi desenvolvido de forma colaborativa como trabalho de conclusão do Bootcamp de Ciência de Dados Avanti `2026.1`. O objetivo principal foi analisar o comportamento de 10.000 clientes de uma instituição bancária e construir um pipeline preditivo capaz de identificar clientes propensos ao *churn* (cancelamento de conta), permitindo que o banco tome ações preventivas de retenção.

---

## 🔍 1. Principais Insights da Análise Exploratória (EDA)

A análise dos dados gerou insights fundamentais para a estratégia de negócio:

* **Fator Etário Crítico:** A idade apresentou a maior correlação linear positiva com o churn (0.29). O risco de evasão escala substancialmente na faixa etária entre **40 e 60 anos** (mediana de churn em 45 anos, contra 36 anos dos clientes retidos).
* **Anomalia Geográfica:** Enquanto França e Espanha mantêm taxas de saída abaixo da média global (20,4%), a **Alemanha apresenta uma taxa alarmante superior a 30%**, indicando um problema severo localizado nesse mercado.
* **O Paradoxo dos Produtos:** A relação entre o número de produtos e o churn é não-linear:
  * **1 Produto:** Risco de churn elevado.
  * **2 Produtos:** Ponto ótimo de engajamento e altíssima fidelidade.
  * **3 ou 4 Produtos:** Evasão próxima a 100%, sinalizando forte insatisfação ou ofertas inadequadas.
* **Perda de Capital:** O churn concentra-se em clientes com **saldo positivo e elevado**. Contas zeradas apresentam maior retenção, o que indica que o banco está perdendo seus clientes de maior valor financeiro.
* **Redução de Dimensionalidade (PCA):** A aplicação de Análise de Componentes Principais demonstrou alta sobreposição linear entre as classes (clientes ativos vs. churn), indicando a necessidade de modelos não-lineares capazes de capturar fronteiras de decisão complexas.

---

## 🛠️ 2. Engenharia de Recursos & Pré-processamento

Para preparar os dados para os algoritmos de Machine Learning, estabelecemos o seguinte fluxo de tratamento:
1. **Seleção de Variáveis:** Remoção de identificadores irrelevantes (`RowNumber`, `CustomerId`, `Surname`).
2. **Tratamento de Dados Faltantes:** Embora a base estivesse completa, o pipeline foi estruturado preventivamente utilizando `SimpleImputer` com estratégia de **mediana** para variáveis numéricas e **moda (most frequent)** para categóricas.
3. **Codificação Categórica:** Transformação de variáveis nominais (como `Geography` e `Gender`) utilizando técnicas de binarização (`pd.get_dummies`).
4. **Escalonamento:** Aplicação do `MinMaxScaler` nas variáveis quantitativas para normalizar os dados no intervalo entre 0 e 1, garantindo o bom desempenho de algoritmos baseados em distância (como o kNN).

---

## 📊 3. Modelagem e Validação Cruzada

### Justificativa das Métricas
O conjunto de dados apresenta um desbalanceamento moderado (20,4% de churn). Por isso, a `acurácia` isolada não é uma métrica confiável. Focamos nossa avaliação em:
* **Recall (Sensibilidade):** Métrica prioritária para o negócio. É mais crítico para o banco identificar um cliente que vai sair (mesmo que ofereça um incentivo por engano a um cliente estável) do que deixar um cliente de alto valor sair sem intervenção.
* **F1-Score:** Equilíbrio (média harmônica) entre Precisão e Recall.
* **ROC-AUC:** Capacidade geral do modelo de diferenciar as duas classes.

### Estratégia de Validação Cruzada
Para garantir que os modelos não sofressem de *overfitting*, adotamos duas estratégias complementares de validação:
1. **Stratified K-Fold (10 divisões):** Garante que a proporção de 20% de churn seja mantida rigidamente em cada dobra de treino e teste.
2. **Validação Cruzada de Monte Carlo (ShuffleSplit com 30 repetições):** Testa a estabilidade dos modelos simulando 30 divisões aleatórias diferentes da base (70% treino / 30% teste).

---

## 📈 4. Resultados Analíticos

Quatro algoritmos de diferentes famílias matemáticas foram testados para mapear o comportamento dos dados. Os resultados médios consolidados nas validações foram:

| Modelo | Acurácia Média | Recall Médio | F1-Score Médio | ROC-AUC Média |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | ~79.6% | 0.0% | 0.0% | 0.500 |
| **Regressão Logística** | ~80.9% | ~23.8% | ~33.8% | ~0.771 |
| **K-Nearest Neighbors (kNN)** | ~81.7% | ~31.8% | ~41.3% | ~0.743 |
| **Random Forest (Balanced)** | **~85.8%** | **~48.1%** | **~57.9%** | **~0.852** |

### Conclusão Técnica
Como previsto na análise multivariada (PCA), os modelos lineares (`Regressão Logística`) e baseados estritamente em distância espacial (`kNN`) falharam em capturar as regras complexas de negócio, entregando um Recall muito baixo. 

O modelo **Random Forest**, configurado com peso de classe balanceado (`class_weight='balanced'`), apresentou o **melhor desempenho geral em todas as métricas**, alcançando uma **ROC-AUC de 0.852**. Ele provou ser o mais apto para mapear as não-linearidades do problema, como o comportamento crítico do número de produtos e as particularidades geográficas da Alemanha.

---

## 👥 Contribuição e Trabalho em Equipe
Este projeto foi executado em formato de *squad* ágil durante o bootcamp, exercitando metodologias de divisão de tarefas, versionamento de código e discussão de hipóteses de negócio em equipe.

**Tecnologias Utilizadas:** Python, Pandas, NumPy, Scikit-Learn, SciPy, Matplotlib, Seaborn.
