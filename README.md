# 🏦 Previsão de Churn Bancário: Pipeline Completo de Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Optuna](https://img.shields.io/badge/Optuna-Bayesian_Tuning-blueviolet.svg)](https://optuna.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Concluído-success.svg)]()

> Projeto prático de Ciência de Dados e Machine Learning desenvolvido durante o **Bootcamp de Ciência de Dados Atlântico Avanti (2026.1)**. O objetivo é analisar o comportamento de 10.000 clientes de uma instituição financeira e construir um pipeline preditivo robusto para identificar precocemente o risco de *churn* (evasão de clientes), subsidiando estratégias orientadas a dados para retenção e redução de perdas financeiras.

---

## 📁 Estrutura do Repositório

```text
├── data/
│   ├── Churn_Modelling.csv         # Conjunto de dados utilizado
│   └── README.md                   # Metadados e dicionário das variáveis
├── notebooks/
│   └── bank_churn_prediction.ipynb # Notebook unificado (EDA, Pipeline, Tuning e Avaliação)
├── models/
│   └── pipeline_churn_random_forest.pkl # Pipeline serializada (pré-processador + modelo)
├── .gitignore                      # Arquivos e pastas ignorados pelo Git
├── README.md                       # Documentação principal do projeto
└── requirements.txt                # Dependências do projeto
```

---

## 🔍 1. Principais Insights da Análise Exploratória (EDA)

A análise exploratória detalhada revelou padrões determinantes para o negócio:

* **Fator Etário Crítico:** A idade apresentou a maior correlação linear positiva com o churn ($r = 0.29$). O risco de evasão escala substancialmente na faixa entre **40 e 60 anos** (mediana de churn em 45 anos, contra 36 anos dos clientes retidos).
* **Anomalia Geográfica:** Enquanto França e Espanha mantêm taxas de evasão abaixo da média global (20,4%), a **Alemanha apresenta uma taxa superior a 30%**, indicando atritos operacionais ou maior pressão competitiva localizada nesse mercado.
* **O Paradoxo dos Produtos:** A relação entre o número de produtos bancários e o churn é altamente não-linear:
  * **1 Produto:** Risco de evasão moderado/alto.
  * **2 Produtos:** Ponto ótimo de engajamento e altíssima taxa de retenção.
  * **3 ou 4 Produtos:** Evasão próxima a 100%, sinalizando insatisfação severa ou processos de *cross-selling* agressivos e mal calibrados.
* **Evasão de Alto Valor (Perda de Capital):** O churn concentra-se fortemente em clientes com **saldo positivo e elevado**. Contas com saldo zerado apresentaram menor taxa de saída, demonstrando que o banco está perdendo justamente seus clientes de maior rentabilidade.
* **Complexidade Não-Linear (PCA):** A análise de componentes principais demonstrou sobreposição linear significativa entre as classes, confirmando a necessidade de algoritmos capazes de mapear fronteiras de decisão não-lineares complexas.

---

## 🛠️ 2. Engenharia de Recursos e Pré-processamento

Para garantir consistência e evitar vazamento de dados (*data leakage*), o fluxo de preparação foi estruturado em uma esteira modular via `ColumnTransformer` e `Pipeline` do Scikit-Learn:

1. **Seleção de Atributos:** Remoção de identificadores únicos sem poder preditivo (`RowNumber`, `CustomerId`, `Surname`).
2. **Divisão Estratificada (Holdout):** Separação inicial de 20% da base para teste final com estratificação (`stratify=y`), garantindo que os dados de teste permaneçam inéditos até a validação definitiva.
3. **Tratamento de Dados Numéricos:** Imputação preventiva pela mediana (`SimpleImputer`) combinada a escalonamento com `StandardScaler` (ajustados estritamente nas dobras de treino).
4. **Codificação Categórica:** Imputação preventiva pela moda e transformação de variáveis nominais (`Geography` e `Gender`) via `OneHotEncoder(drop='first', handle_unknown='ignore')`.
5. **Variáveis Binárias:** Manutenção direta dos valores originais (0 e 1) de `HasCrCard` e `IsActiveMember` via `passthrough`.
   
---

## 🎯 3. Metodologia de Modelagem e Métricas de Negócio

### ⚠️ Por que a Acurácia não é a métrica adequada?
Em bases desbalanceadas (~80% retidos e ~20% churn), a acurácia é uma métrica ilusória. Um modelo ingênuo (*Dummy Classifier*) que simplesmente preveja que **nenhum cliente sairá do banco** atinge ~80% de acurácia, mas possui **utilidade zero para o negócio**, pois é incapaz de antecipar qualquer perda real de cliente.

### ⚖️ O Trade-off da Matriz de Confusão: Falso Positivo vs. Falso Negativo
No contexto de retenção bancária, os custos de cada erro são altamente assimétricos:

| Tipo de Erro | O que acontece na prática | Impacto Financeiro para o Banco |
| :--- | :--- | :--- |
| **Falso Positivo (FP)** | O modelo prevê churn, mas o cliente iria permanecer. | **Custo da ação de retenção:** Desconto tarifário, brinde ou ligação comercial dispensável. |
| **Falso Negativo (FN)** | O modelo prevê permanência, mas o cliente cancela a conta. | **Perda do LTV (*Lifetime Value*):** Perda direta de receita, custódia e alto custo de aquisição (CAC) para repor o cliente. |

> **Decisão Estratégica:** Como o **Falso Negativo custa muito mais caro**, priorizamos o **Recall da classe minoritária ($y=1$)**, balanceado pelo **F1-Score** para evitar que uma enxurrada de Falsos Positivos onere os custos da operação.

### Estratégia de Validação Cruzada: K-Fold Estratificado
Como a base não possui dados sequenciais ou datas transacionais que exijam divisão temporal (*Time Series Split*), adotou-se o **StratifiedKFold ($k=5$)**. A estratificação garante que cada subconjunto de treino e validação preserve rigorosamente a mesma proporção de ~20% de churn, evitando distorções na avaliação das métricas.

---

## 📈 4. Resultados Analíticos e Comparativo de Modelos

Quatro algoritmos de diferentes famílias matemáticas foram avaliados em pipeline completa sob validação cruzada estratificada (`StratifiedKFold`, $k=5$):

| Modelo | Acurácia Média | Recall Médio | F1-Score Médio | ROC-AUC Média |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Dummy)** | 79.62% | 0.00% | 0.00% | 0.5000 |
| **Regressão Logística** | 70.80% | **68.40%** | 48.83% | 0.7662 |
| **K-Nearest Neighbors** | 83.95% | 43.25% | 52.30% | 0.7898 |
| **Random Forest** | **85.94%** | 44.11% | **56.04%** | **0.8507** |

> **Diagnóstico:** 
> - O **Baseline (Dummy)** ilustra o perigo de avaliar classes desbalanceadas por acurácia: apesar de pontuar quase 80%, seu Recall e F1-Score são estritamente nulos ($0.00\%$).
> - A **Regressão Logística** (com pesos balanceados) capturou o maior volume bruto de churners ($68.40\%$ de Recall), porém com menor precisão global.
> - O **Random Forest** alcançou a melhor capacidade discriminativa global (**ROC-AUC de 0.8507**) e o maior equilíbrio harmônico (**F1-Score de 56.04%**), sendo selecionado como o estimador principal para o ajuste fino via Optuna.

---

## ⚙️ 5. Otimização Bayesiana de Hiperparâmetros e Avaliação Final

O algoritmo **Random Forest** avançou para o ajuste fino com **Optuna**, aplicando otimização bayesiana guiada pelo amostrador **TPE (*Tree-structured Parzen Estimator*)** ao longo de 30 iterações com objetivo de maximizar o **F1-Score**.

### Espaço de Hiperparâmetros Otimizado:
* `n_estimators`: Número de árvores no comitê.
* `max_depth`: Limite de profundidade para controle de complexidade e poda.
* `min_samples_split` e `min_samples_leaf`: Regularização contra nós excessivamente específicos.
* `max_features`: Critério de subamostragem de atributos (`sqrt` vs. `log2`).
* `class_weight`: Ponderação balanceada para compensar o desbalanceamento das classes.

---

### Avaliação no Conjunto de Teste Independente (20% Holdout):
A pipeline com os melhores hiperparâmetros foi avaliada nos dados de teste nunca antes vistos pelo modelo:
* **ROC-AUC no Teste:** Superior a 0.85, indicando alta capacidade de discriminação entre clientes propensos vs. retidos.
* **Generalização:** As métricas no teste mantiveram alinhamento com a validação cruzada, atestando a ausência de *data leakage* e *overfitting*.

---

### Importância dos Atributos (*Feature Importance*):
A extração de importância por impureza de Gini confirmou os padrões identificados na análise exploratória como os maiores direcionadores de evasão:

1. **`Age` (Idade):** Principal determinante isolado do risco de churn.
2. **`NumOfProducts` (Número de Produtos):** Sinalizador crítico de atrito em contas com 3 ou mais contratações.
3. **`Balance` (Saldo em Conta):** Concentração de risco em contas com volume expressivo de capital sob custódia.
4. **`IsActiveMember` (Membro Ativo):** Inatividade como alerta precoce de desengajamento.
5. **`Geography_Germany` (Localização: Alemanha):** Mercado regional com taxa de cancelamento substancialmente mais alta.

---

## 💡 6. Recomendações Práticas para o Negócio

1. **Alerta Preditivo Integrado ao CRM:** Consumir as probabilidades preditas (`predict_proba`) em lote mensal para atribuir um *Score de Risco de Churn* a cada cliente ativo. Contas com score elevado devem ser roteadas para réguas de retenção antes do encerramento voluntário.
2. **Plano de Contenção para o Mercado Alemão:** Conduzir auditoria de produto e pesquisa qualitativa com correntistas da Alemanha para mapear gargalos operacionais específicos, cobrança de tarifas ou ofertas agressivas da concorrência local.
3. **Auditoria na Experiência Multicontas (3+ Produtos):** Investigar os pontos de atrito enfrentados por correntistas com 3 ou mais contratos, avaliando sobreposição de tarifas, falhas de usabilidade no aplicativo ou desalinhamento de benefícios nas campanhas de *cross-sell*.
4. **Trilha Prioritária de Relacionamento para Alta Renda:** Acionar gerentes dedicados para correntistas com saldo elevado e idade acima de 40 anos que apresentem sinais de inatividade operacional recente (`IsActiveMember = 0`).
   
---

## 👥 7. Autora:

Projeto desenvolvido por:

* **Isabella Almeida** - [isabella4lmeidafernandes@gmail.com](mailto:isabella4lmeidafernandes@gmail.com)

---

## 🛠️ 8. Tecnologias e Bibliotecas Utilizadas

* **Linguagem:** Python 3.10+
* **Manipulação e Análise de Dados:** Pandas, NumPy
* **Visualização de Dados:** Matplotlib, Seaborn
* **Machine Learning & Validação:** Scikit-Learn (Pipelines, ColumnTransformer, Scalers, Classifiers, Metrics)
* **Otimização de Hiperparâmetros:** Optuna (Bayesian Optimization / TPE)
* **Serialização e Deploy de Modelos:** Joblib
