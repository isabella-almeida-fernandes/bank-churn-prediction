# 🏦 Previsão de Churn Bancário: Pipeline Completo de Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/Status-Concluído-success.svg)]()

> Projeto prático de Ciência de Dados e Machine Learning desenvolvido durante o **Bootcamp de Ciência de Dados Atlântico Avanti (2026.1)**. O objetivo é analisar o comportamento de 10.000 clientes de uma instituição financeira e construir um pipeline preditivo robusto para identificar precocemente o risco de *churn* (evasão de clientes), subsidiando estratégias orientadas a dados para retenção e redução de perdas financeiras.

---

## 📁 Estrutura do Repositório

```text
├── data/
│   └── README.md                               # Descrição do dataset e instruções de download
├── notebooks/
│   ├── 01_analise_exploratoria_de_dados.ipynb  # EDA, análise estatística e PCA
│   └── 02_analise_comparativa_de_modelos.ipynb # Pré-processamento, validação, tuning e exportação
├── .gitignore                                  # Arquivos e pastas ignorados pelo Git
├── README.md                                   # Documentação principal do projeto
└── requirements.txt                            # Bibliotecas e versões necessárias para reprodução
```

---

## 🔍 1. Principais Insights da Análise Exploratória (EDA)

A análise exploratória detalhada no primeiro notebook revelou padrões determinantes para o negócio:

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

Para garantir consistência e evitar vazamento de dados (*data leakage*), o fluxo de preparação foi estruturado da seguinte forma:

1. **Feature Selection:** Remoção de identificadores e metadados irrelevantes (`RowNumber`, `CustomerId`, `Surname`).
2. **Tratamento de Dados Faltantes:** Implementação preventiva via `SimpleImputer` com estratégia de **mediana** para variáveis numéricas e **moda (most frequent)** para categóricas.
3. **Codificação Categórica:** Transformação de variáveis nominais (`Geography` e `Gender`) utilizando One-Hot Encoding (`pd.get_dummies(drop_first=True)`).
4. **Escalonamento de Features:** Normalização com `MinMaxScaler` nas variáveis quantitativas contínuas e discretas para padronização no intervalo $[0, 1]$.

---

## 📊 3. Metodologia de Modelagem e Validação

### Justificativa das Métricas de Avaliação
Devido ao desbalanceamento das classes (~20% de churn e ~80% de retenção), a **Acurácia** foi descartada como critério isolado de decisão[cite: 1]. A avaliação priorizou:
* **Recall (Sensibilidade):** Métrica primordial do negócio. Para a instituição, o custo de não identificar um cliente que vai cancelar (falso negativo) é muito superior ao custo de uma ação preventiva direcionada a um cliente que permaneceria (falso positivo)[cite: 1].
* **F1-Score:** Equilíbrio harmônico entre Precisão e Recall da classe minoritária (`Exited = 1`)[cite: 1].
* **ROC-AUC:** Medida global da capacidade do modelo de distinguir clientes propensos vs. não propensos ao churn[cite: 1].

### Estratégia de Validação Cruzada
Para atestar a estabilidade e capacidade de generalização dos modelos, foram adotadas duas abordagens:
1. **Stratified 10-Fold CV:** Divisão em 10 dobras preservando rigidamente a proporção de 20% da classe positiva em cada partição[cite: 1].
2. **Monte Carlo Cross-Validation (`ShuffleSplit`):** 30 iterações com divisão aleatória (70% treino / 30% teste) para avaliar a variância das métricas[cite: 1].

---

## 📈 4. Resultados Analíticos e Comparativo de Modelos

Quatro modelos de diferentes famílias matemáticas foram submetidos aos mesmos critérios de validação[cite: 1]:

| Modelo | Acurácia Média | Recall Médio | F1-Score Médio | ROC-AUC Média |
| :--- | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline)** | 79.63% | 0.00% | 0.00% | 0.5000 |
| **Regressão Logística** | 81.08% | 20.32% | 30.41% | 0.7653 |
| **K-Nearest Neighbors (kNN)** | 81.43% | 31.47% | 40.66% | 0.7446 |
| **Random Forest (Class Balanced)** | **85.97%** | **44.68%** | **56.40%** | **0.8546** |

## ⚙️ 5. Otimização de Hiperparâmetros e Avaliação Final

O modelo **Random Forest** foi selecionado para ajuste fino (*hyperparameter tuning*) via **`RandomizedSearchCV`** (5-Fold Stratificado) otimizando a métrica `F1-Score`.

### Avaliação no Conjunto de Teste Independente (20% Holdout):
* **ROC-AUC no Teste:** `0.85+`
* **Melhores Hiperparâmetros:** `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` e `class_weight='balanced'`.

### Importância das Features (*Feature Importance*):
As variáveis mais determinantes para as previsões foram:
1. **`Age` (Idade)**
2. **`NumOfProducts` (Quantidade de Produtos)**
3. **`Balance` (Saldo em Conta)**
4. **`IsActiveMember` (Membro Ativo)**
5. **`Geography_Germany` (Localização: Alemanha)**

---

## 💡 6. Recomendações Práticas para o Negócio

1. **Alerta Preditivo no CRM:** Utilizar as probabilidades preditas (`predict_proba`) para gerar um *Score de Risco de Churn*. Clientes com probabilidade $> 0.50$ devem ser direcionados automaticamente para a fila de atendimento prioritário.
2. **Plano de Contenção para o Mercado Alemão:** Conduzir auditoria de produto e pesquisa de satisfação focada na Alemanha para compreender os gargalos da operação local.
3. **Reestruturação do Cross-Selling:** Reformular os pacotes e benefícios para clientes que possuem 3 ou mais produtos, evitando incentivos desalinhados com a jornada do correntista.
4. **Programa de Ativação para Inativos:** Criar campanhas de reativação para clientes de faixas etárias mais altas com saldo positivo que reduziram sua interação com o banco.

---

## 👥 Equipe e Autores

Projeto desenvolvido de forma colaborativa por:

* **Fábio Agostinho** - [fabinhosnf@gmail.com](mailto:fabinhosnf@gmail.com)
* **Isabella Almeida** - [isabella4lmeidafernandes@gmail.com](mailto:isabella4lmeidafernandes@gmail.com)
* **Marcos Felipe Santos** - [marcosfelipessc@alu.ufc.br](mailto:marcosfelipessc@alu.ufc.br)
* **Maria Souza** - [mariacorreia2505@gmail.com](mailto:mariacorreia2505@gmail.com)

---

## 🛠️ Tecnologias e Bibliotecas Utilizadas

* **Linguagem:** Python 3.10+
* **Manipulação e Análise de Dados:** Pandas, NumPy
* **Visualização de Dados:** Matplotlib, Seaborn
* **Machine Learning & Validação:** Scikit-Learn (Pipelines, Imputer, Scalers, Classifiers, Metrics)
* **Serialização de Modelos:** Joblib
