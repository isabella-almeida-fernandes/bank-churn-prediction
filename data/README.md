# 📁 Fonte de Dados: Churn de Clientes Bancários

Este diretório contém as informações de referência sobre o conjunto de dados utilizado no projeto. Seguindo as boas práticas de engenharia e governança de dados, a base bruta não é rastreada diretamente neste repositório devido à separação entre código e armazenamento de dados.

## 🔗 Link Original do Dataset
Os dados utilizados neste projeto são públicos e foram extraídos originalmente do Kaggle:
* **Dataset:** [Predict Bank Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/saurabhbadole/bank-customer-churn-prediction-dataset)

## ⚙️ Como os Dados são Consumidos no Pipeline
Para garantir a reprodutibilidade do projeto sem a necessidade de downloads manuais de arquivos `.csv` locais, os notebooks foram configurados para consumir os dados diretamente via streaming de uma URL pública:

```python
df = pd.read_csv("[https://raw.githubusercontent.com/omadson/datasets/refs/heads/main/datasets/churn.csv](https://raw.githubusercontent.com/omadson/datasets/refs/heads/main/datasets/churn.csv)")
