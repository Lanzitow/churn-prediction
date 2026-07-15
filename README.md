# Predição de Churn em Telecom

Projeto de Machine Learning para prever quais clientes de uma operadora de
telecomunicações têm maior probabilidade de cancelar o serviço (churn),
usando um dataset real com ~7.000 clientes.

## Objetivo

Empresas de assinatura (telecom, SaaS, distribuidoras com clientes recorrentes)
perdem receita quando não conseguem identificar a tempo quais clientes estão
prestes a cancelar. Este projeto constrói um pipeline completo — da limpeza
dos dados até a explicação do modelo — para apoiar decisões de retenção.

## O que o projeto faz

1. **Limpeza e tratamento dos dados** (valores ausentes, tipos incorretos)
2. **Análise exploratória (EDA)** — relação entre tempo de contrato,
   mensalidade e churn
3. **Balanceamento de classes com SMOTE** — o dataset original tem só ~27%
   de clientes que cancelam, o que enviesa o modelo se não for tratado
4. **Treinamento de dois modelos**: Regressão Logística e Random Forest
5. **Avaliação** com AUC-ROC, precisão, recall e matriz de confusão
6. **Interpretabilidade com SHAP** — não basta prever quem vai cancelar,
   é preciso entender *por quê*, para que o time de negócio saiba em qual
   variável agir (ex: tipo de contrato, forma de pagamento, tempo de casa)

## Resultados

| Modelo | AUC-ROC |
|---|---|
| Regressão Logística | 0.845 |
| Random Forest | 0.829 |

As variáveis que mais influenciam a decisão de cancelamento (segundo o
SHAP) incluem tipo de contrato (mês a mês vs. anual), tempo de casa
(tenure) e forma de pagamento — insights que uma área de retenção pode
usar diretamente.

## Como rodar

```bash
pip install -r requirements.txt
python churn_analysis.py
```

Os gráficos gerados (distribuição de churn, curva ROC, importância das
variáveis) são salvos automaticamente na pasta `outputs/`.

## Dataset

Telco Customer Churn — dataset público amplamente usado em projetos de
Data Science, disponibilizado pela IBM.

## Tecnologias

Python · Pandas · Scikit-learn · Imbalanced-learn (SMOTE) · SHAP ·
Matplotlib · Seaborn

---
Projeto desenvolvido por Alan Demetrio Gelsleuchter como parte do
portfólio de Ciência de Dados.
