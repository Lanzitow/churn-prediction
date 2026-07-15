"""
Predição de Churn em Telecom
-----------------------------
Objetivo: prever quais clientes têm maior probabilidade de cancelar o
serviço (churn), usando um dataset real de uma operadora de telecom.

Etapas:
1. Carregamento e limpeza dos dados
2. Análise exploratória (EDA)
3. Pré-processamento (encoding + balanceamento com SMOTE)
4. Treinamento de modelos (Regressão Logística e Random Forest)
5. Avaliação de métricas
6. Interpretabilidade com SHAP (quais variáveis mais pesam na decisão)

Autor: Alan Demetrio Gelsleuchter
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # salva gráficos em arquivo, sem precisar de tela
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
import shap

import os

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
sns.set_style("whitegrid")

# 1. Carregamento e limpeza

def load_and_clean(path="data/telco_churn.csv"):
    df = pd.read_csv(path)

    # TotalCharges vem como string e tem alguns valores vazios (clientes
    # com tenure = 0, ou seja, entraram e ainda não foram cobrados)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop(columns=["customerID"])
    return df

# 2. EDA 
def run_eda(df):
    churn_rate = df["Churn"].value_counts(normalize=True) * 100
    print("\nTaxa de churn:")
    print(churn_rate.round(2))

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="Churn", palette=["#2E86AB", "#E63946"])
    plt.title("Distribuição de Churn")
    plt.savefig(f"{OUT_DIR}/01_churn_distribution.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", palette=["#2E86AB", "#E63946"])
    plt.title("Mensalidade vs Churn")
    plt.savefig(f"{OUT_DIR}/02_monthly_charges_vs_churn.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack",
                 palette=["#2E86AB", "#E63946"], bins=30)
    plt.title("Tempo de Contrato (meses) vs Churn")
    plt.savefig(f"{OUT_DIR}/03_tenure_vs_churn.png", bbox_inches="tight")
    plt.close()

    print(f"Gráficos de EDA salvos em {OUT_DIR}/")

# 3. Pré-processamento

def preprocess(df):
    df = df.copy()
    target = df["Churn"].map({"Yes": 1, "No": 0})
    df = df.drop(columns=["Churn"])

    cat_cols = df.select_dtypes(include="object").columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    return df_encoded, target

# 4. Treinamento com SMOTE (balanceamento de classes)

def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\nAntes do SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"Depois do SMOTE:  {pd.Series(y_train_bal).value_counts().to_dict()}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        auc = roc_auc_score(y_test, probs)
        results[name] = {
            "model": model,
            "auc": auc,
            "preds": preds,
            "probs": probs,
        }

        print(f"\n=== {name} ===")
        print(f"AUC-ROC: {auc:.3f}")
        print(classification_report(y_test, preds, target_names=["Ficou", "Cancelou"]))

    return results, X_train, X_test, X_test_scaled, y_test, scaler


# 5. Curva ROC comparando os modelos

def plot_roc(results, y_test):
    plt.figure(figsize=(6, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["probs"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.title("Curva ROC - Comparação de Modelos")
    plt.legend()
    plt.savefig(f"{OUT_DIR}/04_roc_curve.png", bbox_inches="tight")
    plt.close()
    print(f"\nCurva ROC salva em {OUT_DIR}/04_roc_curve.png")



# 6. Interpretabilidade com SHAP (Random Forest)

def run_shap(results, X_train, X_test_scaled, X_columns):
    rf_model = results["Random Forest"]["model"]

    explainer = shap.TreeExplainer(rf_model)
    # Amostra menor para deixar o cálculo rápido
    sample = X_test_scaled[:200]
    shap_values = explainer.shap_values(sample)

    # shap_values pode vir como lista [classe0, classe1] dependendo da versão
    values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(
        values_to_plot, sample, feature_names=X_columns,
        show=False, plot_size=(8, 6)
    )
    plt.savefig(f"{OUT_DIR}/05_shap_summary.png", bbox_inches="tight")
    plt.close()
    print(f"Gráfico SHAP salvo em {OUT_DIR}/05_shap_summary.png")


# ----------------------------------------------------------------------
def main():
    print("Carregando e limpando os dados...")
    df = load_and_clean()

    print("Rodando EDA...")
    run_eda(df)

    print("Pré-processando (encoding)...")
    X, y = preprocess(df)

    print("Treinando modelos com balanceamento SMOTE...")
    results, X_train, X_test, X_test_scaled, y_test, scaler = train_models(X, y)

    plot_roc(results, y_test)

    print("Calculando importância das variáveis (SHAP)...")
    run_shap(results, X_train, X_test_scaled, X.columns.tolist())

    print("\nAnálise concluída! Veja os gráficos na pasta outputs/")


if __name__ == "__main__":
    main()
