"""
bias_check.py
Detecta viés e mede fairness (equidade) em modelos de IA.

Por que isso importa em governança de IA:
Um modelo pode ter boa acurácia geral mas discriminar sistematicamente
um grupo específico (gênero, raça, faixa etária). Isso é risco legal,
ético e reputacional. Governança de IA exige que isso seja medido.
"""

import numpy as np
import pandas as pd


def taxa_aprovacao_por_grupo(df: pd.DataFrame,
                              coluna_grupo: str,
                              coluna_predicao: str) -> pd.DataFrame:
    """
    Calcula a taxa de aprovação (predição positiva) por grupo.
    Ponto de partida para identificar disparidade entre grupos.
    """
    resultado = (
        df.groupby(coluna_grupo)[coluna_predicao]
        .agg(total="count", aprovados="sum")
        .assign(taxa_aprovacao=lambda x: round(x["aprovados"] / x["total"] * 100, 2))
        .reset_index()
    )
    return resultado


def disparate_impact(taxa_grupo_a: float, taxa_grupo_b: float) -> dict:
    """
    Disparate Impact Ratio (DIR).
    Métrica jurídica originada no direito trabalhista americano (regra dos 4/5).

    Fórmula: taxa do grupo desfavorecido / taxa do grupo favorecido.

    Interpretação:
      DIR >= 0.80 → sem evidência de discriminação
      DIR < 0.80  → potencial discriminação (viola a regra dos 4/5)
      DIR < 0.60  → discriminação severa
    """
    if taxa_grupo_a == 0:
        return {"dir": None, "status": "INDEFINIDO (taxa zero)"}

    dir_valor = round(taxa_grupo_b / taxa_grupo_a, 4)

    if dir_valor >= 0.80:
        status = "EQUITATIVO"
    elif dir_valor >= 0.60:
        status = "ATENCAO — possível discriminação"
    else:
        status = "CRITICO — discriminação severa"

    return {"dir": dir_valor, "status": status}


def equalized_odds(df: pd.DataFrame,
                   coluna_grupo: str,
                   coluna_predicao: str,
                   coluna_real: str) -> pd.DataFrame:
    """
    Equalized Odds: compara TPR e FPR entre grupos.

    TPR (True Positive Rate): dos que realmente merecem aprovação,
                              quantos o modelo aprovou?
    FPR (False Positive Rate): dos que não mereciam, quantos foram
                               aprovados indevidamente?

    Um modelo justo deve ter TPR e FPR similares entre grupos.
    Diferença grande indica que o modelo erra de forma desigual.
    """
    grupos = []
    for grupo, subset in df.groupby(coluna_grupo):
        tp = ((subset[coluna_predicao] == 1) & (subset[coluna_real] == 1)).sum()
        fp = ((subset[coluna_predicao] == 1) & (subset[coluna_real] == 0)).sum()
        tn = ((subset[coluna_predicao] == 0) & (subset[coluna_real] == 0)).sum()
        fn = ((subset[coluna_predicao] == 0) & (subset[coluna_real] == 1)).sum()

        tpr = round(tp / (tp + fn), 4) if (tp + fn) > 0 else None
        fpr = round(fp / (fp + tn), 4) if (fp + tn) > 0 else None

        grupos.append({
            "grupo": grupo,
            "TPR (sensibilidade)": tpr,
            "FPR (falso positivo)": fpr
        })

    return pd.DataFrame(grupos)


def relatorio_fairness(df: pd.DataFrame,
                       coluna_grupo: str,
                       coluna_predicao: str,
                       coluna_real: str,
                       grupo_referencia: str) -> None:
    """
    Relatório completo de fairness para um modelo.
    """
    print("=" * 55)
    print("RELATÓRIO DE FAIRNESS / VIÉS DO MODELO")
    print("=" * 55)

    print("\n[1] TAXA DE APROVAÇÃO POR GRUPO")
    taxas = taxa_aprovacao_por_grupo(df, coluna_grupo, coluna_predicao)
    print(taxas.to_string(index=False))

    print("\n[2] DISPARATE IMPACT (referência: grupo favorecido)")
    taxa_ref = taxas.loc[
        taxas[coluna_grupo] == grupo_referencia, "taxa_aprovacao"
    ].values[0]
