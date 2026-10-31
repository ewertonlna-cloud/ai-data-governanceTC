"""
bias_check.py
Detecta viés e mede fairness (equidade) em modelos de IA.

Por que isso importa em governança de IA:
Um modelo pode ter boa acurácia geral mas discriminar sistematicamente
um grupo específico (gênero, raça, faixa etária). Isso é risco legal,
ético e reputacional. Governança de IA exige que isso seja medido.

Como rodar o exemplo (dados simulados):
    python3 src/bias_check.py
"""

import numpy as np
import pandas as pd

# Diferença máxima aceitável de TPR/FPR entre grupos. 0,10 (10 pontos percentuais)
# é uma convenção usada como ponto de partida: não é uma regra legal e deve ser
# ajustada ao contexto de cada decisão.
LIMITE_EQUALIZED_ODDS = 0.10


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



def _diferenca(valor: float, referencia: float):
    """Diferença (valor - referência), ou None se algum dos dois não existir."""
    if pd.isna(valor) or pd.isna(referencia):
        return None
    return round(float(valor) - float(referencia), 4)


def _formatar_diferenca(valor) -> str:
    return "n/d" if valor is None else f"{valor:+.2f}"


def diferencas_equalized_odds(odds: pd.DataFrame,
                              grupo_referencia: str,
                              limite: float = LIMITE_EQUALIZED_ODDS) -> list:
    """
    Compara o TPR e o FPR de cada grupo com os do grupo de referência.

    Diferença = grupo - referência. TPR negativo significa que o modelo
    reconhece menos casos reais nesse grupo; FPR positivo, que ele marca
    mais casos indevidamente.
    """
    ref = odds[odds["grupo"] == grupo_referencia]
    if ref.empty:
        raise ValueError(f"Grupo de referência '{grupo_referencia}' não existe nos dados.")
    ref = ref.iloc[0]

    resultados = []
    for _, linha in odds.iterrows():
        if linha["grupo"] == grupo_referencia:
            continue
        dif_tpr = _diferenca(linha["TPR (sensibilidade)"], ref["TPR (sensibilidade)"])
        dif_fpr = _diferenca(linha["FPR (falso positivo)"], ref["FPR (falso positivo)"])

        if dif_tpr is None or dif_fpr is None:
            status = "INDEFINIDO — grupo sem casos suficientes"
        elif abs(dif_tpr) > limite or abs(dif_fpr) > limite:
            status = "ATENCAO — o modelo erra de forma desigual"
        else:
            status = "EQUITATIVO"

        resultados.append({
            "grupo": linha["grupo"],
            "dif_tpr": dif_tpr,
            "dif_fpr": dif_fpr,
            "status": status,
        })
    return resultados


def relatorio_fairness(df: pd.DataFrame,
                       coluna_grupo: str,
                       coluna_predicao: str,
                       coluna_real: str,
                       grupo_referencia: str) -> None:
    """
    Relatório completo de fairness para um modelo.

    O atributo de grupo (ex.: tipo de cliente, gênero) é usado só para MEDIR
    a disparidade, nunca como variável do modelo.
    """
    if grupo_referencia not in set(df[coluna_grupo]):
        raise ValueError(f"Grupo de referência '{grupo_referencia}' não existe nos dados.")

    print("=" * 55)
    print("RELATÓRIO DE FAIRNESS / VIÉS DO MODELO")
    print("=" * 55)

    print("\n[1] TAXA DE PREDIÇÃO POSITIVA POR GRUPO (ex.: aprovado, marcado como urgente)")
    taxas = taxa_aprovacao_por_grupo(df, coluna_grupo, coluna_predicao)
    print(taxas.to_string(index=False))

    print("\n[2] DISPARATE IMPACT (referência: grupo favorecido)")
    taxa_ref = taxas.loc[
        taxas[coluna_grupo] == grupo_referencia, "taxa_aprovacao"
    ].values[0]
    for _, linha in taxas.iterrows():
        grupo = linha[coluna_grupo]
        if grupo == grupo_referencia:
            continue
        resultado = disparate_impact(taxa_ref, linha["taxa_aprovacao"])
        print(f"  {grupo} vs {grupo_referencia}: DIR = {resultado['dir']} → {resultado['status']}")

    print("\n[3] EQUALIZED ODDS (TPR e FPR por grupo)")
    odds = equalized_odds(df, coluna_grupo, coluna_predicao, coluna_real)
    print(odds.to_string(index=False))
    print(f"\n  Diferença em relação a '{grupo_referencia}' (limite: ±{LIMITE_EQUALIZED_ODDS:.2f}):")
    for item in diferencas_equalized_odds(odds, grupo_referencia):
        print(
            f"  {item['grupo']}: TPR {_formatar_diferenca(item['dif_tpr'])}"
            f" | FPR {_formatar_diferenca(item['dif_fpr'])} → {item['status']}"
        )
    print("  (TPR negativo = o modelo reconhece menos casos reais nesse grupo)")

    print("\n" + "=" * 55)


if __name__ == "__main__":
    # DADOS SIMULADOS: nenhum registro é real. Cenário fictício, inspirado em
    # operações de data center: um modelo marca chamados de suporte como
    # urgentes ("predito" = 1). "real" = 1 quando o chamado era de fato urgente.
    # O modelo simulado é propositalmente enviesado: reconhece menos chamados
    # urgentes de clientes de hospedagem compartilhada do que de servidor dedicado,
    # o que na prática significa SLA estourado para esse grupo.
    rng = np.random.default_rng(42)

    def simular_grupo(nome, n, taxa_urgentes, tpr, fpr):
        real = rng.binomial(1, taxa_urgentes, n)
        prob_marcar = np.where(real == 1, tpr, fpr)
        predito = rng.binomial(1, prob_marcar)
        return pd.DataFrame({"grupo": nome, "real": real, "predito": predito})

    df = pd.concat([
        simular_grupo("Servidor dedicado", 500, taxa_urgentes=0.30, tpr=0.90, fpr=0.12),
        simular_grupo("Hospedagem compartilhada", 500, taxa_urgentes=0.30, tpr=0.65, fpr=0.10),
    ], ignore_index=True)

    print("DADOS SIMULADOS (1.000 chamados de suporte fictícios)\n")
    relatorio_fairness(df, "grupo", "predito", "real", grupo_referencia="Servidor dedicado")
