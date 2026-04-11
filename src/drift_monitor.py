"""
drift_monitor.py
Detecta data drift: quando os dados que chegam para o modelo
começam a ser diferentes dos dados com que ele foi treinado.

Por que isso importa em governança de IA:
Um modelo treinado em janeiro pode degradar em julho se o
comportamento dos usuários ou do mercado mudar. Sem monitoramento,
você só descobre isso quando o modelo já errou muito.
"""

import numpy as np
from scipy import stats


def psi(esperado: np.ndarray, observado: np.ndarray, bins: int = 10) -> float:
    """
    PSI — Population Stability Index.
    Mede o quanto a distribuição de uma variável mudou entre
    o período de treino (esperado) e produção (observado).

    Interpretação padrão de mercado:
      PSI < 0.10  → estável, sem ação necessária
      PSI < 0.25  → mudança moderada, monitorar
      PSI >= 0.25 → mudança severa, revaliar o modelo
    """
    # Cria os bins com base nos dados esperados
    breakpoints = np.linspace(
        min(esperado.min(), observado.min()),
        max(esperado.max(), observado.max()),
        bins + 1
    )

    esp_counts, _ = np.histogram(esperado, bins=breakpoints)
    obs_counts, _ = np.histogram(observado, bins=breakpoints)

    # Evita divisão por zero
    esp_pct = np.where(esp_counts == 0, 0.0001, esp_counts / len(esperado))
    obs_pct = np.where(obs_counts == 0, 0.0001, obs_counts / len(observado))

    psi_valor = np.sum((obs_pct - esp_pct) * np.log(obs_pct / esp_pct))
    return round(float(psi_valor), 4)


def ks_test(esperado: np.ndarray, observado: np.ndarray) -> dict:
    """
    Teste de Kolmogorov-Smirnov.
    Verifica estatisticamente se duas amostras vêm da mesma distribuição.

    p-value < 0.05 → diferença estatisticamente significativa (drift detectado).
    """
    statistic, p_value = stats.ks_2samp(esperado, observado)
    return {
        "statistic": round(float(statistic), 4),
        "p_value": round(float(p_value), 4),
        "drift_detectado": p_value < 0.05
    }


def classificar_psi(valor: float) -> str:
    if valor < 0.10:
        return "ESTAVEL"
    elif valor < 0.25:
        return "ATENCAO"
    return "CRITICO"


def relatorio_drift(nome_feature: str,
                    dados_treino: np.ndarray,
                    dados_producao: np.ndarray) -> None:
    """
    Relatório completo de drift para uma feature.
    """
    psi_valor = psi(dados_treino, dados_producao)
    ks = ks_test(dados_treino, dados_producao)

    print("=" * 50)
    print(f"DRIFT MONITOR — feature: {nome_feature}")
    print("=" * 50)
    print(f"  Média treino:    {dados_treino.mean():.2f}")
    print(f"  Média produção:  {dados_producao.mean():.2f}")
    print(f"  Desvio treino:   {dados_treino.std():.2f}")
    print(f"  Desvio produção: {dados_producao.std():.2f}")
    print(f"\n  PSI:  {psi_valor} → {classificar_psi(psi_valor)}")
    print(f"  KS:   statistic={ks['statistic']}, p-value={ks['p_value']}")
    print(f"  Drift detectado (KS): {'SIM' if ks['drift_detectado'] else 'NAO'}")
    print("=" * 50)


# Simulação: dados de treino vs produção com drift intencional
if __name__ == "__main__":
    np.random.seed(42)

    # Cenário 1: sem drift — mesma distribuição
    treino_ok = np.random.normal(loc=50, scale=10, size=1000)
    producao_ok = np.random.normal(loc=50, scale=10, size=1000)
    relatorio_drift("renda_mensal (sem drift)", treino_ok, producao_ok)

    print()

    # Cenário 2: com drift — média subiu 15 pontos
    treino_drift = np.random.normal(loc=50, scale=10, size=1000)
    producao_drift = np.random.normal(loc=65, scale=12, size=1000)
    relatorio_drift("renda_mensal (com drift)", treino_drift, producao_drift)
