"""
data_quality.py
Verifica qualidade dos dados antes de alimentar um modelo de IA.
Governança: garante que dados ruins não contaminem o modelo.
"""

import pandas as pd
import numpy as np


def verificar_nulos(df: pd.DataFrame) -> dict:
    """
    Retorna % de valores nulos por coluna.
    Por que importa: modelos de IA se comportam mal com dados faltantes não tratados.
    """
    total = len(df)
    resultado = {}
    for col in df.columns:
        nulos = df[col].isna().sum()
        pct = round((nulos / total) * 100, 2)
        status = "OK" if pct == 0 else ("ALERTA" if pct < 10 else "CRITICO")
        resultado[col] = {"nulos": int(nulos), "percentual": pct, "status": status}
    return resultado


def verificar_duplicatas(df: pd.DataFrame) -> dict:
    """
    Detecta linhas duplicadas.
    Por que importa: duplicatas inflam métricas e criam vazamento de dados (data leakage).
    """
    total = len(df)
    duplicatas = df.duplicated().sum()
    pct = round((duplicatas / total) * 100, 2)
    return {
        "total_linhas": total,
        "duplicatas": int(duplicatas),
        "percentual": pct,
        "status": "OK" if duplicatas == 0 else "ALERTA"
    }


def verificar_outliers(df: pd.DataFrame, colunas_numericas: list) -> dict:
    """
    Usa IQR (intervalo interquartil) para detectar outliers.
    Por que importa: outliers distorcem médias e prejudicam modelos lineares.
    """
    resultado = {}
    for col in colunas_numericas:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr
        outliers = df[(df[col] < limite_inf) | (df[col] > limite_sup)][col].count()
        resultado[col] = {
            "outliers": int(outliers),
            "limite_inferior": round(limite_inf, 4),
            "limite_superior": round(limite_sup, 4),
            "status": "OK" if outliers == 0 else "ALERTA"
        }
    return resultado


def relatorio_completo(df: pd.DataFrame, colunas_numericas: list = None) -> None:
    """
    Roda todas as verificações e imprime um relatório legível.
    """
    if colunas_numericas is None:
        colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    print("=" * 50)
    print("RELATÓRIO DE QUALIDADE DE DADOS")
    print("=" * 50)

    print("\n[1] VALORES NULOS")
    for col, info in verificar_nulos(df).items():
        print(f"  {col}: {info['percentual']}% nulos → {info['status']}")

    print("\n[2] DUPLICATAS")
    dup = verificar_duplicatas(df)
    print(f"  {dup['duplicatas']} duplicatas ({dup['percentual']}%) → {dup['status']}")

    print("\n[3] OUTLIERS (método IQR)")
    for col, info in verificar_outliers(df, colunas_numericas).items():
        print(f"  {col}: {info['outliers']} outliers → {info['status']}")

    print("\n" + "=" * 50)


# Exemplo de uso com dados simulados
if __name__ == "__main__":
    np.random.seed(42)
    dados = {
        "idade": [25, 30, np.nan, 200, 28, 30, 25],   # 200 é outlier, 1 nulo
        "salario": [3000, 5000, 4000, 4500, np.nan, 5000, 3000],  # 1 nulo, duplicata
        "score_credito": [700, 650, 720, 680, 710, 650, 700]
    }
    df = pd.DataFrame(dados)
    relatorio_completo(df)
