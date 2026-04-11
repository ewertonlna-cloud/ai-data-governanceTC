# 🔍 AI Data Governance — Estatística Prática para IA

Repositório com scripts Python para **governança de dados em modelos de Inteligência Artificial**.  
Foco em três pilares fundamentais: qualidade dos dados, estabilidade temporal e equidade algorítmica.

---

## 📌 Por que governança de dados em IA?

Todo modelo de IA é tão confiável quanto os dados que o alimentam.  
Sem monitoramento ativo, três problemas silenciosos degradam modelos em produção:

| Problema | Consequência | Script |
|---|---|---|
| Dados sujos (nulos, duplicatas, outliers) | Modelo aprende padrões errados | `data_quality.py` |
| Data drift (dados mudam com o tempo) | Modelo que funcionava para de funcionar | `drift_monitor.py` |
| Viés nos dados | Modelo discrimina grupos sistematicamente | `bias_check.py` |

---

## 📂 Estrutura do projeto 
ai-data-governance/
│
├── src/
│   ├── data_quality.py   # Verificação de qualidade dos dados
│   ├── drift_monitor.py  # Monitoramento de data drift (PSI + KS test)
│   └── bias_check.py     # Detecção de viés e fairness (DIR + Equalized Odds)
│
└── README.md
---

## 🧪 Scripts

### 1. `data_quality.py` — Qualidade de dados
Verifica três problemas críticos antes de alimentar um modelo:
- **Valores nulos** → classifica por severidade (OK / ALERTA / CRÍTICO)
- **Duplicatas** → detecta linhas repetidas que inflam métricas
- **Outliers** → usa o método IQR (intervalo interquartil) para identificar valores extremos

### 2. `drift_monitor.py` — Monitoramento de drift
Detecta quando os dados de produção divergem dos dados de treino usando duas abordagens:
- **PSI (Population Stability Index)** → métrica padrão do mercado financeiro. PSI ≥ 0.25 indica mudança severa
- **Teste KS (Kolmogorov-Smirnov)** → teste estatístico que verifica se duas amostras vêm da mesma distribuição

### 3. `bias_check.py` — Detecção de viés e fairness
Mede se o modelo trata grupos diferentes de forma equitativa:
- **Disparate Impact Ratio (DIR)** → razão entre taxas de aprovação de grupos. DIR < 0.80 indica possível discriminação (regra dos 4/5)
- **Equalized Odds** → compara TPR e FPR entre grupos para identificar onde o modelo erra de forma desigual

---

## 🛠️ Tecnologias

- Python 3.10+
- pandas
- numpy
- scipy

---

## 👤 Autor

**Luiz Ewerton**  
Analista de IA | Foco em estatística aplicada e governança de modelos  
[GitHub](https://github.com/ewertonlna-cloud)
