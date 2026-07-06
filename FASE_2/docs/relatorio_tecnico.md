# Relatório Técnico — FASE 2
## Otimização de Modelos de Diagnóstico para Saúde da Mulher (PCOS)

**Grupo:** 62  
**Programa:** Pós-Graduação em Inteligência Artificial — FIAP  
**Fase:** 2 — Otimização e Integração com LLMs

---

## 1. Introdução

Este relatório documenta a segunda fase do projeto de diagnóstico de Síndrome dos Ovários Policísticos (PCOS). Na Fase 1, foram desenvolvidos modelos de classificação baseline (Regressão Logística, KNN e Árvore de Decisão) sobre o dataset `PCOS_data_without_infertility.xlsx`. A Fase 2 tem dois objetivos principais:

1. **Otimização dos modelos** via Algoritmos Genéticos (GA) para maximizar métricas de desempenho clínico e equidade demográfica.
2. **Integração com LLMs** para geração de explicações clínicas em linguagem natural a partir das predições.

---

## 2. Dataset

- **Fonte:** `PCOS_data_without_infertility.xlsx`, planilha `Full_new` (FASE 1)
- **Target:** `PCOS (Y/N)` — classificação binária (1 = PCOS, 0 = Sem PCOS)
- **Pré-processamento:**
  - Remoção de colunas administrativas: `Sl. No`, `Patient File No.`, `Unnamed: 44`
  - Coerção de todas as colunas para numérico
  - Imputação de valores ausentes pela mediana (`SimpleImputer`)
- **Divisão treino/teste:** 80%/20%, estratificada pela classe (`random_state=42`)

---

## 3. Modelos Baseline (Fase 1)

Os três modelos exportados da Fase 1 e utilizados como referência:

| Modelo | F1-Score (Teste) | Configuração |
|---|---|---|
| Regressão Logística (LR) | **0.8219** | Padrão scikit-learn |
| KNN | 0.7119 | k=7 (melhor k encontrado) |
| Árvore de Decisão (DT) | 0.7246 | max_depth=4 |

A Regressão Logística apresentou o melhor desempenho e foi escolhida como modelo-alvo para otimização via GA, por ter hiperparâmetros contínuos e discretos bem definidos para codificação genética.

---

## 4. Otimização via Algoritmos Genéticos

### 4.1 Biblioteca e framework

Foi utilizada a biblioteca **DEAP** (Distributed Evolutionary Algorithms in Python), que fornece primitivas de seleção, cruzamento e mutação configuráveis por toolbox.

### 4.2 Codificação dos genes (representação)

Cada indivíduo é representado por um vetor de 4 genes:

| Gene | Parâmetro | Espaço de busca | Decodificação |
|---|---|---|---|
| `gene_log10_c` | Regularização C | `[-2.0, 1.0]` (contínuo) | `C = 10^gene` → espaço logarítmico |
| `gene_penalty` | Tipo de penalidade | `{0, 1}` (binário) | `0 → l1`, `1 → l2` |
| `gene_class_weight` | Peso de classes | `{0, 1}` (binário) | `0 → None`, `1 → "balanced"` |
| `gene_threshold` | Limiar de decisão | `[0.3, 0.7]` (contínuo) | Threshold aplicado sobre `predict_proba` |

A codificação de `C` em escala logarítmica foi uma decisão deliberada: a regularização varia em ordens de magnitude e uma escala linear produziria um espaço de busca desbalanceado.

### 4.3 Função Fitness

A função fitness combina quatro métricas calculadas via validação cruzada estratificada (5-fold):

```
fitness = 0.4 × recall
        + 0.2 × specificity
        + 0.3 × F1-score
        − 0.1 × disparity_demográfica
```

**Justificativa dos pesos:**
- **Recall (40%):** Em diagnóstico médico, falsos negativos (não detectar PCOS quando existe) têm custo clínico elevado — o modelo deve priorizar sensibilidade.
- **F1-score (30%):** Equilíbrio entre precisão e recall, relevante para uso clínico real.
- **Especificidade (20%):** Evita alarmes falsos excessivos.
- **Disparidade demográfica (−10%):** Penaliza modelos que performam de forma desigual entre faixas etárias (abaixo/acima da mediana de idade), promovendo equidade.

A **disparidade demográfica** é calculada como a diferença absoluta de recall entre dois grupos etários:

```python
disparity = |recall(grupo_idade ≤ mediana) − recall(grupo_idade > mediana)|
```

### 4.4 Operadores genéticos

| Operador | Implementação |
|---|---|
| **Seleção** | Torneio com tamanho 3 (`selTournament`) |
| **Cruzamento uniforme** | `cxUniform` com probabilidade de troca por gene = 0.5 |
| **Cruzamento um ponto** | `cxOnePoint` |
| **Mutação** | Operador customizado por tipo de gene: perturbação contínua (±0.4 para C, ±0.1 para threshold) ou inversão binária (penalty, class_weight) |
| **Elitismo** | 2 melhores indivíduos preservados por geração |
| **Early stopping** | Encerra se não houver melhora de fitness por N gerações (patience configurável) |

### 4.5 Experimentos realizados

Foram conduzidos 3 experimentos com configurações distintas:

| Parâmetro | high_mut_uniform | low_mut_onepoint | balanced_config |
|---|---|---|---|
| População | 100 | 100 | 100 |
| Gerações máx. | 20 | 10 | 20 |
| Taxa de mutação (`mutpb`) | **0.50** | **0.20** | 0.30 |
| Taxa de cruzamento (`cxpb`) | 0.70 | **0.80** | 0.75 |
| Tipo de cruzamento | Uniforme | Um ponto | Uniforme |
| `mutation_indpb` | 0.4 | 0.2 | 0.3 |
| Early stop patience | 4 | 5 | 5 |

### 4.6 Resultados dos experimentos

| Experimento | Fitness | Recall | Especificidade | F1 | Disparidade | Tempo (s) |
|---|---|---|---|---|---|---|
| high_mut_uniform | **0.7814** | 0.8611 | 0.9452 | **0.8732** | 0.0065 | 66.4 |
| low_mut_onepoint | 0.7785 | 0.8611 | 0.9315 | 0.8611 | 0.0065 | 66.2 |
| balanced_config | 0.7797 | 0.8611 | 0.9452 | **0.8732** | 0.0065 | 68.6 |

**Melhor experimento por fitness:** `high_mut_uniform`

**Hiperparâmetros ótimos encontrados (high_mut_uniform):**
- `C ≈ 0.056` (regularização forte)
- `penalty = l1`
- `class_weight = None`
- `threshold ≈ 0.468`

---

## 5. Comparativo de Desempenho: Baseline vs. Otimizado

| Métrica | Baseline LR | GA (melhor) | Delta |
|---|---|---|---|
| **F1-Score** | 0.8219 | **0.8732** | **+5,13%** |
| **Recall** | 0.8333 | **0.8611** | +2,78% |
| **Especificidade** | 0.9041 | **0.9452** | +4,11% |
| **Disparidade demográfica** | 0.0390 | **0.0065** | **−83,3%** |

**Destaques:**
- O modelo otimizado superou o baseline em todas as métricas.
- A redução de **83,3% na disparidade demográfica** é o ganho mais expressivo — o modelo otimizado trata grupos etários com muito maior equidade.
- O F1-Score passou de 0.822 para 0.873, uma melhora clinicamente relevante.

---

## 6. Integração com LLMs para Interpretação de Resultados

### 6.1 Objetivo

Integrar um modelo de linguagem para gerar explicações clínicas em linguagem natural a partir das predições do modelo otimizado, tornando os diagnósticos compreensíveis para profissionais de saúde.

### 6.2 Modelo de linguagem utilizado

A integração foi implementada com suporte dual:

- **Modo principal:** OpenAI API (`gpt-4.1-mini`) — ativado quando `OPENAI_API_KEY` está configurada
- **Modo fallback local:** Geração de texto estruturado a partir de templates determinísticos, usado quando a API não está disponível

### 6.3 Pipeline de integração

```
Dataset (FASE_1)
      │
      ▼
  Conjunto de teste (20%)
      │
      ▼
  Modelo GA otimizado (.joblib)
      │
      ├─── predict_proba → probabilidade de PCOS
      ├─── threshold → predição binária
      └─── coeficientes × valores normalizados → top-5 features locais
      │
      ▼
  Anonimização do paciente (remoção de PII)
      │
      ▼
  Construção do prompt (prompt engineering)
      │
      ▼
  LLM (OpenAI API ou fallback local)
      │
      ▼
  Avaliação qualitativa (rubrica heurística)
      │
      ▼
  Saída: CSV + JSONL + matriz qualitativa
```

### 6.4 Prompt Engineering

O prompt foi elaborado com as seguintes estratégias:

**Contexto clínico explícito:** O modelo recebe o papel de "assistente clínico para saúde da mulher", orientando o tom e o vocabulário das respostas.

**Dados estruturados anonimizados:** Os dados clínicos do paciente são incluídos como JSON, garantindo que o modelo processe informações numéricas de forma contextualizada.

**Importância local de features:** As top-5 variáveis com maior contribuição para a predição (calculadas por `coeficiente × valor normalizado`) são incluídas no prompt, orientando o LLM a focar nos fatores mais relevantes do caso específico.

**Contexto social e de comunicação:** O prompt especifica faixa etária ("adulta jovem" / "adulta"), necessidade de linguagem "clara, acolhedora e não alarmista", e exigência de confidencialidade.

**Estrutura de resposta obrigatória:** O LLM é instruído a responder em 4 seções fixas:
1. Interpretação do caso
2. Fatores relevantes
3. Recomendações práticas
4. Confidencialidade e limites do modelo

**Exemplo de prompt gerado:**

```
Você é um assistente clínico para saúde da mulher.

Contexto:
- Paciente anonimizada: P_001
- Probabilidade de PCOS prevista pelo modelo: 0.743
- Predição binária do modelo (1=PCOS, 0=Não PCOS): 1

Dados clínicos resumidos (anonimizados):
{"Age (yrs)": 28.0, "BMI": 24.5, "FSH(mIU/mL)": 5.2, ...}

Principais features (contribuição local para a predição):
[["Follicle No. (R)", 0.84], ["AMH(ng/mL)", 0.71], ...]

Contexto social e de comunicação:
{"faixa_etaria": "adulta jovem", "necessidade_linguagem": "clara, acolhedora e não alarmista", ...}

Tarefas:
1) Explique a predição em linguagem natural para profissional de saúde.
2) Explique os fatores mais relevantes do caso.
3) Traga recomendações práticas e próximos passos clínicos sugeridos.
4) Inclua um aviso de confidencialidade e limitação do modelo.
5) Use linguagem sensível ao contexto de saúde da mulher.

Responda em português com 4 seções: ...
```

### 6.5 Avaliação da qualidade das interpretações

A qualidade das respostas foi avaliada por uma rubrica heurística automatizada com 3 dimensões (escala 1–5):

| Dimensão | Critério de avaliação |
|---|---|
| **Precisão médica** | Presença de termos como "critérios clínicos", "clínic*" |
| **Sensibilidade cultural** | Presença de "saúde da mulher", "acolhedora", "sensível" |
| **Adequação da linguagem** | Presença simultânea de "recomendações" e "limites" |
| **Overall** | Média arredondada das três dimensões |

**Resultados de avaliação (8 pacientes amostrados):**

| Paciente | Precisão Médica | Sensibilidade Cultural | Adequação Linguística | Overall |
|---|---|---|---|---|
| P_001 a P_008 | 4/5 | 4/5 | 4/5 | **4/5** |

Todos os 8 casos avaliados obtiveram nota geral **4/5**, indicando que as respostas geradas atendem consistentemente aos critérios de qualidade definidos.

---

## 7. Desafios Enfrentados e Soluções Implementadas

### 7.1 Espaço de busca heterogêneo no GA

**Desafio:** Os hiperparâmetros da Regressão Logística têm naturezas distintas — `C` é contínuo em escala logarítmica, `penalty` e `class_weight` são categóricos binários, e `threshold` é contínuo em faixa restrita.

**Solução:** Implementação de um operador de mutação customizado (`mutation_operator`) que aplica perturbações adequadas a cada tipo de gene, em vez de usar mutação genérica. Genes contínuos recebem perturbação gaussiana; genes binários são invertidos.

### 7.2 Avaliação de fitness computacionalmente custosa

**Desafio:** Cada avaliação de fitness executa validação cruzada 5-fold com retreinamento completo do modelo — o que torna a avaliação de 100 indivíduos por geração lenta.

**Solução:** Implementação de **early stopping** baseado em platô de fitness (patience de 4–5 gerações sem melhora), que encerrou os experimentos antes de atingir o limite máximo de gerações, reduzindo o tempo médio para ~67 segundos por experimento.

### 7.3 Equidade demográfica como objetivo

**Desafio:** Modelos de ML tendem a otimizar métricas globais, podendo criar disparidades entre subgrupos populacionais.

**Solução:** A disparidade demográfica foi incorporada diretamente na função fitness como penalidade, forçando o GA a encontrar soluções que equilibrem desempenho e equidade. O resultado foi uma redução de 83,3% na disparidade de recall entre faixas etárias.

### 7.4 Disponibilidade da API de LLM

**Desafio:** A dependência de uma API externa (OpenAI) introduz fragilidade ao sistema — indisponibilidade de rede, ausência de chave, ou custos podem interromper o fluxo.

**Solução:** Implementação de um modo **fallback local** com geração de texto estruturado por templates, garantindo que o pipeline funcione offline e que os outputs tenham formato consistente independentemente do modo de execução.

---

## 8. Arquitetura da Solução

```
FASE_1/
└── data/PCOS_data_without_infertility.xlsx   ← Dataset original

FASE_2/
├── models/
│   ├── export_baselines.py     ← Exporta modelos baseline da FASE_1
│   ├── validate_baselines.py   ← Valida artefatos persistidos
│   ├── baseline_lr.joblib      ← Modelo baseline (Regressão Logística)
│   ├── baseline_knn.joblib     ← Modelo baseline (KNN)
│   ├── baseline_dt.joblib      ← Modelo baseline (Árvore de Decisão)
│   ├── baseline_metadata.json  ← Métricas e configurações dos baselines
│   ├── ga_high_mut_uniform.joblib   ← Modelo GA (melhor fitness)
│   ├── ga_low_mut_onepoint.joblib    ← Modelo GA (baixa mutação)
│   └── ga_balanced_config.joblib      ← Modelo GA (configuração balanceada)
│
├── GA/
│   └── ga_optimizer.py         ← Algoritmo Genético (DEAP)
│
├── LLM/
│   └── llm_integration.py      ← Integração LLM + prompt engineering + avaliação
│
├── results/
│   ├── ga_experiments.csv      ← Resultados completos dos 3 experimentos
│   └── ga_summary.json         ← Sumário do melhor experimento
│
└── llm_outputs/
    ├── llm_responses_*.csv              ← Respostas por paciente
    ├── llm_responses_*.jsonl            ← Registros com prompt + resposta
    └── llm_qualitative_matrix_*.csv     ← Scores de qualidade
```

**Fluxo de dados:**

```
Dataset → Pré-processamento → Divisão treino/teste
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                     │
             Exportar baselines                    GA Optimizer
             (export_baselines.py)            (ga_optimizer.py)
                    │                                     │
             baseline_*.joblib              ga_*.joblib + results/
                    │                                     │
                    └─────────────────┬─────────────────┘
                                      │
                               LLM Integration
                            (llm_integration.py)
                                      │
                               llm_outputs/
```

---

## 9. Conclusão

A Fase 2 demonstrou que a aplicação de Algoritmos Genéticos para otimização de hiperparâmetros trouxe ganhos consistentes em todas as métricas avaliadas, com destaque para a redução expressiva da disparidade demográfica — um aspecto crítico em sistemas de apoio à decisão clínica.

A integração com LLM adicionou uma camada de interpretabilidade que transforma predições numéricas em explicações acionáveis para profissionais de saúde, com qualidade avaliada sistematicamente por rubrica heurística.

Os artefatos gerados (modelos `.joblib`, resultados em CSV/JSON/JSONL) estão estruturados para uso direto na Fase 3, que expandirá o sistema com dados textuais.

---

*Documento criado em junho de 2026 — Tech Challenge, FIAP.*
