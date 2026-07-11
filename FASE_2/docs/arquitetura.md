# Arquitetura da Solução — FASE 2

Documento de arquitetura do projeto de otimização de modelos de diagnóstico de PCOS
(Síndrome dos Ovários Policísticos). Descreve os componentes, o fluxo de dados, as
decisões de implementação e as práticas de observabilidade adotadas na Fase 2.

Para o detalhamento técnico dos experimentos e resultados, consulte
[`relatorio_tecnico.md`](relatorio_tecnico.md).

---

## 1. Visão geral

A Fase 2 evolui os modelos clássicos da Fase 1 adicionando duas camadas:

1. **Otimização de hiperparâmetros via Algoritmo Genético (GA)**, usando a biblioteca
   DEAP, com uma função de fitness multiobjetivo que equilibra desempenho preditivo e
   equidade demográfica.
2. **Interpretação das previsões via LLM**, transformando as saídas numéricas do modelo
   em explicações clínicas em linguagem natural, com modo de fallback local para
   operação offline.

Cada fase é um projeto autocontido. A comunicação entre componentes acontece por
**artefatos versionados em disco** (`.joblib`, `.csv`, `.json`, `.jsonl`), não por
imports diretos — o que mantém baixo acoplamento e permite reexecutar cada etapa de
forma independente.

---

## 2. Componentes

| Componente | Arquivo | Responsabilidade |
|---|---|---|
| **Exportação de baselines** | `models/export_baselines.py` | Reproduz o pré-processamento da Fase 1, treina os modelos baseline (LR, KNN, Árvore de Decisão) e persiste os artefatos `.joblib` + metadados. |
| **Validação de baselines** | `models/validate_baselines.py` | Recarrega os artefatos persistidos e valida consistência das predições/metadados. |
| **Otimizador Genético** | `GA/ga_optimizer.py` | Executa os experimentos de GA (DEAP) para otimizar a Regressão Logística, salva os modelos otimizados e os resultados comparativos. |
| **Integração LLM** | `LLM/llm_integration.py` | Carrega o modelo otimizado, anonimiza os dados da paciente, monta o prompt, chama a LLM (OpenAI) ou o fallback local e avalia a qualidade da resposta. |
| **Notebook de demonstração** | `notebooks/01_demonstracao.ipynb` | Executa e visualiza o fluxo GA + LLM com gráficos comparativos. |

### Diretórios de artefatos

```
FASE_1/
└── data/PCOS_data_without_infertility.xlsx   ← Dataset original (fonte única)

FASE_2/
├── models/                 ← Modelos baseline e otimizados (.joblib) + metadados
│   ├── baseline_lr.joblib
│   ├── baseline_knn.joblib
│   ├── baseline_dt.joblib
│   ├── baseline_metadata.json
│   ├── ga_high_mut_uniform.joblib     ← melhor fitness
│   ├── ga_low_mut_onepoint.joblib
│   └── ga_balanced_config.joblib
├── GA/ga_optimizer.py
├── LLM/llm_integration.py
├── results/                ← Saídas do GA
│   ├── ga_experiments.csv
│   └── ga_summary.json
└── llm_outputs/            ← Saídas da integração LLM
    ├── llm_responses_*.csv
    ├── llm_responses_*.jsonl
    └── llm_qualitative_matrix_*.csv
```

---

## 3. Fluxo de dados

```
Dataset (PCOS .xlsx)
        │
        ▼
Pré-processamento (drop de colunas, coerção numérica, imputação por mediana)
        │
        ▼
Divisão treino/teste (train_test_split estratificado, RANDOM_STATE=42)
        │
        ├──────────────────────────┬───────────────────────────┐
        ▼                          ▼                           │
Exportar baselines           GA Optimizer                      │
(export_baselines.py)        (ga_optimizer.py)                 │
        │                          │                           │
 baseline_*.joblib          ga_*.joblib + results/             │
        │                          │                           │
        └──────────────┬───────────┘                           │
                       ▼                                       │
              Modelo otimizado selecionado ◄───────────────────┘
                       │
                       ▼
              Integração LLM (llm_integration.py)
              (anonimização → prompt → LLM/fallback → rubrica)
                       │
                       ▼
                  llm_outputs/
```

A reprodutibilidade é garantida por `RANDOM_STATE = 42` fixo em todos os scripts e pela
imputação por mediana idêntica à da Fase 1, de modo que a mesma divisão treino/teste é
obtida de forma determinística em cada componente.

---

## 4. Decisões de implementação

### 4.1 Por que Regressão Logística no GA

Entre os baselines da Fase 1, a Regressão Logística foi escolhida como alvo da
otimização por combinar bom desempenho com **interpretabilidade** — seus coeficientes
permitem calcular a contribuição local de cada feature, insumo direto para as
explicações da LLM. Modelos de ensemble (Random Forest, XGBoost) ficaram registrados no
backlog como extensão futura.

### 4.2 Por que DEAP

O DEAP oferece controle explícito sobre codificação de genes, operadores genéticos e o
laço evolutivo, o que era necessário porque o espaço de busca é **heterogêneo**: `C`
(contínuo em escala log), `penalty` e `class_weight` (categóricos binários) e `threshold`
(contínuo restrito). Isso motivou um operador de mutação customizado (`mutation_operator`)
que aplica perturbação gaussiana a genes contínuos e inversão a genes binários.

### 4.3 Função de fitness multiobjetivo

O fitness combina recall (0,4), especificidade (0,2) e F1 (0,3) e **penaliza a
disparidade demográfica** (0,1) de recall entre faixas etárias. A equidade entra
diretamente como objetivo de otimização, e não como métrica de auditoria posterior.

### 4.4 Early stopping

Cada avaliação de fitness roda validação cruzada 5-fold com retreino completo, o que é
custoso para 100 indivíduos por geração. Um early stopping por platô de fitness
(paciência de 4–5 gerações) encerra os experimentos ao estabilizar, reduzindo o tempo
médio por experimento.

### 4.5 Modo fallback local na LLM

A dependência de uma API externa introduz fragilidade (rede, ausência de chave, custo).
O pipeline detecta a presença de `OPENAI_API_KEY` e a flag `--force-local`, registra o
modo em execução (`openai_api` ou `local_fallback`) e, na ausência da API, gera
explicações estruturadas por template. O campo `mode` é gravado em todos os outputs
(CSV/JSONL), tornando explícito como cada resposta foi produzida.

### 4.6 Privacidade

Antes de montar o prompt, `anonymize_patient_row` remove identificadores diretos
(`Patient File No.`, `Sl. No`), mantendo apenas o contexto clínico e sociodemográfico
necessário à interpretação.

---

## 5. Observabilidade (logging)

Todos os scripts usam o módulo `logging` da biblioteca padrão em vez de `print()`, com
formato uniforme:

```
%(asctime)s | %(levelname)-8s | %(name)s | %(message)s
```

- Cada componente tem um logger nomeado: `fase2.ga`, `fase2.llm`, `fase2.baselines`.
- O nível é controlado pela variável de ambiente **`LOG_LEVEL`** (default `INFO`);
  valores válidos: `DEBUG`, `INFO`, `WARNING`, `ERROR`.
- Eventos de degradação (early stopping, uso de fallback local, falha de chamada de API)
  são registrados como `WARNING`.
- Tempo de execução e métricas de teste de cada experimento ficam registrados em log,
  atendendo ao requisito de monitoramento e tracking de desempenho.

Exemplo de execução com nível ajustado:

```bash
LOG_LEVEL=WARNING python LLM/llm_integration.py --force-local --sample-count 8
```

---

## 6. Execução

```bash
# 1. Exportar baselines da Fase 1
python models/export_baselines.py

# 2. Otimizar via Algoritmo Genético (3 experimentos)
python GA/ga_optimizer.py

# 3. Gerar interpretações via LLM (modo local/offline)
python LLM/llm_integration.py --force-local --sample-count 8

# 3b. Gerar interpretações com a API OpenAI (requer OPENAI_API_KEY)
python LLM/llm_integration.py --sample-count 8 --openai-model gpt-4.1-mini
```

Consulte o [`README.md`](../README.md) para o setup do ambiente virtual e dependências.

---

*Documento de arquitetura — Tech Challenge FASE 2, FIAP.*
