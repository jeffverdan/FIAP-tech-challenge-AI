# Plano e Tasks — FASE_2

Consolida em um único documento o plano de ação original, o checklist de entrega da avaliação FIAP e o backlog de melhorias técnicas do projeto.

## Legenda

| Símbolo | Significado |
|---|---|
| `[ ]` | Pendente |
| `[x]` | Concluída |
| `[-]` | Cancelada / descartada |

---

## Objetivo da Fase 2

Otimizar os modelos da Fase 1 utilizando:

- Algoritmos Genéticos (GA) para tuning de hiperparâmetros
- Integração com LLMs para interpretação clínica das previsões
- Persistência de artefatos para uso na Fase 3

---

## Etapas do plano (concluídas)

| Etapa | Entrega | Status |
|---|---|---|
| 1 — Setup e estrutura base | venv, `requirements.txt`, estrutura de pastas | ✅ |
| 2 — Persistência e baseline | `models/export_baselines.py`, `models/validate_baselines.py`, `baseline_*.joblib` | ✅ |
| 3 — Otimização com GA | `GA/ga_optimizer.py`, 3 experimentos, `results/ga_experiments.csv` | ✅ |
| 4 — Integração com LLMs | `LLM/llm_integration.py`, respostas em `llm_outputs/` | ✅ |

Detalhes de implementação (codificação de genes, operadores genéticos, função fitness, pipeline de prompt engineering) estão documentados em [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md).

---

## 🔴 Crítico — entregáveis obrigatórios da avaliação

### T01 — Relatório técnico
- [x] `docs/relatorio_tecnico.md` com implementação do GA, resultados, integração LLM, prompts, avaliação qualitativa e comparativo baseline vs. otimizado

### T02 — Notebook de demonstração
- [x] `notebooks/01_demonstracao.ipynb` com execução do GA, gráficos comparativos e demonstração da integração LLM

### T03 — Vídeo de demonstração
- [x] Gravar vídeo (até 15 min, YouTube/Vimeo) demonstrando GA + LLM e resultados — https://youtu.be/OXV5hpyWxjs
- [x] Adicionar link do vídeo no `README.md` (raiz e Fase 2)

---

## 🟠 Importante — pode ser questionado na avaliação

### T04 — Executar com LLM real (não fallback local)
- [x] Configurar `OPENAI_API_KEY` válida (via `.env`)
- [x] Rodar `LLM/llm_integration.py --openai-model gpt-4.1-mini` com API real (outputs `openai_api` em `llm_outputs/`)
- [x] Outputs reais de LLM presentes em `llm_outputs/` (0 fallback)
- [x] Registrar explicitamente no relatório que a integração foi executada com LLM real (seção 6.2)
- **Referência:** Requisito 3 — LLM pré-treinada integrada (GPT, Falcon, LLaMA etc.)

---

## 🟡 Necessário — requisitos explícitos ainda não atendidos

### T05 — Logging estruturado
- [x] Substituir todos os `print()` por `logging.info()` / `.warning()` / `.error()` em `GA/ga_optimizer.py`, `LLM/llm_integration.py`, `models/export_baselines.py`
- [x] Configurar nível de log via variável de ambiente `LOG_LEVEL`
- [x] Garantir que tempo de execução e métricas fiquem registrados em log
- **Referência:** Requisito 2 — Monitoramento e logging adequados para tracking de desempenho

### T06 — Documentação de arquitetura
- [x] Criar `docs/arquitetura.md` com diagrama/descrição dos componentes (GA, LLM, models, results)
- [x] Documentar decisões de implementação (ex: por que regressão logística no GA, escolha do DEAP, modo fallback)
- [x] Descrever fluxo de dados: dataset → baseline → GA → LLM → outputs
- **Referência:** Requisito 2 — Documentar arquitetura e decisões de implementação

---

## Backlog técnico (melhorias além do escopo mínimo)

### Alta prioridade

- **Testes do fitness do GA** — criar `tests/test_ga_fitness.py`: cálculo de recall/especificidade/F1 isolado, penalidade de disparidade demográfica com dados sintéticos, elitismo (top 2 preservados), early stopping (patience 4-5 gerações). Alvo: `GA/ga_optimizer.py`
- **Testes do pipeline LLM** — criar `tests/test_llm_integration.py`: anonimização (nenhum PII sobrevive), flag `mode: "local_fallback"` no modo offline, consistência do scoring qualitativo. Alvo: `LLM/llm_integration.py`
- **Testes de persistência de modelos** — criar `tests/test_model_persistence.py`: `.joblib` recarregado reproduz as mesmas predições, consistência de `baseline_metadata.json`, `validate_baselines.py` como parte da suite. Alvo: `models/`
- **Centralizar configuração** — criar `config.py` na raiz: caminho do dataset, `RANDOM_STATE`, paths de saída, parâmetros de GA, suporte a `.env` para a chave OpenAI. Substituir hardcoded paths em `GA/ga_optimizer.py`, `LLM/llm_integration.py`, `models/export_baselines.py`
- **Validação de schema do dataset** — função `validate_dataset_schema(df)` em `models/export_baselines.py`, verificando colunas/tipos esperados e emitindo erro claro se o arquivo estiver ausente ou mal formatado
- **Distinguir fallback local vs. API real no output do LLM** — campo `source` (`"openai_api"` / `"local_fallback"`) nos outputs, aviso visível no CSV/JSONL e warning no console quando em modo fallback. Alvo: `LLM/llm_integration.py`

### Média prioridade

- **Notebook de análise comparativa** — `notebooks/01_resultados_comparativos.ipynb`: tabela + gráfico de barras baseline vs. GA, curvas ROC, confusion matrices lado a lado, disparidade demográfica por faixa etária antes/depois do GA
- **Expandir métricas de fairness no GA** — disparidade de recall por faixa de IMC, taxa de falsos negativos como métrica secundária, pesos do fitness configuráveis via `config.py`. Alvo: `GA/ga_optimizer.py`

### Baixa prioridade / futuro

- **Random Forest / XGBoost no GA** — suporte a `RandomForestClassifier` e `XGBClassifier` no gene encoding, espaço de hiperparâmetros próprio, comparação com os experimentos de regressão logística existentes
- **API REST para inferência** — `api/main.py` com FastAPI (`POST /predict`, `POST /explain`), Dockerfile, `fastapi`/`uvicorn` no `requirements.txt`
- **CI/CD com GitHub Actions** — `.github/workflows/tests.yml` rodando a suite a cada push/PR em `main`/`develop`, badge de status no README, cache de dependências
- **Interface Streamlit (opcional)** — `app_streamlit.py`: entrada de dados, carregamento do modelo otimizado, exibição das explicações da LLM
- **Infraestrutura cloud (opcional)** — `docs/cloud_architecture.md` documentando deploy via Terraform/ARM/AWS/Azure

---

## Progresso geral

| Categoria | Total | Concluídas |
|---|---|---|
| 🔴 Crítico | 3 | 3 |
| 🟠 Importante | 1 | 1 |
| 🟡 Necessário | 2 | 2 |
| Backlog técnico | 11 | 0 |
