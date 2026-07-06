# FASE 2 - Otimizacao de Modelos de Diagnostico para Saude da Mulher

Este projeto continua a FASE_1 e adiciona:

- Otimizacao de hiperparametros com Algoritmos Geneticos (GA)
- Integracao com LLM para interpretacao de previsoes
- Persistencia de artefatos para uso na Fase 3

## Etapa 1 (concluida)

- Estrutura base de pastas criada
- `requirements.txt` configurado
- Instrucoes de ambiente virtual e instalacao adicionadas

## Estrutura de pastas

```text
FASE_2/
  GA/
  LLM/
  notebooks/
  models/
  results/
  llm_outputs/
  docs/
  tests/
  PLANO.md
  README.md
  requirements.txt
```

## Setup do ambiente (venv)

No Windows (PowerShell):

```powershell
cd FASE_2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No Linux/macOS:

```bash
cd FASE_2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dependencias da Fase 2

- deap
- joblib
- openai
- transformers
- accelerate
- pandas
- scikit-learn
- numpy
- openpyxl
- jupyter

## Proximas etapas

Pendencias, checklist de entrega e backlog tecnico estao consolidados em `TASKS.md`.

## Etapa 2 (persistencia e baseline)

Comandos para exportar e validar os artefatos baseline com base no notebook da Fase 1:

```powershell
& .\.venv\Scripts\python.exe .\models\export_baselines.py
& .\.venv\Scripts\python.exe .\models\validate_baselines.py
```

Artefatos gerados em `models/`:

- `baseline_lr.joblib`
- `baseline_knn.joblib`
- `baseline_dt.joblib`
- `baseline_metadata.json`

## Etapa 3 (otimizacao com algoritmo genetico)

Comando para executar os 3 experimentos de GA:

```powershell
& .\.venv\Scripts\python.exe .\GA\ga_optimizer.py
```

Saidas principais:

- `results/ga_experiments.csv`
- `results/ga_summary.json`
- `models/ga_high_mut_uniform.joblib`
- `models/ga_low_mut_onepoint.joblib`
- `models/ga_balanced_config.joblib`

## Etapa 4 (integracao com LLMs)

Comando em modo local (offline) para gerar explicacoes:

```powershell
& .\.venv\Scripts\python.exe .\LLM\llm_integration.py --force-local --sample-count 8
```

Comando usando API OpenAI (quando `OPENAI_API_KEY` estiver configurada):

```powershell
& .\.venv\Scripts\python.exe .\LLM\llm_integration.py --sample-count 8 --openai-model gpt-4.1-mini
```

Saidas geradas em `llm_outputs/`:

- `llm_responses_<timestamp>.csv`
- `llm_responses_<timestamp>.jsonl`
- `llm_qualitative_matrix_<timestamp>.csv`
