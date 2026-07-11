# FASE 2 - Otimizacao de Modelos de Diagnostico para Saude da Mulher

Este projeto continua a FASE_1 e adiciona:

- Otimizacao de hiperparametros com Algoritmos Geneticos (GA)
- Integracao com LLM para interpretacao de previsoes
- Persistencia de artefatos para uso na Fase 3

## Video de demonstracao

[Assistir no YouTube](https://youtu.be/OXV5hpyWxjs) — demonstracao do GA, resultados comparativos e integracao com LLM.

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

## Documentacao

- `docs/relatorio_tecnico.md` — relatorio tecnico completo (GA, resultados, LLM, prompts)
- `docs/arquitetura.md` — arquitetura, componentes, fluxo de dados e decisoes de implementacao

## Configuracao (.env)

A integracao com a OpenAI le a chave da variavel `OPENAI_API_KEY`. Para nao expor a chave, copie o template e preencha:

```powershell
Copy-Item .env.example .env
# edite .env e coloque sua OPENAI_API_KEY
```

O arquivo `.env` (na pasta `FASE_2/`) e carregado automaticamente por `LLM/llm_integration.py` via `python-dotenv` e e ignorado pelo git. Sem `.env`/chave, o script roda em modo fallback local.

## Logging

Todos os scripts usam `logging` estruturado. O nivel e controlado pela variavel de ambiente `LOG_LEVEL` (default `INFO`), tambem configuravel no `.env`:

```powershell
$env:LOG_LEVEL = "DEBUG"; & .\.venv\Scripts\python.exe .\GA\ga_optimizer.py
```

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
