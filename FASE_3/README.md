# FASE 3 — Assistente Virtual Médico com LLM Customizada, LangChain e LangGraph

Tech Challenge da Pós-graduação em IA para Devs (FIAP) — **Grupo 62**.

Continuação das Fases 1 e 2: o mesmo problema clínico (**Síndrome dos Ovários Policísticos**)
sobe uma camada — de modelos preditivos para um **assistente conversacional de apoio à
decisão**, treinado com os documentos internos de um hospital e operando dentro de um fluxo
de decisão automatizado, auditável e com validação humana obrigatória.

> ⚠️ **Todos os dados são sintéticos.** O "Hospital Aurora", seus protocolos, prontuários,
> exames e pacientes foram gerados para este trabalho acadêmico. Nenhum dado real de paciente
> foi utilizado. O sistema não é um dispositivo médico e não substitui julgamento clínico.

---

## O que foi construído

| Entrega do enunciado | Onde está |
| --- | --- |
| Fine-tuning de LLM com dados médicos internos | `finetune/train_lora.py`, `finetune/notebook_colab.ipynb` |
| Preprocessing, anonimização e curadoria | `datagen/anonymizer.py`, `datagen/build_finetune_dataset.py` |
| Pipeline LangChain integrando a LLM customizada | `assistant/chains.py`, `assistant/llm_provider.py` |
| Consulta a base de dados estruturada | `assistant/db_tools.py`, `data/hospital.db` |
| Contextualização com dados atualizados do paciente | nó `contexto_paciente` em `assistant/graph.py` |
| Fluxos do LangGraph | `assistant/graph.py` |
| Limites de atuação (nunca prescrever sem validação) | `assistant/guardrails.py` |
| Logging detalhado para auditoria | `assistant/audit.py` → `logs/audit/*.jsonl` |
| Explainability (fonte da informação) | `assistant/retriever.py` + campo `fontes` de toda resposta |
| Dataset anonimizado / sintético | `data/processed/` + `data/processed/dataset_card.md` |
| Relatório técnico e diagrama do fluxo | `docs/relatorio_tecnico.md`, `docs/arquitetura.md` |

---

## Setup

### Windows (PowerShell)

```powershell
cd FASE_3
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env      # opcional: ajuste modelo, device e caminhos
```

### Linux / macOS

```bash
cd FASE_3
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

> **Rodar sem GPU e sem baixar pesos?** Basta `FORCE_FALLBACK=true` no `.env`. Todo o fluxo
> LangGraph — guardrails, RAG, alertas, auditoria e validação humana — funciona em modo
> determinístico offline. Veja "Modo fallback" abaixo.

---

## Execução

### 1. Gerar a base clínica sintética e o dataset

Os artefatos já vêm versionados. Para regerar do zero:

```bash
python datagen/generate_patients.py --n-pacientes 40 --seed 62
python datagen/build_finetune_dataset.py --seed 62
```

Saídas:

| Arquivo | Conteúdo |
| --- | --- |
| `data/raw/prontuarios_brutos.jsonl` | prontuários sintéticos **com PII**, para demonstrar a anonimização |
| `data/hospital.db` | base SQLite **anonimizada** — 40 pacientes, 99 consultas, 522 exames, 36 prescrições |
| `data/processed/train.jsonl` | 306 amostras de treino |
| `data/processed/val.jsonl` | 38 amostras de validação |
| `data/processed/test.jsonl` | 38 amostras de teste |
| `data/processed/documentos_rag.jsonl` | 46 trechos de protocolo indexáveis |
| `data/processed/dataset_card.md` | composição, preprocessing e limitações do dataset |

### 2. Fine-tuning (Google Colab, GPU T4)

Abra `finetune/notebook_colab.ipynb` no Colab, selecione GPU T4 e execute as células.
O notebook clona o repositório, roda um smoke test, treina, avalia e exporta o adapter.

> `.ipynb` é um notebook, não um script: ele não roda com `python arquivo.ipynb`. Abra no
> Colab (**Arquivo → Abrir notebook → GitHub**) ou, localmente, com `jupyter notebook`.

Localmente (requer GPU):

```bash
python finetune/train_lora.py --config finetune/configs/lora_tinyllama.yaml
python finetune/evaluate.py --adapter finetune/outputs/adapter --comparar-base
```

Descompacte o `adapter_fase3.zip` do Colab em `finetune/outputs/adapter` para que o
assistente carregue o modelo ajustado automaticamente.

#### Problemas conhecidos do ambiente Colab

**`ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only
versions above 0.16.0 are supported`** — surge em `get_peft_model`, logo após o download dos
pesos. O Colab traz `torchao 0.10.0` pré-instalado e o `peft` recente exige `>= 0.16`; ao
montar as camadas LoRA, o `peft` chama `is_torchao_available()`, que **levanta `ImportError`**
em vez de retornar `False` quando encontra uma versão antiga.

O `torchao` só é usado para quantização que este treino não faz, então a correção é removê-lo:

```python
!pip uninstall -y torchao
```

Não é preciso reiniciar o runtime — o treino roda em um subprocesso (`!python ...`), que nasce
já sem o pacote. A célula de instalação do notebook já faz isso. Evite `pip install -U torchao`
como alternativa: a versão 0.16+ costuma exigir um `torch` diferente do que o Colab tem.

**`TypeError: TrainingArguments.__init__() got an unexpected keyword argument 'warmup_ratio'`**
— as majors novas do `transformers` reorganizaram `TrainingArguments` e removeram parâmetros do
construtor. Por isso o notebook fixa `transformers>=4.46,<5`: é a faixa em que a configuração
descrita no relatório técnico existe inteira.

Como rede de segurança, `train_lora.py` não chama `TrainingArguments` com argumentos fixos —
`montar_training_arguments()` monta a chamada contra a **assinatura real** da classe instalada,
resolve renomeações conhecidas (`evaluation_strategy` → `eval_strategy`) e descarta o que a
versão não aceitar. Um parâmetro descartado altera o treino, então nunca some em silêncio: vai
para o log em nível `WARNING` e para o campo `training_args_ignorados` de
`metadados_treino.json`. Se esse campo vier não vazio, o treino **não** é o documentado no
relatório — reinstale dentro da faixa fixada e rode de novo.

**`ERROR: pip's dependency resolver ... gradio / diffusers requires huggingface-hub>=1.16`**
— não é falha de instalação, e sim o relatório pós-instalação do pip. Fixar `transformers<5`
rebaixa o `huggingface-hub` para a série 0.x, deixando `gradio` e `diffusers` (pré-instalados
no Colab, e não usados neste treino) com requisito insatisfeito. O comando termina com sucesso.

Depois de instalar, **reinicie a sessão** (*Ambiente de execução → Reiniciar sessão*): o kernel
mantém em memória a versão do `transformers` que já havia sido importada. Os pacotes e o
`/content` sobrevivem ao restart — ao voltar, pule a célula de instalação.

**`torch_dtype is deprecated! Use dtype instead`** — apenas um aviso do `transformers` 4.56+.
O parâmetro continua funcionando, e `torch_dtype` é o nome compatível com as versões anteriores.

**GPU diferente de T4** — o Colab pode alocar L4 ou A100 conforme a disponibilidade. O
`train_lora.py` detecta o suporte a bfloat16 e ajusta o dtype sozinho; o log informa qual foi
escolhido. Em GPU com bf16 o treino é mais rápido do que os ~20 min estimados para a T4.

### 3. Demonstração em linha de comando

```bash
python run_demo.py                    # roteiro de 6 interações usado no vídeo
python run_demo.py --fallback         # força o modo offline
python run_demo.py --paciente PAC-XXXXXXXX --json
```

### 4. Interface Streamlit

```bash
streamlit run app/streamlit_app.py
```

Quatro abas: **Assistente** (pergunta contextualizada com fontes citadas), **Painel da
paciente** (dossiê, exames pendentes e alertas), **Validação humana** (fila
`pending_human_validation`) e **Auditoria** (trilha nó a nó).

### 5. Testes

```bash
pytest -q         # 107 testes
```

Cobrem anonimização, motor de regras clínicas, guardrails, RAG, integridade do dataset e o
fluxo LangGraph ponta a ponta.

---

## Arquitetura em uma tela

```text
                   pergunta do médico + paciente_id
                                │
                          ┌─────▼─────┐
                          │  triagem  │  guardrail de entrada (intenção)
                          └─────┬─────┘
                  bloqueado ◄───┴───► segue
                       │                │
                       │        ┌───────▼────────┐
                       │        │contexto_paciente│  SQLite somente leitura
                       │        └───────┬────────┘
                       │   alerta CRÍTICO│  regras determinísticas
                       ├────────────────┤
                       │        ┌───────▼────────┐
                       │        │exames_pendentes│  clinical_rules.py
                       │        └───────┬────────┘
                       │        ┌───────▼──────────┐
                       │        │recuperar_protocolo│  RAG (FAISS ou BM25)
                       │        └───────┬──────────┘
                       │        ┌───────▼────────┐
                       │        │sugerir_conduta │  ← única chamada à LLM
                       │        └───────┬────────┘
                       │        ┌───────▼────────┐
                       │        │guardrails_saida│  dose, PII, fonte, marcação
                       │        └───────┬────────┘
              ┌────────▼────────┐       │
              │resposta_bloqueada│──────┤
              └─────────────────┘       │
                              ┌─────────▼──────┐
                              │ emitir_alertas │
                              └─────────┬──────┘
                              ┌─────────▼──────┐
                              │ fila_validacao │  pending_human_validation
                              └────────────────┘
```

Diagrama detalhado (Mermaid) em [`docs/arquitetura.md`](docs/arquitetura.md).

### Princípio de projeto: a LLM redige, as regras decidem

Todo gatilho de segurança — alerta crítico, exame pendente, checklist pré-indução, bloqueio
de prescrição — é calculado em Python determinístico (`assistant/clinical_rules.py`,
`assistant/guardrails.py`), **não** pela LLM. O modelo recebe esses fatos prontos e cuida da
redação e da contextualização. É isso que garante que um alerta crítico não deixe de disparar
por variação de amostragem, e é o que torna o comportamento testável.

---

## Segurança e governança

O `PROT-007` (em `data/raw/protocolos/`) é um protocolo institucional que governa o próprio
assistente. Ele é, ao mesmo tempo, documento de treino, fonte de RAG e especificação dos
guardrails implementados.

**O assistente não pode**, e isso é imposto por código:

- emitir prescrição, dose, via ou posologia — bloqueio na entrada + sanitização na saída;
- confirmar ou descartar diagnóstico de forma definitiva;
- alterar o prontuário — a conexão SQLite é aberta em `mode=ro`;
- responder fora do escopo clínico institucional;
- responder sem citar a fonte recuperada.

**Toda conduta** é marcada como `SUGESTÃO NÃO VALIDADA` e entra na fila
`pending_human_validation`, só sendo considerada válida após decisão nominal de um médico
(`aprovado`, `aprovado_com_ressalva` ou `rejeitado`), registrada na trilha de auditoria.

### Auditoria

`logs/audit/aurora-YYYY-MM-DD.jsonl`, append-only, **um evento por nó do grafo**, com
`interaction_id`, `timestamp_utc`, `usuario_crm`, `paciente_pseudonimo`, `no_do_grafo`,
`entrada`, `saida`, `fontes`, `guardrails_acionados`, `severidade_alerta`, `latencia_ms`,
`modelo` e `adapter_hash`. Um filtro remove qualquer campo com PII antes da gravação.

---

## Modo fallback

Seguindo o padrão da Fase 2, o assistente degrada com aviso explícito no log em vez de falhar:

| Componente | Preferencial | Fallback |
| --- | --- | --- |
| LLM | modelo base + adapter LoRA | redator determinístico extrativo, sem pesos |
| RAG | embeddings multilíngues + FAISS | Okapi BM25 em Python puro |

Guardrails, regras clínicas, alertas, auditoria e validação humana são **idênticos** nos dois
modos — não há caminho de execução que pule uma verificação de segurança.

---

## Estrutura

```text
FASE_3/
├── app/streamlit_app.py          # interface de demonstração
├── assistant/
│   ├── audit.py                  # trilha de auditoria JSONL append-only
│   ├── chains.py                 # cadeias LangChain (LCEL)
│   ├── clinical_rules.py         # motor determinístico de regras clínicas
│   ├── config.py                 # configuração via .env
│   ├── db_tools.py               # tools LangChain sobre o SQLite (somente leitura)
│   ├── graph.py                  # fluxo LangGraph + API do assistente
│   ├── guardrails.py             # limites de atuação (PROT-007)
│   ├── llm_provider.py           # LoRA/HuggingFace + fallback determinístico
│   ├── prompts.py                # system prompt institucional (fonte única)
│   └── retriever.py              # RAG: FAISS ou BM25
├── data/
│   ├── raw/protocolos/           # 7 protocolos internos sintéticos
│   ├── raw/modelos/              # laudo, receituário e POP de coleta
│   ├── raw/faq_medicos.jsonl     # 45 FAQs com fonte citada
│   ├── processed/                # dataset de fine-tuning + dataset card
│   └── hospital.db               # base clínica anonimizada
├── datagen/                      # geração, anonimização e curadoria
├── docs/                         # arquitetura e relatório técnico
├── finetune/                     # pipeline LoRA, métricas, notebook Colab
├── tests/                        # 107 testes
├── run_demo.py                   # roteiro de demonstração
├── PLANO.md
└── requirements.txt
```

---

## Vídeo de demonstração

_(link a inserir)_

## Equipe

Gabriel Pontin Buranello · Jeferson Verdan Oliveira · Josue Monteiro de Oliveira ·
Larissa Nunes da Silva · Oryange Strifezze
