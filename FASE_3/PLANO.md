# PLANO — FASE 3

## Objetivo

Construir um **assistente virtual medico** treinado com dados proprios do hospital, capaz de
auxiliar em condutas clinicas, responder duvidas de medicos e sugerir procedimentos com base
nos protocolos internos — com fluxos de decisao automatizados e seguros orquestrados por
**LangChain + LangGraph**.

O dominio clinico continua o fio condutor do repositorio: **SOP / PCOS e saude da mulher**
(Fases 1 e 2), agora na camada de suporte a decisao conversacional.

## Decisoes de arquitetura

| Tema | Decisao |
| --- | --- |
| Modelo base | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (default) ou `meta-llama/Llama-3.2-1B-Instruct` |
| Tecnica de fine-tuning | LoRA / PEFT (SFT supervisionado, formato instrucao-resposta) |
| Ambiente de treino | Google Colab (GPU T4 gratuita) — notebook versionado no repo |
| Dados | Corpus sintetico do "Hospital Aurora" (protocolos, FAQs, laudos) + subset MedQuAD |
| Runtime do assistente | Adapter LoRA local via HuggingFace, com **fallback deterministico** offline |
| Orquestracao | LangChain (chains, tools, retriever) + LangGraph (fluxo de decisao com estado) |
| Base estruturada | SQLite sintetico (`data/hospital.db`) — pacientes, exames, prescricoes, alertas |
| RAG | Embeddings + FAISS sobre os protocolos internos (fonte citada em toda resposta) |
| Interface | Streamlit (`app/streamlit_app.py`) |
| Auditoria | Log estruturado JSONL append-only em `logs/audit/` |

## Etapas

### Etapa 0 — Scaffold
- Estrutura de pastas, `requirements.txt`, `.env.example`, `PLANO.md`, `README.md`.

### Etapa 1 — Corpus medico e base estruturada
- `datagen/generate_protocols.py` — protocolos internos sinteticos versionados (markdown).
- `datagen/generate_patients.py` — prontuarios sinteticos -> `data/hospital.db` (SQLite).
- `datagen/anonymizer.py` — deteccao e mascaramento de PII (nome, CPF, telefone, email,
  endereco, RG, cartao SUS, datas de nascimento) com pseudonimizacao estavel por hash.
- `datagen/build_finetune_dataset.py` — curadoria, deduplicacao, anonimizacao, split
  train/val/test em `data/processed/*.jsonl`.

### Etapa 2 — Fine-tuning
- `finetune/train_lora.py` — pipeline SFT com PEFT/LoRA, parametrizado por YAML.
- `finetune/configs/lora_tinyllama.yaml` — hiperparametros.
- `finetune/notebook_colab.ipynb` — execucao ponta a ponta no Colab (T4).
- `finetune/evaluate.py` — avaliacao do modelo (perplexidade, ROUGE-L, taxa de recusa
  em pedidos de prescricao direta, aderencia a citacao de fonte).

### Etapa 3 — Assistente LangChain / LangGraph
- `assistant/llm_provider.py` — carrega modelo base + adapter; fallback offline.
- `assistant/retriever.py` — indexacao FAISS dos protocolos + recuperacao com metadados.
- `assistant/db_tools.py` — tools LangChain sobre o SQLite (paciente, exames pendentes,
  medicacoes ativas, historico).
- `assistant/chains.py` — chains de contextualizacao e de resposta clinica.
- `assistant/graph.py` — **LangGraph**: triagem -> contexto do paciente -> exames pendentes
  -> recuperacao de protocolo -> sugestao de conduta -> guardrails -> alertas ->
  fila de validacao humana.

### Etapa 4 — Seguranca, auditoria e explainability
- `assistant/guardrails.py` — bloqueio de prescricao direta, escopo clinico, deteccao de
  emergencia, exigencia de validacao humana, filtro de PII na saida.
- `assistant/audit.py` — trace completo por interacao (id, timestamp, no do grafo, prompt,
  fontes, decisao do guardrail, latencia, hash do modelo).
- Toda resposta carrega `fontes: [protocolo, secao, trecho]`.

### Etapa 5 — App Streamlit
- Consulta por paciente, chat clinico, painel de alertas, visualizador de trilha de auditoria.

### Etapa 6 — Documentacao
- `docs/arquitetura.md`, `docs/relatorio_tecnico.md` (com diagrama Mermaid do fluxo),
  `README.md` completo e atualizacao do README raiz.

## Criterios de aceite (mapeados ao enunciado)

| Requisito FIAP | Onde e atendido |
| --- | --- |
| Fine-tuning com protocolos, FAQs e modelos de laudo | `datagen/`, `finetune/` |
| Preprocessing, anonimizacao e curadoria | `datagen/anonymizer.py`, `build_finetune_dataset.py` |
| Pipeline LangChain integrando a LLM customizada | `assistant/chains.py`, `llm_provider.py` |
| Consulta em base estruturada | `assistant/db_tools.py`, `data/hospital.db` |
| Contextualizacao com dados atualizados do paciente | no `patient_context` do LangGraph |
| Limites de atuacao (nunca prescrever sem validacao) | `assistant/guardrails.py` |
| Logging detalhado para auditoria | `assistant/audit.py`, `logs/audit/*.jsonl` |
| Explainability / fonte da informacao | `assistant/retriever.py` + campo `sources` |
| Codigo modular em Python + README | estrutura de pacotes + `README.md` |
| Dataset anonimizado / sintetico | `data/processed/*.jsonl` |
| Relatorio tecnico + diagrama do fluxo | `docs/relatorio_tecnico.md` |
