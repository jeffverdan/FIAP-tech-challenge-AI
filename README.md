# Tech Challenge — Pós-Graduação em IA para Devs | Grupo 62

Repositório central dos **Tech Challenges** da pós-graduação em Inteligência Artificial para Devs (FIAP). O curso é dividido em 4 fases, cada uma com um projeto prático avaliado, culminando em um **projeto final** que integra o trabalho de todas as fases em uma solução única.

> 👋 Se você é recrutador(a) ou avaliador(a) técnico, este README serve como mapa rápido do repositório: o que cada fase entrega, quais tecnologias foram usadas e onde encontrar código, notebooks e relatórios.

## Equipe

| Aluno |
| --- |
| Gabriel Pontin Buranello |
| Jeferson Verdan Oliveira |
| Josue Monteiro de Oliveira |
| Larissa Nunes da Silva |
| Oryange Strifezze |

---

## Contexto do projeto

O fio condutor das 4 fases é um problema real de saúde da mulher: **diagnóstico de Síndrome dos Ovários Policísticos (PCOS)** a partir de dados clínicos, hormonais e metabólicos. A cada fase, uma nova camada de IA é adicionada sobre o trabalho anterior — de machine learning clássico a otimização bio-inspirada e integração com LLMs — evoluindo para um sistema mais completo de suporte à decisão médica.

---

## Fases

| Fase | Pasta | Status | Foco técnico |
| --- | --- | --- | --- |
| **Fase 1** | [`FASE_1/`](./FASE_1) | ✅ Concluída | EDA, pré-processamento e modelos de classificação (ML clássico) |
| **Fase 2** | [`FASE_2/`](./FASE_2) | ✅ Concluída | Otimização de hiperparâmetros com Algoritmos Genéticos + interpretação via LLM |
| **Fase 3** | [`FASE_3/`](./FASE_3) | ✅ Concluída | Fine-tuning de LLM (LoRA) + assistente médico com LangChain/LangGraph |
| **Fase 4** | [`FASE_4/`](./FASE_4) | 🚧 Em entrega | Monitoramento multimodal (vídeo, áudio, sinais vitais, prescrições) + Azure Speech/Language + detecção de anomalias |
| **Projeto Final** | _a definir_ | ⏳ Planejada | Integração das 4 fases em uma solução completa |

### Fase 1 — Diagnóstico via Machine Learning

Modelos preditivos de classificação (Regressão Logística, KNN, Árvore de Decisão) para identificar pacientes com PCOS a partir de 45 features clínicas, com EDA completa, avaliação de métricas e explicabilidade via SHAP.

- **Stack:** Python, pandas, scikit-learn, SHAP, Jupyter, Docker
- **Dataset:** [PCOS (Kaggle)](https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos) — 541 pacientes
- **Destaques:** notebooks de EDA e modelagem, relatório técnico, Dockerfile pronto para execução, [vídeo de demonstração](https://youtu.be/hvWllv-rNHU)
- 📄 [README da Fase 1](./FASE_1/README.md)

### Fase 2 — Otimização com Algoritmos Genéticos + LLMs

Evolui os modelos da Fase 1 com **otimização de hiperparâmetros via Algoritmo Genético (DEAP)** e adiciona uma camada de **interpretação das previsões usando LLMs** (OpenAI API ou modelos locais via Transformers).

- **Stack:** Python, DEAP, scikit-learn, joblib, OpenAI API, Hugging Face Transformers/Accelerate
- **Destaques:** pipeline de experimentos de GA com múltiplas configurações, persistência de artefatos (`models/`, `results/`), integração com LLM em modo local ou via API, [vídeo de demonstração](https://youtu.be/OXV5hpyWxjs)
- 📄 [README da Fase 2](./FASE_2/README.md)

### Fase 3 — Assistente Médico com LLM Customizada, LangChain e LangGraph

Sobe uma camada sobre as fases anteriores: em vez de prever um rótulo, o sistema **conversa
com o médico sobre uma paciente específica**. Um LLM aberto é ajustado por **LoRA** aos
documentos internos de um hospital sintético (protocolos, FAQs de médicos, modelos de laudo),
e opera dentro de um **fluxo de decisão LangGraph** que cruza protocolo com prontuário
eletrônico, emite alertas por severidade e exige validação humana para toda conduta.

- **Stack:** Python, PEFT/LoRA, Hugging Face Transformers, LangChain, LangGraph, FAISS/BM25, SQLite, Streamlit, pytest
- **Dados:** corpus 100% sintético — 7 protocolos internos, 3 modelos de documento, 45 FAQs, 40 prontuários anonimizados
- **Destaques:** anonimização com pseudonimização HMAC e portão de qualidade anti-PII; guardrails determinísticos que impedem prescrição, fechamento de diagnóstico e alteração de prontuário; trilha de auditoria append-only com um evento por nó do grafo; explicabilidade por citação de protocolo, versão e seção; 107 testes automatizados; modo fallback que roda tudo sem GPU
- 📄 [README da Fase 3](./FASE_3/README.md) · [Relatório técnico](./FASE_3/docs/relatorio_tecnico.md) · [Arquitetura](./FASE_3/docs/arquitetura.md)

### Fase 4 — Monitoramento Multimodal com Detecção de Anomalias

O hospital passa a **monitorar o paciente continuamente**. Os dados vêm de quatro fontes: sinais
vitais, prescrições, áudio de consultas e vídeo do quarto e da fisioterapia. Cada detector emite
eventos padronizados, e um **motor de fusão determinístico** cruza as modalidades: por exemplo,
SpO₂ caindo no monitor **e** o paciente dizendo que está sem ar resultam em *insuficiência
respiratória provável*. Esse motor dispara alertas com justificativa, destinatários e prazo.

- **Stack:** Python, Azure AI Speech (STT), Azure AI Language (Text Analytics for Health + sentimento), YOLOv8-pose/YOLOv8, OpenPose, wav2vec 2.0, Praat/librosa, scikit-learn (Isolation Forest, Gradient Boosting), Streamlit, pytest
- **Dados públicos:** VitalDB, MIMIC-IV demo, PriMock57, TORGO, Coswara, REHAB24-6, UR Fall
- **Destaques:**
  - preditor de hipotensão em 10 min com AUROC de 0,79;
  - 82% dos episódios graves antecipados, em mediana 6 min antes do alarme;
  - detecção de queda com latência de 0,97 s;
  - YOLOv8-pose × OpenPose comparados contra *motion capture*;
  - armadilhas metodológicas tratadas (vazamento de falante, confusão por lote, mudança de domínio);
  - conta Azure trocável em 1 comando, com cache de respostas para *replay*;
  - dashboard em tempo real.
- 📄 [README da Fase 4](./FASE_4/README.md) · [Relatório técnico](./FASE_4/docs/relatorio_tecnico.md) · [Roteiro do vídeo](./FASE_4/docs/roteiro_video.md)

---

## Como navegar

Cada pasta `FASE_N/` é um projeto autocontido, com seu próprio ambiente virtual Python, `requirements.txt` e instruções de execução no respectivo README. Não há dependência de build entre as fases — o código e os artefatos (modelos, resultados) são compartilhados via arquivos versionados, não via imports diretos.

```text
tech-challenge-group-62/
├── FASE_1/    # ML clássico — diagnóstico de PCOS
├── FASE_2/    # Otimização genética + LLM
├── FASE_3/    # Assistente médico: fine-tuning LoRA + LangChain/LangGraph
├── FASE_4/    # Monitoramento multimodal: vídeo, áudio, sinais vitais + Azure
└── PROJETO_FINAL/  # (em breve)
```

Para rodar qualquer fase, entre na pasta correspondente e siga as instruções do README local (setup de venv, dependências e comandos de execução).
