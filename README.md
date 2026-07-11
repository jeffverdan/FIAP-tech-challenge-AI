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
| **Fase 3** | _a definir_ | ⏳ Planejada | — |
| **Fase 4** | _a definir_ | ⏳ Planejada | — |
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

---

## Como navegar

Cada pasta `FASE_N/` é um projeto autocontido, com seu próprio ambiente virtual Python, `requirements.txt` e instruções de execução no respectivo README. Não há dependência de build entre as fases — o código e os artefatos (modelos, resultados) são compartilhados via arquivos versionados, não via imports diretos.

```text
tech-challenge-group-62/
├── FASE_1/    # ML clássico — diagnóstico de PCOS
├── FASE_2/    # Otimização genética + LLM
├── FASE_3/    # (em breve)
├── FASE_4/    # (em breve)
└── PROJETO_FINAL/  # (em breve)
```

Para rodar qualquer fase, entre na pasta correspondente e siga as instruções do README local (setup de venv, dependências e comandos de execução).
