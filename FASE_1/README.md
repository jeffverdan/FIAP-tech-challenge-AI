# Tech Challenge — Fase 1 | Grupo 62

Diagnóstico de Síndrome dos Ovários Policísticos (PCOS) via Machine Learning.

## Equipe

| Aluno |
| --- |
| Gabriel Pontin Buranello |
| Jeferson Verdan Oliveira |
| Josue Monteiro de Oliveira |
| Larissa Nunes da Silva |
| Oryange Strifezze |

---

## Sobre o projeto

O projeto constrói modelos preditivos de classificação para identificar pacientes com PCOS (Polycystic Ovary Syndrome / Síndrome dos Ovários Policísticos) a partir de dados clínicos, hormonais, metabólicos e de ultrassom. O objetivo é criar uma ferramenta de suporte ao diagnóstico médico — não um substituto ao julgamento clínico.

**Dataset:** PCOS (Polycystic Ovary Syndrome) — 541 pacientes, 45 features clínicas

Fonte: <https://www.kaggle.com/datasets/prasoonkottarathil/polycystic-ovary-syndrome-pcos>

---

## Estrutura do projeto

```text
data/
  PCOS_data_without_infertility.xlsx   # dataset principal (541 × 45)
  PCOS_infertility.csv                 # dataset complementar (541 × 6)
notebooks/
  01_eda_without_infertility.ipynb     # EDA + tratamento — dataset principal
  01_eda_infertility.ipynb             # EDA + tratamento — dataset complementar
  02_modeling.ipynb                    # correlação, modelagem, métricas, SHAP
notebooks/relatorio-tecnico.ipynb      # relatório técnico completo (com gráficos)
Dockerfile
requirements.txt
test_setup.py
```

---

## Como executar

### Ambiente local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para validar o setup:

```bash
python test_setup.py
```

Se aparecer `Setup OK!` no final, o ambiente está pronto. Abra os notebooks com:

```bash
jupyter notebook
```

### Docker

```bash
docker build -t tech-challenge-grupo62 .
docker run -p 8888:8888 -v $(pwd):/app tech-challenge-grupo62
```

Acesse o Jupyter no navegador pelo link exibido no terminal (token incluído na URL).

---

## Fluxo do projeto

| Etapa | Notebook | Conteúdo |
| --- | --- | --- |
| Exploração (dataset principal) | `01_eda_without_infertility.ipynb` | Estatísticas descritivas, distribuições, análise por classe |
| Exploração (dataset complementar) | `01_eda_infertility.ipynb` | Análise dos marcadores hormonais isolados |
| Modelagem e avaliação | `02_modeling.ipynb` | Correlação, pré-processamento, 3 modelos, métricas, SHAP |

---

## Modelos implementados

- **Regressão Logística** — melhor desempenho em Recall e F1-Score
- **K-Nearest Neighbors (KNN)** — otimizado via validação cruzada com 5 folds
- **Árvore de Decisão** — interpretabilidade nativa + feature importance

---

## Dependências

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
jupyter
openpyxl
```

---

## Entregáveis

- [x] Repositório Git organizado com histórico de PRs
- [x] Código-fonte completo nos notebooks
- [x] Dockerfile + README com instruções de execução
- [x] Dataset incluído na pasta `data/`
- [x] Relatório técnico (`notebooks/relatorio-tecnico.ipynb`)
- [x] [Vídeo de demonstração (até 15 min — YouTube/Vimeo)](https://youtu.be/hvWllv-rNHU)
