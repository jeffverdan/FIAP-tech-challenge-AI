# FASE 4 — Monitoramento Multimodal Contínuo com Detecção de Anomalias

Tech Challenge da Pós-graduação em IA para Devs (FIAP).
Autoria individual: **Jeferson Verdan Oliveira — RM370792**.

O Hospital Aurora, que nas fases anteriores ganhou modelos preditivos (Fases 1–2) e um assistente
médico com LLM (Fase 3), agora **monitora o paciente continuamente** com dados de vários tipos:
sinais vitais, prescrições, áudio de consultas e vídeo do quarto e da fisioterapia. O objetivo é
identificar sinais precoces de risco e avisar a equipe certa no momento certo.

> ⚠️ **Projeto acadêmico.** Todos os dados vêm de datasets públicos de pesquisa (lista abaixo). O
> sistema não é um dispositivo médico e não substitui o julgamento clínico.

O princípio herdado da Fase 3 continua valendo: **os modelos detectam, as regras decidem.**
Visão computacional, fala e aprendizado de máquina produzem *evidências*. A decisão de alertar (a
severidade, quem é avisado e com que prazo) é tomada por código determinístico e auditável em
`monitor/fusion/`.

---

## O que foi construído

| Entrega do enunciado | Onde está |
| --- | --- |
| Vídeo: análise postural (OpenPose) | `monitor/video/pose.py` — backends **YOLOv8-pose** e **OpenPose**, comparados contra *motion capture* |
| Vídeo: objetos e áreas críticas (YOLOv8) | `monitor/video/pose.py::DetectorObjetos`, zonas de leito em `monitor/video/quedas.py` |
| Vídeo: relatório automático de desvios | `monitor/video/reabilitacao.py` → `results/video/relatorios/*.md` |
| Áudio: Azure Speech to Text | `monitor/azure/speech.py` (reconhecimento contínuo, tempo por palavra) |
| Áudio: termos críticos e sentimento com Azure Text Analytics | `monitor/azure/language.py` (Text Analytics for Health + sentimento + frases-chave) |
| Áudio: alterações vocais (fadiga, disartria, respiração) | `monitor/audio/acustica.py`, `monitor/audio/modelos.py` |
| Anomalias em sinais vitais | `monitor/vitals/detectores.py` (regras sustentadas, z-score robusto, Isolation Forest, NEWS2) e `monitor/vitals/predicao.py` (preditor de hipotensão) |
| Anomalias na evolução das prescrições | `monitor/prescricoes.py` |
| Anomalias na movimentação do paciente | `monitor/video/quedas.py` (queda, imobilidade após a queda, zona do leito) |
| Fusão multimodal + alertas automáticos | `monitor/fusion/motor.py`, `monitor/fusion/notificacao.py` |
| Integração com serviços gerenciados Azure | `monitor/azure/`, `scripts/azure_setup.ps1`, `scripts/check_azure.py` |
| Dashboard em tempo real | `app/streamlit_app.py` |
| Relatório técnico | [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) |
| Roteiro do vídeo de demonstração | [`docs/roteiro_video.md`](docs/roteiro_video.md) |

**Vídeo de demonstração:** _(link do YouTube, não listado — adicionar após a gravação)_

## Resultados principais

Os detalhes estão no [relatório técnico](docs/relatorio_tecnico.md).

| Modalidade | Resultado |
| --- | --- |
| Sinais vitais (30 cirurgias de teste) | As camadas precoces anteciparam **82%** dos episódios graves, com mediana de **6 min** antes do alarme por limiar. O preditor de hipotensão em 10 min teve AUROC de **0,79** (contra 0,75 usando só a PAM atual). |
| Consultas (Azure Speech + Text Analytics for Health) | A queixa principal foi encontrada em **7/7** consultas, com WER de transcrição de **21%**. O sistema também encontrou um erro de metadado no próprio dataset. |
| Prescrições (MIMIC-IV demo) | 840 alterações suspeitas em 250 internações: salto de dose, dose atípica, instabilidade e combinações perigosas. |
| Voz | Disartria: AUROC de **0,77** por gravação e **0,91** por falante (wav2vec 2.0, validação por falante). Dificuldade respiratória: 0,68, com a confusão por lote corrigida. A calibração de domínio reduziu os falsos alarmes de 10/11 para **0/11**. |
| Pose | YOLOv8-pose × OpenPose contra *motion capture*: PCK@0,2 de 0,93–0,995, com erro do ângulo do ombro de **6,6°** a 10,5°. O YOLOv8-pose é 21× mais rápido. |
| Fisioterapia | Detecção de repetição incorreta: AUROC 0,67–0,70 sem rótulos (padrão esperado) e **0,78–0,81** supervisionado. A contagem de repetições erra em média 0,7. |
| Quedas (UR Fall) | Sensibilidade de **83%**, especificidade de 100% em atividades sem deitar e latência de **0,97 s**. |
| Fusão | MIMIC: 71 alertas de fusão reais (prescrição → sinal vital). Leito 12: 4 alertas multimodais, sendo 3 críticos. |

---

## Arquitetura em uma figura

```
 VitalDB / MIMIC (sinais vitais) ─► limpeza ─► regras sustentadas ─┐
                                              z-score robusto ────┤
                                              Isolation Forest ───┤
                                              preditor hipotensão ┤
 MIMIC (prescrições) ─► salto de dose / dose atípica / combinações ┤
 PriMock57 (consulta) ─► Azure Speech ─► Azure Text Analytics ────┤   Evento   ┌──────────────┐   Alerta
                         └► marcadores acústicos ─► modelos de voz ┼──────────►│ Motor de     │──────────► painel
 REHAB24-6 (fisio) ─► YOLOv8-pose/OpenPose ─► ângulos ─► desvios ──┤            │ fusão        │            webhook
 UR Fall (quarto) ─► YOLOv8-pose + YOLOv8 (leito) ─► queda ────────┘            │ (regras)     │            auditoria
                                                                               └──────────────┘
```

Cada detector, de qualquer modalidade, emite o mesmo objeto `Evento` (`monitor/events.py`). O
motor de fusão (1) transforma eventos de severidade "atenção" ou maior em alertas, suprimindo
repetições; (2) aplica **regras de correlação entre modalidades** (por exemplo, SpO₂ caindo **e**
paciente dizendo que está sem ar resultam em *insuficiência respiratória provável*, crítico); e (3)
calcula um escore de risco por paciente.

---

## Datasets (todos públicos)

| Modalidade | Dataset | Uso |
| --- | --- | --- |
| Sinais vitais (2 s) | **VitalDB** (PhysioNet), 100 cirurgias | Regras, z-score, Isolation Forest, preditor de hipotensão |
| Sinais vitais + prescrições | **MIMIC-IV Clinical Database Demo** (PhysioNet), 100 pacientes | NEWS2, prescrições, fusão real de duas modalidades do mesmo paciente |
| Consultas | **PriMock57** (consultas simuladas, canal do paciente separado) | Azure Speech + Text Analytics |
| Disartria | **TORGO** (espelho no Hugging Face) | Classificador de disartria |
| Respiração/fadiga | **Coswara** (IISc) | Classificadores de dificuldade respiratória e fadiga |
| Fisioterapia | **REHAB24-6** (Zenodo), repetições rotuladas como certas ou erradas | Desvios do padrão esperado |
| Movimentação | **UR Fall Detection** | Queda e imobilidade |

`python scripts/download_data.py` baixa só os subsets usados (cerca de 2,5 GB). Os dados brutos não
vão para o git. Os resultados, eventos e modelos pequenos ficam em `results/` e `data/processed/`.
Por isso **o dashboard funciona sem baixar nada.**

---

## Setup

### Windows (PowerShell)

```powershell
cd FASE_4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cd FASE_4
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### Azure: criar os recursos e trocar de conta

São usados dois serviços com tier gratuito F0: **Speech** e **Language**.

```powershell
az login                                  # na conta que vai ser usada
.\scripts\azure_setup.ps1                 # cria rg-aurora-fase4 + Speech + Language e grava as chaves no .env
python scripts\check_azure.py             # valida
```

**Para trocar de conta**, basta fazer `az login` na conta nova e rodar o `azure_setup.ps1` de novo.
Nada no código muda. Se o tier F0 já estiver em uso naquela assinatura, rode
`.\scripts\azure_setup.ps1 -Sku S0`.

Toda resposta do Azure fica gravada em `data/azure_cache/`, indexada pelo **conteúdo** (hash do
áudio ou do texto), não pela conta. O comportamento é controlado por `AZURE_MODE` no `.env`:

| Modo | Comportamento |
| --- | --- |
| `auto` (padrão) | usa o cache; se não houver, chama o Azure; se o Azure falhar, usa o fallback local |
| `live` | só Azure (falha sem chave) |
| `replay` | só o cache: a demo roda com as respostas reais mesmo **depois que a conta expirar** |
| `offline` | só fallback local: transcrição de referência do dataset ou Whisper, e léxico de sintomas |

---

## Como rodar

```powershell
python scripts\download_data.py            # dados públicos (uma vez)
python scripts\run_vitals.py               # sinais vitais (~15 min em CPU)
python scripts\run_audio.py                # áudio (~15 min; com Azure ativo, popula o cache)
python scripts\run_video.py                # vídeo (~40 min em CPU; a pose fica em cache)
python scripts\run_fusao.py                # fusão (MIMIC + cenário composto do Leito 12)
streamlit run app\streamlit_app.py         # dashboard → http://127.0.0.1:8501
pytest -q                                  # testes unitários
```

Depois de configurar o Azure, rode `python scripts\run_audio.py --so-consultas` para que as
consultas passem pelo Speech e pelo Text Analytics de verdade. O WER contra a transcrição manual
aparece em `results/audio/consultas_resumo.csv`.

> **Dashboard no Windows:** abra `http://127.0.0.1:8501`, não `localhost` (o Uvicorn liga só em
> IPv6 e o websocket não conecta).

---

## Estrutura

```
FASE_4/
├── monitor/
│   ├── config.py, events.py          configuração (.env) e contrato Evento
│   ├── azure/                        Speech, Language, cache e modos
│   ├── vitals/                       canais, NEWS2, detectores streaming, preditor de hipotensão
│   ├── prescricoes.py                anomalias de prescrição
│   ├── audio/                        acústica, modelos de voz, pipeline de consulta
│   ├── video/                        pose, cinemática, fisioterapia, quedas, vídeos anotados
│   └── fusion/                       motor de fusão, notificação e auditoria
├── scripts/                          download, experimentos, setup Azure
├── app/streamlit_app.py              Central de Alertas
├── data/processed/                   modelos, features, cenário de demonstração, cache Azure
├── results/                          métricas, figuras, eventos, relatórios e vídeos anotados
├── docs/                             relatório técnico e roteiro do vídeo
└── tests/
```
