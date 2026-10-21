# FASE 4 — Plano de execução

Monitoramento multimodal contínuo do **Hospital Aurora**: vídeo, áudio, texto e sinais vitais
fundidos numa linha do tempo por paciente, com detecção de anomalias e alerta à equipe.

Princípio herdado da FASE_3: **os modelos detectam, as regras decidem.** Toda decisão de
alerta (severidade, escalonamento, destinatário) é Python determinístico e auditável em
`monitor/fusion/`. Modelos de visão, fala e linguagem produzem *evidências*; nunca disparam
um alerta sozinhos.

## Datasets (todos públicos)

| Modalidade | Dataset | Uso |
| --- | --- | --- |
| Sinais vitais (alta frequência) | **VitalDB** (PhysioNet) | FC, PA sistólica/diastólica/média, SpO₂, FR — stream "tempo real" |
| Sinais vitais + prescrições | **MIMIC-IV Clinical Database Demo** (PhysioNet, 100 pacientes) | `chartevents` (UTI) e `prescriptions`/`emar` |
| Áudio de consultas | **PriMock57** (consultas simuladas, canal do paciente separado) | Azure Speech → Azure Text Analytics for Health + sentimento |
| Fala patológica | **TORGO** (via Hugging Face) | Classificador disartria × controle com marcadores acústicos |
| Vídeo de fisioterapia | **REHAB24-6** (Zenodo, repetições rotuladas certo/errado) | Pose → ângulos → desvio do padrão esperado |
| Vídeo de movimentação | **UR Fall Detection** | Queda / imobilidade / saída do leito |

Os dados brutos **não vão para o git** (`data/raw/`). `scripts/download_data.py` baixa só os
subsets usados. Resultados pequenos (features, keypoints, eventos) ficam em `data/processed/`
para a demo rodar sem baixar nada.

## Azure — troca de conta em 3 passos

Serviços: **Speech** (Speech to Text) e **Language** (Text Analytics: sentimento, frases-chave,
Text Analytics for Health). Ambos têm tier gratuito **F0**.

1. `scripts/azure_setup.ps1` (ou `.sh`) cria grupo + recursos na assinatura ativa do `az`
   e grava as chaves no `.env`.
2. `python scripts/check_azure.py` valida as chaves.
3. Pronto. Nenhuma chave aparece no código.

**Cache de respostas:** toda chamada Azure é gravada em `data/azure_cache/` (JSON). Com
`AZURE_MODE=replay` a demo roda com as respostas reais já obtidas, mesmo depois que a conta
expirar. Modos: `live` (só Azure), `auto` (cache → Azure → fallback local), `replay`
(só cache), `offline` (fallback local: Whisper/heurísticas).

> Azure Anomaly Detector foi descontinuado pela Microsoft (out/2026) — por isso a detecção de
> anomalias é local (scikit-learn), o que também é explicado no relatório.

## Módulos

1. **Anomalias** — `monitor/vitals/` (NEWS2 por regra + z-score móvel + Isolation Forest),
   `monitor/prescriptions.py` (salto de dose, troca/suspensão abrupta, duplicidade de classe).
2. **Áudio** — `monitor/audio/` (Azure STT + TA4H + sentimento; features acústicas com
   librosa/Praat: taxa de fala, pausas, F0, jitter, shimmer, HNR; classificador de disartria).
3. **Vídeo** — `monitor/video/` (YOLOv8-pose e OpenPose como backends de pose; YOLOv8 para
   objetos/pessoas e zonas críticas; ângulos articulares por repetição; detecção de queda).
4. **Fusão e alerta** — `monitor/fusion/` (eventos → linha do tempo do paciente → escore →
   alerta com severidade e justificativa; saída em dashboard, JSONL e webhook opcional).
5. **App** — `app/streamlit_app.py` (Central de Alertas com replay acelerado).

## Cronograma (entrega antes da conta Azure expirar)

| Dia | Entrega |
| --- | --- |
| 1 | Estrutura, downloads, anomalias em sinais vitais e prescrições |
| 2 | Áudio: Azure + acústica + disartria |
| 3 | Vídeo: pose, reabilitação, quedas |
| 4 | Fusão, alertas, dashboard |
| 5 | Relatório técnico, testes, roteiro do vídeo |
| 6–7 | Jeff: criar recursos Azure, rodar em `live` para popular o cache, gravar o vídeo |
