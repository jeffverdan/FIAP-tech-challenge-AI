"""Azure AI Speech — Speech to Text com marcação de tempo por palavra.

Usa reconhecimento contínuo do SDK (áudios de consulta têm 5–15 min; o modo "single shot" corta
em 15 s). A saída é normalizada para um formato independente do provedor:

.. code-block:: python

    {"texto": "...", "duracao_s": 578.1, "idioma": "en-GB",
     "segmentos": [{"inicio": 2.7, "fim": 3.6, "texto": "Hello.", "confianca": 0.93}, ...],
     "palavras":  [{"palavra": "hello", "inicio": 2.7, "fim": 3.1, "confianca": 0.95}, ...]}

Fallbacks (``AZURE_MODE=auto|offline``), nesta ordem:
  1. transcrição de referência do dataset, quando existe (PriMock57 traz TextGrid manual);
  2. ``faster-whisper`` local, se instalado (``pip install faster-whisper``).
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from monitor.azure import cache
from monitor.config import CONFIG

TICKS = 1e7  # o serviço devolve tempos em unidades de 100 ns


def _azure_transcrever(caminho: Path, idioma: str, timeout_s: float) -> dict:
    import azure.cognitiveservices.speech as speechsdk

    cfg = speechsdk.SpeechConfig(subscription=CONFIG.speech_key, region=CONFIG.speech_region)
    cfg.speech_recognition_language = idioma
    cfg.request_word_level_timestamps()
    cfg.output_format = speechsdk.OutputFormat.Detailed
    audio = speechsdk.audio.AudioConfig(filename=str(caminho))
    rec = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)

    resultados: list[dict] = []
    erro: list[str] = []
    fim = threading.Event()

    def reconhecido(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            resultados.append(json.loads(evt.result.json))

    def cancelado(evt):
        det = evt.cancellation_details
        if det.reason == speechsdk.CancellationReason.Error:
            erro.append(f"{det.error_code}: {det.error_details}")
        fim.set()

    rec.recognized.connect(reconhecido)
    rec.canceled.connect(cancelado)
    rec.session_stopped.connect(lambda _evt: fim.set())
    rec.start_continuous_recognition()
    fim.wait(timeout_s)
    rec.stop_continuous_recognition()
    if erro:
        raise cache.AzureIndisponivel("Azure Speech: " + erro[0])

    segmentos, palavras = [], []
    for r in resultados:
        melhor = (r.get("NBest") or [{}])[0]
        ini = r["Offset"] / TICKS
        segmentos.append({"inicio": round(ini, 3), "fim": round(ini + r["Duration"] / TICKS, 3),
                          "texto": r.get("DisplayText", melhor.get("Display", "")),
                          "confianca": round(float(melhor.get("Confidence", 0.0)), 3)})
        for w in melhor.get("Words", []):
            wi = w["Offset"] / TICKS
            palavras.append({"palavra": w["Word"], "inicio": round(wi, 3), "fim": round(wi + w["Duration"] / TICKS, 3),
                             "confianca": round(float(w.get("Confidence", melhor.get("Confidence", 0.0))), 3)})
    return {"texto": " ".join(s["texto"] for s in segmentos).strip(), "segmentos": segmentos,
            "palavras": palavras, "idioma": idioma, "provedor": "azure_speech"}


# --------------------------------------------------------------------------------------
# Fallbacks
# --------------------------------------------------------------------------------------
def ler_textgrid(caminho: Path) -> list[dict]:
    """Lê os intervalos não vazios de um TextGrid (formato longo do Praat)."""
    txt = Path(caminho).read_text(encoding="utf-8", errors="ignore")
    segs = []
    for m in re.finditer(r"xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = \"(.*?)\"", txt, re.S):
        texto = m.group(3).strip()
        if texto:
            segs.append({"inicio": round(float(m.group(1)), 3), "fim": round(float(m.group(2)), 3),
                         "texto": texto, "confianca": 1.0})
    return segs


def _fallback(caminho: Path, referencia: Path | None, idioma: str) -> dict:
    if referencia is not None and Path(referencia).exists():
        segs = ler_textgrid(referencia)
        return {"texto": " ".join(s["texto"] for s in segs), "segmentos": segs, "palavras": [],
                "idioma": idioma, "provedor": "transcricao_referencia"}
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise cache.AzureIndisponivel(
            "Sem Azure Speech, sem transcrição de referência e sem faster-whisper instalado") from e
    modelo = WhisperModel("base", device="cpu", compute_type="int8")
    segs_w, _ = modelo.transcribe(str(caminho), language=idioma.split("-")[0], word_timestamps=True)
    segmentos, palavras = [], []
    for s in segs_w:
        segmentos.append({"inicio": round(s.start, 3), "fim": round(s.end, 3), "texto": s.text.strip(),
                          "confianca": round(float(2 ** s.avg_logprob), 3)})
        for w in s.words or []:
            palavras.append({"palavra": w.word.strip(), "inicio": round(w.start, 3), "fim": round(w.end, 3),
                             "confianca": round(float(w.probability), 3)})
    return {"texto": " ".join(s["texto"] for s in segmentos), "segmentos": segmentos, "palavras": palavras,
            "idioma": idioma, "provedor": "faster_whisper"}


def transcrever(caminho: Path, idioma: str | None = None, referencia: Path | None = None,
                timeout_s: float = 1800) -> dict:
    """Transcreve um WAV (16 kHz mono recomendado) respeitando ``AZURE_MODE``."""
    caminho = Path(caminho)
    idioma = idioma or CONFIG.speech_language
    k = cache.chave("speech", "stt_continuo", cache.hash_arquivo(caminho), idioma=idioma)
    res = cache.executar(
        "speech", k,
        ao_vivo=lambda: _azure_transcrever(caminho, idioma, timeout_s),
        configurado=CONFIG.speech_configurado,
        fallback=lambda: _fallback(caminho, referencia, idioma),
        meta={"arquivo": caminho.name, "regiao": CONFIG.speech_region},
    )
    return res
