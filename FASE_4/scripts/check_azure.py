"""Valida a configuração Azure do ``.env`` com chamadas mínimas (custo ≈ zero).

    python scripts/check_azure.py

Speech: sintetiza 1 s de silêncio + tom e pede reconhecimento (só valida autenticação/região).
Language: analisa o sentimento de uma frase curta.
Não grava nada no cache.
"""

from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitor.config import CONFIG  # noqa: E402


def checar_speech() -> tuple[bool, str]:
    if not CONFIG.speech_configurado:
        return False, "AZURE_SPEECH_KEY/AZURE_SPEECH_REGION vazios"
    import azure.cognitiveservices.speech as speechsdk

    sr = 16000
    amostras = [int(0.1 * math.sin(2 * math.pi * 220 * i / sr) * 32767) for i in range(sr)]
    dados = struct.pack(f"<{sr}h", *amostras)
    tmp = Path(tempfile.gettempdir()) / "check_azure.wav"
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr), w.writeframes(dados)
    cfg = speechsdk.SpeechConfig(subscription=CONFIG.speech_key, region=CONFIG.speech_region)
    rec = speechsdk.SpeechRecognizer(cfg, speechsdk.audio.AudioConfig(filename=str(tmp)))
    r = rec.recognize_once()
    if r.reason == speechsdk.ResultReason.Canceled:
        d = r.cancellation_details
        if d.reason == speechsdk.CancellationReason.Error:
            return False, f"{d.error_code}: {d.error_details}"
    return True, f"ok (região {CONFIG.speech_region}, resultado: {r.reason.name})"


def checar_language() -> tuple[bool, str]:
    if not CONFIG.language_configurado:
        return False, "AZURE_LANGUAGE_KEY/AZURE_LANGUAGE_ENDPOINT vazios"
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    cli = TextAnalyticsClient(CONFIG.language_endpoint, AzureKeyCredential(CONFIG.language_key))
    try:
        doc = cli.analyze_sentiment(["I feel short of breath since yesterday."], language="en")[0]
    except Exception as e:  # noqa: BLE001
        return False, str(e).splitlines()[0]
    if doc.is_error:
        return False, f"{doc.error.code}: {doc.error.message}"
    return True, f"ok (sentimento de teste: {doc.sentiment})"


def main() -> int:
    print(json.dumps(CONFIG.resumo_azure(), indent=2, ensure_ascii=False))
    ok = True
    for nome, f in (("Speech", checar_speech), ("Language", checar_language)):
        try:
            sucesso, msg = f()
        except Exception as e:  # noqa: BLE001
            sucesso, msg = False, f"{type(e).__name__}: {e}"
        ok &= sucesso
        print(f"{'✔' if sucesso else '✘'} {nome}: {msg}")
    n_cache = sum(1 for _ in CONFIG.azure_cache_dir.rglob("*.json")) if CONFIG.azure_cache_dir.exists() else 0
    print(f"Respostas Azure em cache: {n_cache} (usadas em AZURE_MODE=auto/replay)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
