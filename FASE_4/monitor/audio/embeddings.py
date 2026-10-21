"""Embeddings de fala auto-supervisionados (wav2vec 2.0) — opcional.

O wav2vec 2.0 (Baevski et al., 2020) foi pré-treinado em 960 h de fala sem rótulo; as camadas
intermediárias codificam articulação e prosódia muito melhor que MFCC. Usamos o modelo
congelado como extrator: média temporal da camada ``CAMADA`` → vetor de 768 dimensões, e um
classificador linear por cima. É a abordagem padrão atual para disartria com poucos falantes.

Requer ``torchaudio`` (pesos baixados de download.pytorch.org na primeira execução, ~360 MB).
Sem ele, os modelos de voz usam só os marcadores acústicos clássicos.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

CAMADA = 6   # camadas intermediárias (5–8) são as mais informativas para paralinguística


@lru_cache(maxsize=1)
def _modelo():
    import torch
    import torchaudio

    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    modelo = bundle.get_model().eval()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    return modelo, bundle.sample_rate


def disponivel() -> bool:
    try:
        import torchaudio  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 — ImportError ou DLL incompatível (torch e torchaudio de versões diferentes)
        return False


def embedding(y: np.ndarray, sr: int = 16000, max_s: float = 30.0) -> np.ndarray:
    """Vetor de 768 dimensões (média temporal da camada ``CAMADA``)."""
    import torch

    modelo, sr_modelo = _modelo()
    assert sr == sr_modelo, "reamostre para 16 kHz antes"
    y = y[: int(max_s * sr)]
    if len(y) < sr // 2:
        y = np.pad(y, (0, sr // 2 - len(y)))
    with torch.inference_mode():
        x = torch.from_numpy(y.astype(np.float32))[None]
        camadas, _ = modelo.extract_features(x, num_layers=CAMADA)
        return camadas[-1][0].mean(0).numpy()


def embedding_segmentos(y: np.ndarray, segmentos: list[tuple[float, float]] | None, sr: int = 16000,
                        max_total_s: float = 120.0) -> np.ndarray:
    """Média dos embeddings dos trechos de fala (para áudios longos de consulta)."""
    if not segmentos:
        return embedding(y, sr)
    vets, total = [], 0.0
    for a, b in segmentos:
        if b - a < 0.5:
            continue
        vets.append(embedding(y[int(a * sr): int(b * sr)], sr))
        total += b - a
        if total >= max_total_s:
            break
    return np.mean(vets, axis=0) if vets else embedding(y, sr)
