"""Marcadores acústicos da fala — o que o Azure Speech **não** mede.

O Speech to Text diz *o que* o paciente falou; esta camada mede *como* ele falou. Os marcadores
são os usados na literatura de avaliação perceptivo-acústica da voz e de disartria:

* **Fonte glótica** (Praat/parselmouth): jitter e shimmer locais, HNR (harmônico-ruído) e CPPS
  (proeminência do pico cepstral suavizado — o melhor correlato isolado de voz soprosa/rouca).
* **Prosódia:** F0 média e variabilidade em semitons (monotonia é sinal de disartria e fadiga).
* **Fluência e respiração:** taxa de fala e de articulação (núcleos silábicos, de Jong &
  Wempe 2009), pausas por minuto, duração média de pausa, comprimento médio de frase entre
  pausas (fala "entrecortada" na dispneia), proporção de fonação.
* **Energia:** intensidade média e sua tendência ao longo da gravação (queda = fadiga vocal).
* **Espectro:** médias e desvios de 13 MFCC (entrada do classificador de disartria).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

SR = 16000


def carregar(caminho: Path, sr: int = SR) -> np.ndarray:
    import librosa

    y, _ = librosa.load(str(caminho), sr=sr, mono=True)
    return y


def _recortar(y: np.ndarray, sr: int, segmentos: list[tuple[float, float]] | None) -> np.ndarray:
    """Concatena só os trechos de fala do paciente (ex.: segmentos devolvidos pelo STT),
    com 150 ms de silêncio entre eles para não criar transições artificiais."""
    if not segmentos:
        return y
    gap = np.zeros(int(0.15 * sr), dtype=y.dtype)
    partes = []
    for a, b in segmentos:
        trecho = y[int(a * sr): int(b * sr)]
        if len(trecho) > 0.2 * sr:
            partes += [trecho, gap]
    return np.concatenate(partes) if partes else y


def intervalos_fala(y: np.ndarray, sr: int, top_db: float = 30, pausa_min_s: float = 0.25) -> list[tuple[float, float]]:
    import librosa

    iv = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=256)
    fala: list[list[float]] = []
    for a, b in iv / sr:
        if fala and a - fala[-1][1] < pausa_min_s:
            fala[-1][1] = b          # pausas curtas demais são coarticulação, não pausa
        else:
            fala.append([a, b])
    return [(a, b) for a, b in fala if b - a >= 0.05]


def _nucleos_silabicos(snd, intensidade_min_db: float | None = None) -> int:
    """Conta picos de intensidade ≥ 2 dB acima do vale anterior, em trechos sonoros
    (algoritmo de de Jong & Wempe, 2009, simplificado)."""
    inten = snd.to_intensity(minimum_pitch=50, time_step=0.01)
    v = inten.values[0]
    t = inten.xs()
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    if intensidade_min_db is None:
        intensidade_min_db = float(np.median(v))
    n, vale = 0, v[0] if len(v) else 0
    for i in range(1, len(v) - 1):
        vale = min(vale, v[i])
        if v[i] > v[i - 1] and v[i] >= v[i + 1] and v[i] > intensidade_min_db and v[i] - vale >= 2.0:
            f0 = pitch.get_value_at_time(t[i])
            if f0 and not math.isnan(f0):
                n += 1
                vale = v[i]
    return n


def features_voz(y: np.ndarray, sr: int = SR, segmentos: list[tuple[float, float]] | None = None,
                 mfcc: bool = True) -> dict[str, float]:
    import librosa
    import parselmouth
    from parselmouth.praat import call

    y = _recortar(y, sr, segmentos)
    dur = len(y) / sr
    f: dict[str, float] = {"duracao_s": round(dur, 2)}
    if dur < 0.5 or np.max(np.abs(y)) < 1e-4:
        return f

    # ---- fluência e pausas ------------------------------------------------------------
    fala = intervalos_fala(y, sr)
    t_fala = sum(b - a for a, b in fala)
    pausas = [fala[i + 1][0] - fala[i][1] for i in range(len(fala) - 1)]
    f["proporcao_fonacao"] = t_fala / dur
    f["pausas_por_min"] = len(pausas) / (dur / 60)
    f["pausa_media_s"] = float(np.mean(pausas)) if pausas else 0.0
    f["pausas_longas_por_min"] = sum(p >= 1.0 for p in pausas) / (dur / 60)
    f["frase_media_s"] = float(np.mean([b - a for a, b in fala])) if fala else 0.0

    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=sr)
    sil = _nucleos_silabicos(snd)
    f["taxa_fala_sil_s"] = sil / dur
    f["taxa_articulacao_sil_s"] = sil / t_fala if t_fala > 0 else 0.0

    # ---- prosódia ---------------------------------------------------------------------
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=500)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) > 10:
        st = 12 * np.log2(f0 / np.median(f0))
        f["f0_media_hz"] = float(np.mean(f0))
        f["f0_desvio_st"] = float(np.std(st))
        f["f0_amplitude_st"] = float(np.percentile(st, 95) - np.percentile(st, 5))
        f["fracao_sonora"] = len(f0) / max(pitch.get_number_of_frames(), 1)

    # ---- fonte glótica ------------------------------------------------------------------
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 500)
        f["jitter_local_pct"] = 100 * call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        f["shimmer_local_pct"] = 100 * call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:  # noqa: BLE001 — trecho sem periodicidade suficiente
        pass
    try:
        harm = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=75)
        hv = harm.values[0]
        f["hnr_db"] = float(np.mean(hv[hv > -200])) if (hv > -200).any() else np.nan
    except Exception:  # noqa: BLE001
        pass
    try:
        # CPPS é caro (escala com a duração): estima em até 8 s de fala, que já estabiliza a média
        trecho = snd if dur <= 8 else snd.extract_part(from_time=0, to_time=8, preserve_times=False)
        pc = call(trecho, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        f["cpps_db"] = call(pc, "Get CPPS", "no", 0.02, 0.0005, 60, 330, 0.05, "Parabolic", 0.001, 0.05,
                            "Exponential decay", "Robust")
    except Exception:  # noqa: BLE001 — assinatura do comando varia entre versões do Praat
        pass

    # ---- energia ------------------------------------------------------------------------
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    db = 20 * np.log10(rms + 1e-9)
    ativo = db > (np.max(db) - 35)
    f["intensidade_media_db"] = float(np.mean(db[ativo])) if ativo.any() else np.nan
    if ativo.sum() > 20 and dur > 20:
        tt = np.arange(len(db))[ativo] * 256 / sr / 60
        f["intensidade_tendencia_db_min"] = float(np.polyfit(tt, db[ativo], 1)[0])

    # ---- espectro -----------------------------------------------------------------------
    if mfcc:
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            f[f"mfcc{i}_media"] = float(m[i].mean())
            f[f"mfcc{i}_desvio"] = float(m[i].std())
        f["centroide_espectral_hz"] = float(librosa.feature.spectral_centroid(y=y, sr=sr).mean())
    return {k: (round(v, 4) if isinstance(v, float) and not math.isnan(v) else v) for k, v in f.items()}


#: Marcadores interpretáveis (sem MFCC) — usados nos relatórios e no dashboard.
MARCADORES = ["taxa_fala_sil_s", "taxa_articulacao_sil_s", "pausas_por_min", "pausa_media_s",
              "pausas_longas_por_min", "frase_media_s", "proporcao_fonacao", "f0_media_hz", "f0_desvio_st",
              "f0_amplitude_st", "jitter_local_pct", "shimmer_local_pct", "hnr_db", "cpps_db",
              "intensidade_media_db", "intensidade_tendencia_db_min"]
