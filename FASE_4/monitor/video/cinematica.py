"""Cinemática 2D a partir dos keypoints: ângulos articulares, inclinação de tronco, simetria.

Ângulos em graus. Keypoints com confiança < ``CONF_MIN`` viram NaN e as séries são suavizadas
com mediana móvel (remove "saltos" de um quadro, comuns em pose 2D) e interpoladas em lacunas
curtas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from monitor.video.pose import KP

CONF_MIN = 0.3


def _pontos(kps: np.ndarray, nome: str) -> np.ndarray:
    p = kps[:, KP[nome], :2].astype(float).copy()
    p[kps[:, KP[nome], 2] < CONF_MIN] = np.nan
    return p


def angulo(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Ângulo ABC (vértice em B), em graus, vetorizado por quadro."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def angulo_com_vertical(baixo: np.ndarray, cima: np.ndarray) -> np.ndarray:
    """Inclinação do segmento baixo→cima em relação à vertical (0° = em pé). y cresce para baixo."""
    v = cima - baixo
    return np.degrees(np.arctan2(np.abs(v[:, 0]), -v[:, 1]))


def suavizar(s: pd.Series, janela: int = 5, max_lacuna: int = 5) -> pd.Series:
    return s.interpolate(limit=max_lacuna, limit_area="inside").rolling(janela, center=True, min_periods=1).median()


def serie_cinematica(t: np.ndarray, kps: np.ndarray) -> pd.DataFrame:
    P = {n: _pontos(kps, n) for n in KP}
    meio_ombro = (P["ombro_e"] + P["ombro_d"]) / 2
    meio_quadril = (P["quadril_e"] + P["quadril_d"]) / 2
    df = pd.DataFrame({"t": t})
    for lado, s in (("d", "d"), ("e", "e")):
        # abdução do ombro: ângulo entre o tronco (ombro→quadril) e o braço (ombro→cotovelo)
        df[f"abducao_ombro_{lado}"] = angulo(P[f"quadril_{s}"], P[f"ombro_{s}"], P[f"cotovelo_{s}"])
        df[f"cotovelo_{lado}"] = angulo(P[f"ombro_{s}"], P[f"cotovelo_{s}"], P[f"punho_{s}"])
        df[f"joelho_{lado}"] = angulo(P[f"quadril_{s}"], P[f"joelho_{s}"], P[f"tornozelo_{s}"])
        df[f"quadril_{lado}"] = angulo(P[f"ombro_{s}"], P[f"quadril_{s}"], P[f"joelho_{s}"])
        # punho acima do ombro (normalizado pela altura do tronco; >0 = acima)
        tronco = np.linalg.norm(meio_ombro - meio_quadril, axis=1)
        df[f"punho_acima_ombro_{lado}"] = (P[f"ombro_{s}"][:, 1] - P[f"punho_{s}"][:, 1]) / (tronco + 1e-9)
    df["inclinacao_tronco"] = angulo_com_vertical(meio_quadril, meio_ombro)
    # inclinação lateral dos ombros (linha ombro-ombro vs. horizontal) — compensação no plano frontal
    v = P["ombro_e"] - P["ombro_d"]
    df["inclinacao_ombros"] = np.degrees(np.arctan2(np.abs(v[:, 1]), np.abs(v[:, 0]) + 1e-9))
    df["altura_quadril_norm"] = meio_quadril[:, 1]
    df["tronco_px"] = np.linalg.norm(meio_ombro - meio_quadril, axis=1)
    for c in df.columns:
        if c != "t":
            df[c] = suavizar(df[c])
    return df
