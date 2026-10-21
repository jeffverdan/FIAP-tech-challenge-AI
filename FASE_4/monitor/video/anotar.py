"""Gera vídeos anotados (esqueleto, zonas, alertas) — usados na demonstração gravada."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from monitor.video.pose import ESQUELETO

COR_SEV = {0: (180, 180, 180), 1: (0, 200, 255), 2: (0, 140, 255), 3: (0, 0, 230)}   # BGR


def desenhar_esqueleto(img, kps: np.ndarray, cor=(80, 220, 80), conf_min: float = 0.3, esp: int = 2):
    import cv2

    if kps is None or np.isnan(kps).all():
        return img
    for a, b in ESQUELETO:
        if kps[a, 2] >= conf_min and kps[b, 2] >= conf_min:
            cv2.line(img, tuple(int(v) for v in kps[a, :2]), tuple(int(v) for v in kps[b, :2]), cor, esp, cv2.LINE_AA)
    for x, y, c in kps:
        if c >= conf_min:
            cv2.circle(img, (int(x), int(y)), esp + 1, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def faixa_texto(img, linhas: list[str], cor=(40, 40, 40), topo: bool = True, escala: float = 0.55):
    import cv2

    h, w = img.shape[:2]
    alt = int(24 * escala / 0.55)
    y0 = 0 if topo else h - alt * len(linhas) - 8
    cv2.rectangle(img, (0, y0), (w, y0 + alt * len(linhas) + 8), cor, -1)
    for i, ln in enumerate(linhas):
        cv2.putText(img, ln, (8, y0 + alt * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, escala, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def renderizar(video: Path, trilha, eventos, destino: Path, titulo: str, recorte_x=None, largura_saida: int = 960,
               zonas: list | None = None, info_por_t=None, duracao_alerta_s: float = 3.0, fps_saida: float | None = None):
    """Desenha a trilha de pose quadro a quadro e mostra cada evento por ``duracao_alerta_s``.

    ``info_por_t(t) -> list[str]`` opcional escreve medidas ao vivo (ex.: ângulo do ombro).
    """
    import cv2

    from monitor.video.pose import info_video, ler_video

    inf = info_video(video)
    fps = fps_saida or trilha.fps_amostrado
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    escrita = None
    idx = {int(q): i for i, q in enumerate(trilha.quadro)}
    for q in ler_video(video, fps, recorte_x):
        img = q.imagem.copy()
        i = idx.get(q.indice)
        if i is None:
            i = int(np.argmin(np.abs(trilha.t - q.t)))
        for z in zonas or []:
            cv2.rectangle(img, (int(z[0]), int(z[1])), (int(z[2]), int(z[3])), (255, 200, 0), 2)
            cv2.putText(img, "zona segura (leito)", (int(z[0]) + 4, int(z[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 200, 0), 1, cv2.LINE_AA)
        ativos = [e for e in eventos if 0 <= q.t - e.t <= duracao_alerta_s]
        cor = COR_SEV[max((int(e.severidade) for e in ativos), default=0)] if ativos else (80, 220, 80)
        desenhar_esqueleto(img, trilha.kps[i], cor=cor if ativos else (80, 220, 80), esp=max(2, img.shape[1] // 400))
        esc = largura_saida / img.shape[1]
        img = cv2.resize(img, (largura_saida, int(img.shape[0] * esc)))
        linhas = [f"{titulo}  |  t = {q.t:5.1f} s"]
        if info_por_t:
            linhas += info_por_t(q.t)
        faixa_texto(img, linhas)
        if ativos:
            e = ativos[-1]
            txt = f"ALERTA [{e.severidade.rotulo.upper()}] {e.descricao}"
            import textwrap

            largura_chars = max(40, int(img.shape[1] / 10))
            partes = textwrap.wrap(txt, largura_chars)[:3]
            faixa_texto(img, partes, cor=COR_SEV[int(e.severidade)], topo=False)
        if escrita is None:
            h, w = img.shape[:2]
            escrita = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        escrita.write(img)
    if escrita is not None:
        escrita.release()
    return destino
