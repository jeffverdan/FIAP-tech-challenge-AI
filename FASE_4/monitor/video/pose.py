"""Estimação de pose e detecção de objetos em vídeo — backends intercambiáveis.

Todos os backends devolvem keypoints no **formato COCO-17** (x, y, confiança em pixels do
quadro original), para que o resto do pipeline não saiba qual modelo rodou:

    0 nariz  1 olho_e  2 olho_d  3 orelha_e  4 orelha_d  5 ombro_e  6 ombro_d  7 cotovelo_e
    8 cotovelo_d  9 punho_e  10 punho_d  11 quadril_e  12 quadril_d  13 joelho_e  14 joelho_d
    15 tornozelo_e  16 tornozelo_d

Backends:
  * ``yolo``     — Ultralytics YOLOv8-pose (padrão; ~15 quadros/s em CPU com o modelo *n*).
  * ``openpose`` — OpenPose (CMU, Cao et al. 2017), implementação PyTorch do modelo *body_25/
                   COCO-18* distribuída em ``lllyasviel/Annotators`` (via ``controlnet_aux``).
                   Mais lento (~1 quadro/s em CPU); usado na comparação do relatório.
  * :class:`DetectorObjetos` — YOLOv8 de detecção (COCO-80) para pessoas, leito, cadeira,
    sofá etc. — define as "áreas críticas" e quantas pessoas estão em cena.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

NOMES_KP = ["nariz", "olho_e", "olho_d", "orelha_e", "orelha_d", "ombro_e", "ombro_d", "cotovelo_e",
            "cotovelo_d", "punho_e", "punho_d", "quadril_e", "quadril_d", "joelho_e", "joelho_d",
            "tornozelo_e", "tornozelo_d"]
KP = {n: i for i, n in enumerate(NOMES_KP)}
ESQUELETO = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
             (12, 14), (14, 16), (0, 5), (0, 6)]


@dataclass
class Pessoa:
    keypoints: np.ndarray            # (17, 3): x, y, confiança
    caixa: tuple[float, float, float, float]   # x1, y1, x2, y2
    confianca: float

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.caixa
        return max(x2 - x1, 0) * max(y2 - y1, 0)


@dataclass
class Objeto:
    classe: str
    caixa: tuple[float, float, float, float]
    confianca: float


# --------------------------------------------------------------------------------------
class YoloPose:
    nome = "yolov8-pose"

    def __init__(self, pesos: str = "yolov8n-pose.pt", imgsz: int = 640, conf: float = 0.25) -> None:
        from ultralytics import YOLO

        self.modelo = YOLO(pesos)
        self.imgsz, self.conf = imgsz, conf

    def estimar(self, quadro: np.ndarray) -> list[Pessoa]:
        r = self.modelo.predict(quadro, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        if r.keypoints is None or len(r.boxes) == 0:
            return []
        kxy = r.keypoints.xy.cpu().numpy()
        kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2])
        caixas = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        return [Pessoa(np.concatenate([kxy[i], kc[i][:, None]], axis=1), tuple(caixas[i]), float(confs[i]))
                for i in range(len(caixas))]


#: OpenPose COCO-18 → COCO-17 (o pescoço, índice 1, não existe no COCO-17).
_OP18_PARA_COCO17 = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]


class OpenPose:
    nome = "openpose"

    def __init__(self, lado_max: int = 512) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from controlnet_aux.open_pose import OpenposeDetector

        self.det = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        self.lado_max = lado_max

    def estimar(self, quadro: np.ndarray) -> list[Pessoa]:
        import cv2

        h, w = quadro.shape[:2]
        esc = min(1.0, self.lado_max / max(h, w))
        img = cv2.resize(quadro, (int(w * esc), int(h * esc))) if esc < 1 else quadro
        rgb = img[:, :, ::-1].copy()                      # OpenCV é BGR; o detector espera RGB
        pessoas = []
        for pose in self.det.detect_poses(rgb):
            kps = np.zeros((17, 3))
            for j_coco, j_op in enumerate(_OP18_PARA_COCO17):
                k = pose.body.keypoints[j_op] if j_op < len(pose.body.keypoints) else None
                if k is not None:
                    kps[j_coco] = (k.x * w, k.y * h, max(float(k.score), 0.0))   # coords normalizadas
            ok = kps[:, 2] > 0
            if ok.sum() < 4:
                continue
            x1, y1 = kps[ok, 0].min(), kps[ok, 1].min()
            x2, y2 = kps[ok, 0].max(), kps[ok, 1].max()
            pessoas.append(Pessoa(kps, (x1, y1, x2, y2), float(kps[ok, 2].mean())))
        return pessoas


def criar_backend(nome: str, **kw):
    if nome == "openpose":
        return OpenPose(**kw)
    if nome == "yolo":
        return YoloPose(**kw)
    raise ValueError(f"backend de pose desconhecido: {nome}")


class DetectorObjetos:
    nome = "yolov8-det"

    def __init__(self, pesos: str = "yolov8n.pt", conf: float = 0.3, classes: set[str] | None = None) -> None:
        from ultralytics import YOLO

        self.modelo = YOLO(pesos)
        self.conf = conf
        self.classes = classes

    def detectar(self, quadro: np.ndarray) -> list[Objeto]:
        r = self.modelo.predict(quadro, conf=self.conf, verbose=False)[0]
        nomes = r.names
        out = []
        for c, b, s in zip(r.boxes.cls.cpu().numpy(), r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            nome = nomes[int(c)]
            if self.classes is None or nome in self.classes:
                out.append(Objeto(nome, tuple(float(v) for v in b), float(s)))
        return out


# --------------------------------------------------------------------------------------
@dataclass
class QuadroVideo:
    indice: int
    t: float
    imagem: np.ndarray


def ler_video(caminho: Path, fps_alvo: float | None = None, recorte_x: tuple[int, int] | None = None,
              inicio_s: float = 0.0, fim_s: float | None = None) -> Iterator[QuadroVideo]:
    """Itera quadros reamostrados para ``fps_alvo``. ``recorte_x`` corta colunas (ex.: a metade
    RGB do UR Fall, que vem lado a lado com o mapa de profundidade)."""
    import cv2

    cap = cv2.VideoCapture(str(caminho))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    passo = max(1, int(round(fps / fps_alvo))) if fps_alvo else 1
    i = 0
    if inicio_s > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(inicio_s * fps))
        i = int(inicio_s * fps)
    while True:
        if not cap.grab():               # grab() avança sem converter o quadro (mais barato)
            break
        t = i / fps
        if fim_s is not None and t > fim_s:
            break
        if i % passo == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            if recorte_x:
                img = img[:, recorte_x[0]:recorte_x[1]]
            yield QuadroVideo(i, t, img)
        i += 1
    cap.release()


def info_video(caminho: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(caminho))
    d = {"fps": cap.get(cv2.CAP_PROP_FPS), "quadros": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
         "largura": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "altura": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    d["duracao_s"] = d["quadros"] / d["fps"] if d["fps"] else 0
    return d


def pessoa_principal(pessoas: list[Pessoa], anterior: Pessoa | None = None) -> Pessoa | None:
    """Rastreamento simples: a pessoa mais próxima da anterior; sem anterior, a maior."""
    if not pessoas:
        return None
    if anterior is None:
        return max(pessoas, key=lambda p: p.area)
    cx = lambda p: np.array([(p.caixa[0] + p.caixa[2]) / 2, (p.caixa[1] + p.caixa[3]) / 2])
    return min(pessoas, key=lambda p: np.linalg.norm(cx(p) - cx(anterior)) - 0.001 * p.area ** 0.5)


@dataclass
class TrilhaPose:
    """Série temporal de pose da pessoa principal de um vídeo."""
    video: str
    backend: str
    fps_amostrado: float
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    quadro: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    kps: np.ndarray = field(default_factory=lambda: np.zeros((0, 17, 3)))
    caixas: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))
    n_pessoas: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    largura: int = 0
    altura: int = 0

    def salvar(self, caminho: Path) -> None:
        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(caminho, t=self.t, quadro=self.quadro, kps=self.kps.astype(np.float32),
                            caixas=self.caixas.astype(np.float32), n_pessoas=self.n_pessoas,
                            meta=np.array([self.video, self.backend, str(self.fps_amostrado), str(self.largura),
                                           str(self.altura)]))

    @classmethod
    def carregar(cls, caminho: Path) -> "TrilhaPose":
        z = np.load(caminho, allow_pickle=False)
        video, backend, fps, w, h = [str(x) for x in z["meta"]]
        return cls(video, backend, float(fps), z["t"], z["quadro"], z["kps"], z["caixas"], z["n_pessoas"],
                   int(w), int(h))


def extrair_trilha(caminho: Path, backend, fps_alvo: float = 10, recorte_x=None, inicio_s: float = 0.0,
                   fim_s: float | None = None, progresso: bool = False) -> TrilhaPose:
    info = info_video(caminho)
    ts, qs, ks, cs, ns = [], [], [], [], []
    anterior = None
    largura = info["largura"] if not recorte_x else recorte_x[1] - recorte_x[0]
    for q in ler_video(caminho, fps_alvo, recorte_x, inicio_s, fim_s):
        pessoas = backend.estimar(q.imagem)
        p = pessoa_principal(pessoas, anterior)
        ts.append(q.t)
        qs.append(q.indice)
        ns.append(len(pessoas))
        if p is None:
            ks.append(np.full((17, 3), np.nan))
            cs.append(np.full(4, np.nan))
        else:
            anterior = p
            ks.append(p.keypoints)
            cs.append(np.array(p.caixa))
        if progresso and len(ts) % 100 == 0:
            print(f"    {Path(caminho).name}: {len(ts)} quadros")
    return TrilhaPose(Path(caminho).name, backend.nome, fps_alvo, np.array(ts), np.array(qs, dtype=int),
                      np.array(ks) if ks else np.zeros((0, 17, 3)), np.array(cs) if cs else np.zeros((0, 4)),
                      np.array(ns, dtype=int), largura, info["altura"])
