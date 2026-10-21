"""Monitoramento de movimentação no quarto: queda, imobilidade no chão e uso de zonas seguras.

Regra de queda (streaming, quadro a quadro) — duas condições em sequência:
  1. **descida rápida**: o centro do quadril desce mais que ``V_DESCIDA`` alturas-de-tronco
     por segundo (janela de 0,5 s);
  2. **postura deitada** logo depois (≤ 2 s): tronco a mais de 55° da vertical *ou* caixa da
     pessoa mais larga que alta (razão > 1,2), sustentada por ≥ 0,4 s.

Deitar *devagar* na cama ou sentar rápido numa cadeira não dispara: no primeiro falta a
descida rápida, no segundo falta a postura deitada. Se o quadril estiver dentro de uma **zona
segura** detectada pelo YOLOv8 (cama, sofá), o evento vira "deitou no leito" (informativo).

Após a queda, se a pessoa continua deitada e quase imóvel por ≥ ``T_IMOVEL`` s → **crítico**
("paciente no chão sem conseguir levantar").
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from monitor.events import Evento, Severidade
from monitor.video.cinematica import CONF_MIN
from monitor.video.pose import KP

V_DESCIDA = 1.2         # alturas de tronco por segundo
ANG_DEITADO = 55.0
RAZAO_DEITADO = 1.2
T_IMOVEL = 2.0          # nos clipes do UR Fall o vídeo acaba ~2 s após a queda; em produção: 30–60 s
ZONAS_SEGURAS = {"bed", "couch"}


@dataclass
class EstadoQuadro:
    t: float
    quadril: np.ndarray | None
    tronco_px: float | None
    angulo_tronco: float | None
    razao_caixa: float | None


def medir(t: float, kps: np.ndarray | None, caixa) -> EstadoQuadro:
    if kps is None or np.isnan(kps).all():
        return EstadoQuadro(t, None, None, None, None)
    ok = kps[:, 2] >= CONF_MIN
    def p(n):
        i = KP[n]
        return kps[i, :2] if ok[i] else None
    oe, od, qe, qd = p("ombro_e"), p("ombro_d"), p("quadril_e"), p("quadril_d")
    ombros = [x for x in (oe, od) if x is not None]
    quadris = [x for x in (qe, qd) if x is not None]
    quadril = np.mean(quadris, axis=0) if quadris else None
    ombro = np.mean(ombros, axis=0) if ombros else None
    tronco = ang = None
    if quadril is not None and ombro is not None:
        v = ombro - quadril
        tronco = float(np.linalg.norm(v))
        ang = float(np.degrees(np.arctan2(abs(v[0]), -v[1])))
    razao = None
    if caixa is not None and not np.isnan(caixa).any():
        w, h = caixa[2] - caixa[0], caixa[3] - caixa[1]
        razao = float(w / h) if h > 0 else None
    return EstadoQuadro(t, quadril, tronco, ang, razao)


@dataclass
class DetectorQuedas:
    paciente_id: str
    zonas_seguras: list[tuple[float, float, float, float]] = field(default_factory=list)
    t0: float = 0.0
    _hist: deque = field(default_factory=lambda: deque(maxlen=64))
    _ref_tronco: deque = field(default_factory=lambda: deque(maxlen=50))
    _descida_t: float | None = None
    _deitado_desde: float | None = None
    _queda_t: float | None = None
    _alertou_imovel: bool = False
    _ultimo_evento: float = -1e9
    trilha: list = field(default_factory=list)

    def _na_zona_segura(self, ponto) -> bool:
        if ponto is None:
            return False
        return any(x1 <= ponto[0] <= x2 and y1 <= ponto[1] <= y2 for x1, y1, x2, y2 in self.zonas_seguras)

    def atualizar(self, t: float, kps, caixa) -> list[Evento]:
        e = medir(t, kps, caixa)
        eventos: list[Evento] = []
        if e.tronco_px and (e.angulo_tronco or 90) < 30:
            self._ref_tronco.append(e.tronco_px)     # tamanho do tronco medido em pé
        ref = float(np.median(self._ref_tronco)) if self._ref_tronco else (e.tronco_px or 100.0)
        self._hist.append(e)

        # 1) descida rápida do quadril
        vel = None
        anteriores = [h for h in self._hist if h.quadril is not None and t - 0.5 <= h.t < t]
        if e.quadril is not None and anteriores:
            h0 = anteriores[0]
            if t - h0.t > 0.1:
                vel = (e.quadril[1] - h0.quadril[1]) / (t - h0.t) / ref
                # guarda o *início* da descida (a velocidade média fica alta por ~0,5 s depois)
                if vel > V_DESCIDA and (self._descida_t is None or t - self._descida_t > 3.0):
                    self._descida_t = t
        # 2) postura deitada
        deitado = (e.angulo_tronco is not None and e.angulo_tronco > ANG_DEITADO) or \
                  (e.razao_caixa is not None and e.razao_caixa > RAZAO_DEITADO)
        # pose de pessoa deitada falha muito (oclusão, ângulo de câmera): quadro sem medida
        # confiável mantém o estado anterior em vez de "levantar" o paciente
        sem_medida = e.angulo_tronco is None and (e.razao_caixa is None or 0.8 <= e.razao_caixa <= RAZAO_DEITADO)
        if sem_medida and self._deitado_desde is not None:
            deitado = True
        if deitado:
            self._deitado_desde = self._deitado_desde if self._deitado_desde is not None else t
        else:
            self._deitado_desde = None
            if self._queda_t is not None and e.angulo_tronco is not None and e.angulo_tronco < 35:
                self._queda_t, self._alertou_imovel = None, False      # levantou

        self.trilha.append({"t": t, "vel_quadril": vel, "angulo_tronco": e.angulo_tronco, "razao": e.razao_caixa,
                            "deitado": deitado})

        if (self._deitado_desde is not None and t - self._deitado_desde >= 0.4 and self._descida_t is not None
                and -1.0 <= self._deitado_desde - self._descida_t <= 2.0 and self._queda_t is None
                and t - self._ultimo_evento > 5):
            self._ultimo_evento = t
            if self._na_zona_segura(e.quadril):
                eventos.append(Evento(self.paciente_id, "movimentacao", "deitou_no_leito", Severidade.INFO,
                                      self.t0 + t, "Paciente deitou rapidamente, dentro da zona do leito",
                                      "pose", {"angulo_tronco": e.angulo_tronco}))
            else:
                self._queda_t = t
                eventos.append(Evento(
                    self.paciente_id, "movimentacao", "queda", Severidade.URGENTE, self.t0 + t,
                    f"Queda detectada: descida rápida do quadril seguida de postura deitada fora do leito "
                    f"(tronco {e.angulo_tronco or 0:.0f}° da vertical)", "pose",
                    {"angulo_tronco": e.angulo_tronco, "razao_caixa": e.razao_caixa,
                     "descida_t": self.t0 + self._descida_t}))
        # 3) imobilidade após a queda
        if self._queda_t is not None and not self._alertou_imovel and deitado and t - self._queda_t >= T_IMOVEL:
            recentes = [h.quadril for h in self._hist if h.quadril is not None and h.t >= t - T_IMOVEL]
            if len(recentes) >= 3:
                desloc = np.max(np.linalg.norm(np.array(recentes) - recentes[0], axis=1)) / ref
                if desloc < 0.5:
                    self._alertou_imovel = True
                    eventos.append(Evento(
                        self.paciente_id, "movimentacao", "imobilidade_pos_queda", Severidade.CRITICO, self.t0 + t,
                        f"Paciente permanece no chão, sem se levantar, há {t - self._queda_t:.0f} s após a queda",
                        "pose", {"segundos_no_chao": round(t - self._queda_t, 1), "deslocamento_rel": round(float(desloc), 2)}))
        return eventos


def zonas_de_objetos(objetos) -> list[tuple[float, float, float, float]]:
    return [o.caixa for o in objetos if o.classe in ZONAS_SEGURAS]
