"""NEWS2 — National Early Warning Score 2 (Royal College of Physicians, 2017).

Escore determinístico de deterioração clínica para pacientes internados. Usado aqui como a
camada de *regra* na enfermaria/UTI: é o que um enfermeiro calcularia à beira do leito, e é o
que decide a severidade do alerta. Os detectores estatísticos e de ML só antecipam ou reforçam.

Faixas de resposta (RCP):
  * 0–4  → baixo   (monitorização de rotina)
  * 3 em um único parâmetro → baixo-médio (avaliação urgente na enfermaria)
  * 5–6  → médio   (resposta urgente)
  * ≥ 7  → alto    (resposta de emergência)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def _faixa(valor: float, tabela: list[tuple[float, float, int]]) -> int | None:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return None
    for lo, hi, pts in tabela:
        if lo <= valor <= hi:
            return pts
    return None


# (mínimo, máximo, pontos) — limites inclusivos, conforme a tabela oficial
TAB_RR = [(-1e9, 8, 3), (9, 11, 1), (12, 20, 0), (21, 24, 2), (25, 1e9, 3)]
TAB_SPO2 = [(-1e9, 91, 3), (92, 93, 2), (94, 95, 1), (96, 1e9, 0)]          # escala 1
TAB_SBP = [(-1e9, 90, 3), (91, 100, 2), (101, 110, 1), (111, 219, 0), (220, 1e9, 3)]
TAB_HR = [(-1e9, 40, 3), (41, 50, 1), (51, 90, 0), (91, 110, 1), (111, 130, 2), (131, 1e9, 3)]
TAB_TEMP = [(-1e9, 35.0, 3), (35.05, 36.0, 1), (36.05, 38.0, 0), (38.05, 39.0, 1), (39.05, 1e9, 2)]


@dataclass
class ResultadoNEWS2:
    total: int
    componentes: dict[str, int]
    parametro_extremo: bool       # algum parâmetro isolado com 3 pontos
    faltantes: list[str]

    @property
    def risco(self) -> str:
        if self.total >= 7:
            return "alto"
        if self.total >= 5:
            return "médio"
        if self.parametro_extremo:
            return "baixo-médio"
        return "baixo"


def calcular_news2(rr=None, spo2=None, sbp=None, hr=None, temp=None,
                   em_oxigenio: bool = False, consciencia_alterada: bool = False) -> ResultadoNEWS2:
    comp: dict[str, int] = {}
    faltantes: list[str] = []
    for nome, valor, tab in (("RR", rr, TAB_RR), ("SPO2", spo2, TAB_SPO2), ("SBP", sbp, TAB_SBP),
                             ("HR", hr, TAB_HR), ("TEMP", temp, TAB_TEMP)):
        # arredonda como na prática clínica (inteiros; temperatura com 1 casa)
        if valor is not None and not (isinstance(valor, float) and math.isnan(valor)):
            valor = round(float(valor), 1 if nome == "TEMP" else 0)
        pts = _faixa(valor, tab)
        if pts is None:
            faltantes.append(nome)
        else:
            comp[nome] = pts
    comp["O2"] = 2 if em_oxigenio else 0
    comp["CONSC"] = 3 if consciencia_alterada else 0
    total = sum(comp.values())
    return ResultadoNEWS2(total, comp, any(v == 3 for v in comp.values()), faltantes)
