"""Detectores de anomalia em séries temporais de sinais vitais — todos em modo *streaming*.

Três camadas complementares, do mais explicável ao mais sensível:

1. :class:`DetectorRegras` — limiares clínicos *sustentados* (ex.: PAM < 65 mmHg por ≥ 60 s).
   É a camada que decide severidade. Exigir sustentação elimina a maioria dos falsos alarmes
   por artefato, que é o principal problema de monitores reais (fadiga de alarme).
2. :class:`DetectorZScore` — z-score robusto (mediana/MAD) dos últimos 30 s contra a linha de
   base do próprio paciente nos 10 min anteriores. Pega mudanças *abruptas* mesmo dentro da
   faixa normal (FC que sobe de 60 para 95 em um minuto).
3. :class:`DetectorIForest` — Isolation Forest sobre janelas multivariadas (média, desvio,
   tendência de cada canal + índice de choque + pressão de pulso). Pega combinações atípicas
   que nenhum canal isolado denuncia.

Batch e tempo real usam **o mesmo código**: ``processar_df`` só itera ``atualizar`` linha a
linha, que é exatamente o que o dashboard faz no replay.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from monitor.events import Evento, Severidade
from monitor.vitals.canais import CANAIS, NOMES, UNIDADES
from monitor.vitals.news2 import calcular_news2


def _nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


# ======================================================================================
# 1. Regras clínicas sustentadas
# ======================================================================================
@dataclass(frozen=True)
class RegraLimiar:
    tipo: str
    canal: str
    op: str                 # "<" ou ">"
    limite: float
    sustentado_s: float
    severidade: Severidade
    descricao: str

    def viola(self, v: float) -> bool:
        return (v < self.limite) if self.op == "<" else (v > self.limite)


#: Perfil intraoperatório / UTI com monitorização contínua (VitalDB).
#: Referências: hipotensão intraoperatória PAM < 65 (Sessler 2019; Wesselink 2018),
#: dessaturação SpO2 < 90–92, taqui/bradicardia sustentadas.
PERFIL_CONTINUO: list[RegraLimiar] = [
    RegraLimiar("hipotensao", "MAP", "<", 65, 60, Severidade.URGENTE, "PA média < 65 mmHg por ≥ 1 min"),
    RegraLimiar("hipotensao_grave", "MAP", "<", 55, 60, Severidade.CRITICO, "PA média < 55 mmHg por ≥ 1 min"),
    RegraLimiar("dessaturacao", "SPO2", "<", 92, 30, Severidade.URGENTE, "SpO₂ < 92% por ≥ 30 s"),
    RegraLimiar("dessaturacao_grave", "SPO2", "<", 88, 30, Severidade.CRITICO, "SpO₂ < 88% por ≥ 30 s"),
    RegraLimiar("taquicardia", "HR", ">", 120, 60, Severidade.URGENTE, "FC > 120 bpm por ≥ 1 min"),
    RegraLimiar("bradicardia", "HR", "<", 45, 60, Severidade.URGENTE, "FC < 45 bpm por ≥ 1 min"),
    RegraLimiar("hipertensao", "SBP", ">", 180, 120, Severidade.ATENCAO, "PA sistólica > 180 mmHg por ≥ 2 min"),
    RegraLimiar("hipotermia", "TEMP", "<", 35.5, 600, Severidade.ATENCAO, "Temperatura < 35,5 °C por ≥ 10 min"),
    RegraLimiar("febre", "TEMP", ">", 38.3, 600, Severidade.ATENCAO, "Temperatura > 38,3 °C por ≥ 10 min"),
]


#: Perfil para registros horários de enfermaria/UTI (MIMIC-IV): sem exigência de sustentação —
#: cada registro já é uma média/medida validada pela enfermagem.
PERFIL_HORARIO: list[RegraLimiar] = [
    RegraLimiar("hipotensao", "MAP", "<", 65, 0, Severidade.URGENTE, "PA média < 65 mmHg"),
    RegraLimiar("hipotensao_grave", "MAP", "<", 55, 0, Severidade.CRITICO, "PA média < 55 mmHg"),
    RegraLimiar("dessaturacao", "SPO2", "<", 92, 0, Severidade.URGENTE, "SpO₂ < 92%"),
    RegraLimiar("dessaturacao_grave", "SPO2", "<", 88, 0, Severidade.CRITICO, "SpO₂ < 88%"),
    RegraLimiar("taquicardia", "HR", ">", 120, 0, Severidade.URGENTE, "FC > 120 bpm"),
    RegraLimiar("bradicardia", "HR", "<", 45, 0, Severidade.URGENTE, "FC < 45 bpm"),
    RegraLimiar("taquipneia", "RR", ">", 24, 0, Severidade.URGENTE, "FR > 24 irpm"),
    RegraLimiar("bradipneia", "RR", "<", 10, 0, Severidade.URGENTE, "FR < 10 irpm"),
    RegraLimiar("febre", "TEMP", ">", 38.3, 0, Severidade.ATENCAO, "Temperatura > 38,3 °C"),
]


@dataclass
class _EstadoRegra:
    inicio: float | None = None
    disparou: bool = False
    extremo: float | None = None
    evento: Evento | None = None


class DetectorRegras:
    """Dispara uma vez por episódio quando a condição se sustenta pelo tempo mínimo."""

    def __init__(self, paciente_id: str, regras: Iterable[RegraLimiar] = PERFIL_CONTINUO,
                 max_gap_s: float = 30.0) -> None:
        self.paciente_id = paciente_id
        self.regras = list(regras)
        self.max_gap_s = max_gap_s
        self._estado = {r.tipo: _EstadoRegra() for r in self.regras}
        self._ultimo_t: dict[str, float] = {}
        self.episodios: list[dict] = []     # para avaliação: início, alerta, fim, extremo

    def atualizar(self, t: float, amostra: dict) -> list[Evento]:
        eventos: list[Evento] = []
        for r in self.regras:
            st = self._estado[r.tipo]
            v = amostra.get(r.canal)
            if _nan(v):
                # lacuna longa encerra o episódio (não sabemos o que aconteceu)
                if st.inicio is not None and t - self._ultimo_t.get(r.canal, t) > self.max_gap_s:
                    self._encerrar(r, st, self._ultimo_t.get(r.canal, t))
                continue
            self._ultimo_t[r.canal] = t
            if r.viola(v):
                if st.inicio is None:
                    st.inicio, st.extremo = t, v
                st.extremo = min(st.extremo, v) if r.op == "<" else max(st.extremo, v)
                if not st.disparou and t - st.inicio >= r.sustentado_s:
                    st.disparou = True
                    st.evento = Evento(
                        paciente_id=self.paciente_id, origem="vitais", tipo=r.tipo,
                        severidade=r.severidade, t=t, detector="regra",
                        descricao=f"{r.descricao} (atual {v:.0f} {UNIDADES[r.canal]})",
                        evidencias={"canal": r.canal, "valor": round(float(v), 1), "limite": r.limite,
                                    "inicio_t": st.inicio, "sustentado_s": round(t - st.inicio, 1)},
                    )
                    eventos.append(st.evento)
            elif st.inicio is not None:
                self._encerrar(r, st, t)
        return eventos

    def _encerrar(self, r: RegraLimiar, st: _EstadoRegra, t_fim: float) -> None:
        if st.disparou:
            self.episodios.append({"tipo": r.tipo, "canal": r.canal, "inicio": st.inicio,
                                   "alerta": st.evento.t if st.evento else None, "fim": t_fim,
                                   "extremo": st.extremo, "severidade": int(r.severidade)})
            if st.evento is not None:
                st.evento.duracao_s = round(t_fim - st.inicio, 1)
                st.evento.evidencias["extremo"] = round(float(st.extremo), 1)
        self._estado[r.tipo] = _EstadoRegra()

    def finalizar(self, t: float) -> None:
        for r in self.regras:
            st = self._estado[r.tipo]
            if st.inicio is not None:
                self._encerrar(r, st, t)


# ======================================================================================
# 2. Z-score robusto contra a linha de base do próprio paciente
# ======================================================================================
#: Piso do desvio robusto por canal — evita z explosivo quando o sinal está "parado".
PISO_MAD = {"HR": 3.0, "SPO2": 1.0, "SBP": 5.0, "DBP": 4.0, "MAP": 4.0, "RR": 2.0, "TEMP": 0.2}


class DetectorZScore:
    def __init__(self, paciente_id: str, canais=("HR", "SPO2", "MAP", "RR"), janela_base_s: float = 600,
                 janela_recente_s: float = 30, limiar: float = 4.0, passo_s: float = 10,
                 refratario_s: float = 600, cobertura_min: float = 0.5, persistencia: int = 2) -> None:
        self.paciente_id = paciente_id
        self.canais = canais
        self.base_s, self.rec_s = janela_base_s, janela_recente_s
        self.limiar, self.passo_s, self.refratario_s = limiar, passo_s, refratario_s
        self.cobertura_min = cobertura_min
        self._buf = {c: deque() for c in canais}
        self._ultimo_passo = -1e18
        self._ultimo_alerta = {c: -1e18 for c in canais}
        self.persistencia = persistencia
        self._seguidas = {c: 0 for c in canais}
        self._sinal = {c: 0 for c in canais}
        self._dt: float | None = None
        self._t_ant: float | None = None

    def atualizar(self, t: float, amostra: dict) -> list[Evento]:
        if self._t_ant is not None and self._dt is None and t > self._t_ant:
            self._dt = t - self._t_ant
        self._t_ant = t
        for c in self.canais:
            v = amostra.get(c)
            if not _nan(v):
                self._buf[c].append((t, float(v)))
            while self._buf[c] and self._buf[c][0][0] < t - self.base_s - self.rec_s:
                self._buf[c].popleft()
        if t - self._ultimo_passo < self.passo_s or self._dt is None:
            return []
        self._ultimo_passo = t
        eventos = []
        for c in self.canais:
            if t - self._ultimo_alerta[c] < self.refratario_s:
                continue
            arr = np.array(self._buf[c])
            if len(arr) == 0:
                continue
            base = arr[arr[:, 0] < t - self.rec_s, 1]
            rec = arr[arr[:, 0] >= t - self.rec_s, 1]
            if len(base) < self.cobertura_min * self.base_s / self._dt or \
               len(rec) < self.cobertura_min * self.rec_s / self._dt:
                continue
            med = float(np.median(base))
            mad = 1.4826 * float(np.median(np.abs(base - med)))
            z = (float(np.median(rec)) - med) / max(mad, PISO_MAD[c])
            # persistência: o desvio precisa aparecer em `persistencia` passos seguidos (mesmo sentido)
            sinal = int(np.sign(z)) if abs(z) >= self.limiar else 0
            self._seguidas[c] = self._seguidas[c] + 1 if sinal and sinal == self._sinal[c] else (1 if sinal else 0)
            self._sinal[c] = sinal
            if sinal and self._seguidas[c] >= self.persistencia:
                self._ultimo_alerta[c] = t
                direcao = "subida" if z > 0 else "queda"
                eventos.append(Evento(
                    paciente_id=self.paciente_id, origem="vitais", tipo=f"desvio_{c.lower()}",
                    severidade=Severidade.ATENCAO, t=t, detector="estatistico",
                    descricao=(f"{direcao.capitalize()} abrupta de {NOMES[c]}: {np.median(rec):.0f} "
                               f"{UNIDADES[c]} vs. linha de base {med:.0f} (z={z:+.1f})"),
                    evidencias={"canal": c, "z": round(z, 2), "mediana_recente": round(float(np.median(rec)), 1),
                                "mediana_base": round(med, 1), "desvio_robusto": round(mad, 2)},
                ))
        return eventos


# ======================================================================================
# 3. Isolation Forest multivariado
# ======================================================================================
CANAIS_ML = ["HR", "SPO2", "SBP", "DBP", "MAP", "RR"]


BASE_IFOREST_S = 600.0


def linha_de_base(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Mediana e desvio robusto (1,4826·MAD, com piso clínico) de cada canal."""
    base = {}
    for c in CANAIS_ML:
        v = df[c].to_numpy(dtype=float)
        v = v[~np.isnan(v)]
        if len(v) >= 10:
            med = float(np.median(v))
            base[c] = (med, max(1.4826 * float(np.median(np.abs(v - med))), PISO_MAD[c]))
    return base


def features_janela(janela: pd.DataFrame, base: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
    """Vetor de features de uma janela, **relativo à linha de base do próprio paciente**.

    Para cada canal: ``_z`` (mediana da janela vs. mediana dos 10 min anteriores, em desvios
    robustos), ``_tend`` (tendência por minuto, na mesma escala) e ``_var`` (variabilidade da
    janela / desvio da base). Mais ``_media`` absoluta (usada só se ``usar_absolutos``).

    Normalizar pelo próprio paciente é o que permite ao modelo multivariado aprender *como os
    canais costumam variar juntos* (FC sobe com PA na dor; FC sobe e PA cai no choque) em vez
    de aprender "o que é um paciente típico".
    """
    f: dict[str, float] = {}
    base = base or {}
    t = janela["t_s"].to_numpy(dtype=float)
    for c in CANAIS_ML:
        v = janela[c].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        med_b, esc = base.get(c, (np.nan, PISO_MAD[c]))
        if ok.sum() >= 3:
            f[f"{c}_media"] = float(v[ok].mean())
            tt = t[ok] - t[ok].mean()
            tend = float((tt * (v[ok] - v[ok].mean())).sum() / max((tt ** 2).sum(), 1e-9) * 60)
            f[f"{c}_tend"] = tend / esc
            f[f"{c}_var"] = float(v[ok].std()) / esc
            f[f"{c}_z"] = (float(np.median(v[ok])) - med_b) / esc if not np.isnan(med_b) else 0.0
        else:
            f[f"{c}_media"] = np.nan
            f[f"{c}_tend"] = f[f"{c}_var"] = f[f"{c}_z"] = 0.0
    sbp, hr = f["SBP_media"], f["HR_media"]
    f["indice_choque"] = hr / sbp if sbp and not math.isnan(sbp) and sbp > 0 else np.nan
    f["pressao_pulso"] = f["SBP_media"] - f["DBP_media"]
    return f


def janelas(df: pd.DataFrame, largura_s: float = 60, passo_s: float = 30) -> pd.DataFrame:
    linhas = []
    t0, tf = float(df["t_s"].min()), float(df["t_s"].max())
    ts = df["t_s"].to_numpy()
    ini = t0
    while ini + largura_s <= tf:
        a, b = np.searchsorted(ts, [ini, ini + largura_s])
        a0 = np.searchsorted(ts, ini - BASE_IFOREST_S)
        f = features_janela(df.iloc[a:b], linha_de_base(df.iloc[a0:a]) if a > a0 else None)
        f["t_fim"] = ini + largura_s
        linhas.append(f)
        ini += passo_s
    return pd.DataFrame(linhas)


class ModeloIForest:
    """Detector multivariado não supervisionado + padronização + explicação por feature.

    ``metodo``:
      * ``"iforest"``  — Isolation Forest (Liu et al., 2008)
      * ``"elliptic"`` — Envelope elíptico com covariância robusta (MCD): distância de
        Mahalanobis. Mais sensível quando a anomalia é um desvio forte em poucas features
        correlacionadas — o caso típico de sinais vitais (ver relatório, seção de resultados).
    Em ambos ``score`` é maior = mais normal; o limiar é o quantil ``quantil_alarme`` do treino.
    Serializável com joblib.
    """

    def __init__(self, quantil_alarme: float = 0.01, random_state: int = 42, usar_absolutos: bool = False,
                 metodo: str = "iforest") -> None:
        self.metodo = metodo
        self.quantil_alarme = quantil_alarme
        self.random_state = random_state
        self.usar_absolutos = usar_absolutos
        self.colunas: list[str] = []

    def fit(self, X: pd.DataFrame) -> "ModeloIForest":
        """``usar_absolutos=False`` (padrão) treina só com variações *relativas ao próprio paciente*
        (delta vs. linha de base, tendência, variabilidade). Médias absolutas variam muito entre
        pacientes e diluem o sinal de mudança — e níveis absolutos já são cobertos pelas regras."""
        from sklearn.ensemble import IsolationForest

        X = X.drop(columns=[c for c in ("t_fim",) if c in X])
        if not self.usar_absolutos:
            X = X.drop(columns=[c for c in X if c.endswith("_media") or c in ("indice_choque", "pressao_pulso")])
        self.colunas = list(X.columns)
        self.mediana = X.median()
        Xi = X.fillna(self.mediana)
        self.media, self.desvio = Xi.mean(), Xi.std().replace(0, 1.0)
        Z = (Xi - self.media) / self.desvio
        if self.metodo == "elliptic":
            from sklearn.covariance import EllipticEnvelope

            self.iforest = EllipticEnvelope(support_fraction=0.9, contamination=0.01,
                                            random_state=self.random_state)
        else:
            self.iforest = IsolationForest(n_estimators=100, max_samples=512, random_state=self.random_state)
        self.iforest.fit(Z.to_numpy())
        scores = self.iforest.score_samples(Z.to_numpy())
        self.limiar = float(np.quantile(scores, self.quantil_alarme))
        self.n_treino = len(X)
        return self

    def _z(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X[self.colunas].fillna(self.mediana) - self.media) / self.desvio

    def _z_np(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "_cache_np"):
            self._cache_np = (self.mediana[self.colunas].to_numpy(float), self.media[self.colunas].to_numpy(float),
                              self.desvio[self.colunas].to_numpy(float))
        med, mu, sd = self._cache_np
        A = X[self.colunas].to_numpy(dtype=float)
        A = np.where(np.isnan(A), med, A)
        return (A - mu) / sd

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.iforest.score_samples(self._z_np(X))

    def explicar(self, linha: pd.Series, k: int = 3) -> list[tuple[str, float]]:
        z = self._z(linha.to_frame().T).iloc[0]
        top = z.abs().sort_values(ascending=False).head(k)
        return [(nome, round(float(z[nome]), 2)) for nome in top.index]

    def salvar(self, caminho: Path) -> None:
        import joblib

        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, caminho)

    @staticmethod
    def carregar(caminho: Path) -> "ModeloIForest":
        import joblib

        return joblib.load(caminho)


class DetectorIForest:
    def __init__(self, paciente_id: str, modelo: ModeloIForest, largura_s: float = 60, passo_s: float = 30,
                 consecutivas: int = 2, refratario_s: float = 600) -> None:
        self.paciente_id, self.modelo = paciente_id, modelo
        self.largura_s, self.passo_s = largura_s, passo_s
        self.consecutivas, self.refratario_s = consecutivas, refratario_s
        self._buf: deque = deque()
        self._proximo: float | None = None
        self._seguidas = 0
        self._ultimo_alerta = -1e18
        self.scores: list[tuple[float, float]] = []

    def atualizar(self, t: float, amostra: dict) -> list[Evento]:
        self._buf.append({"t_s": t, **{c: amostra.get(c, np.nan) for c in CANAIS_ML}})
        while self._buf and self._buf[0]["t_s"] < t - self.largura_s - BASE_IFOREST_S:
            self._buf.popleft()
        if self._proximo is None:
            self._proximo = t + self.largura_s
        if t < self._proximo:
            return []
        self._proximo = t + self.passo_s
        tudo = pd.DataFrame(list(self._buf))
        na_janela = tudo["t_s"] > t - self.largura_s
        base = linha_de_base(tudo[~na_janela]) if (~na_janela).any() else None
        f = pd.Series(features_janela(tudo[na_janela], base))
        if f[[f"{c}_media" for c in CANAIS_ML]].isna().sum() > 2:   # dados insuficientes
            self._seguidas = 0
            return []
        s = float(self.modelo.score(f.to_frame().T)[0])
        self.scores.append((t, s))
        self._seguidas = self._seguidas + 1 if s < self.modelo.limiar else 0
        if self._seguidas >= self.consecutivas and t - self._ultimo_alerta >= self.refratario_s:
            self._ultimo_alerta = t
            expl = self.modelo.explicar(f)
            texto = ", ".join(f"{n} (z={z:+.1f})" for n, z in expl)
            return [Evento(
                paciente_id=self.paciente_id, origem="vitais", tipo="padrao_multivariado_atipico",
                severidade=Severidade.ATENCAO, t=t, detector="ml",
                descricao=f"Combinação atípica de sinais vitais — principais desvios: {texto}",
                evidencias={"score": round(s, 4), "limiar": round(self.modelo.limiar, 4),
                            "explicacao": expl, "janelas_consecutivas": self._seguidas},
            )]
        return []


# ======================================================================================
# 4. NEWS2 para dados horários de enfermaria/UTI (MIMIC-IV)
# ======================================================================================
class DetectorNEWS2:
    """Calcula NEWS2 a cada registro e alerta quando a faixa de risco fica **pior que a pior faixa
    das últimas 6 h** (histerese: oscilar entre médio e alto não gera um alerta por hora) ou
    quando o escore sobe ≥ 3 em até 12 h (deterioração), mesmo abaixo do limiar absoluto."""

    SEVERIDADE = {"baixo": Severidade.INFO, "baixo-médio": Severidade.ATENCAO,
                  "médio": Severidade.URGENTE, "alto": Severidade.CRITICO}
    ORDEM = ["baixo", "baixo-médio", "médio", "alto"]

    def __init__(self, paciente_id: str, validade_s: float = 4 * 3600, janela_tendencia_s: float = 12 * 3600,
                 histerese_s: float = 6 * 3600):
        self.paciente_id = paciente_id
        self.histerese_s = histerese_s
        self._riscos: deque = deque()
        self.validade_s, self.janela_tendencia_s = validade_s, janela_tendencia_s
        self._ultimo: dict[str, tuple[float, float]] = {}
        self._hist: deque = deque()
        self.serie: list[tuple[float, int]] = []

    def atualizar(self, t: float, amostra: dict) -> list[Evento]:
        for c in ("RR", "SPO2", "SBP", "HR", "TEMP"):
            v = amostra.get(c)
            if not _nan(v):
                self._ultimo[c] = (t, float(v))
        val = {c: v for c, (tc, v) in self._ultimo.items() if t - tc <= self.validade_s}
        r = calcular_news2(rr=val.get("RR"), spo2=val.get("SPO2"), sbp=val.get("SBP"), hr=val.get("HR"),
                           temp=val.get("TEMP"), em_oxigenio=bool(amostra.get("O2") or 0),
                           consciencia_alterada=bool(amostra.get("CONSC_ALTERADA") or 0))
        if len(r.faltantes) > 2:
            return []
        self.serie.append((t, r.total))
        self._hist.append((t, r.total))
        while self._hist and self._hist[0][0] < t - self.janela_tendencia_s:
            self._hist.popleft()
        eventos = []
        evid = {"news2": r.total, "componentes": r.componentes, "risco": r.risco}
        nivel = self.ORDEM.index(r.risco)
        while self._riscos and self._riscos[0][0] < t - self.histerese_s:
            self._riscos.popleft()
        pior_recente = max((n for _, n in self._riscos), default=0)
        self._riscos.append((t, nivel))
        if nivel > pior_recente:
            eventos.append(Evento(
                paciente_id=self.paciente_id, origem="vitais", tipo="news2_" + r.risco.replace("-", "_"),
                severidade=self.SEVERIDADE[r.risco], t=t, detector="regra",
                descricao=f"NEWS2 = {r.total} (risco {r.risco})", evidencias=evid))
        minimo = min(s for _, s in self._hist)
        if r.total - minimo >= 3 and not eventos and r.total >= 3:
            eventos.append(Evento(
                paciente_id=self.paciente_id, origem="vitais", tipo="news2_deterioracao",
                severidade=Severidade.ATENCAO, t=t, detector="regra",
                descricao=f"NEWS2 subiu de {minimo} para {r.total} em até 12 h", evidencias=evid))
            self._hist.clear()
            self._hist.append((t, r.total))
        return eventos


# ======================================================================================
# Orquestrador
# ======================================================================================
class MonitorSinaisVitais:
    """Junta as camadas. ``atualizar`` é chamado a cada amostra do monitor (ou do replay)."""

    def __init__(self, paciente_id: str, modelo_iforest: ModeloIForest | None = None,
                 regras: Iterable[RegraLimiar] = PERFIL_CONTINUO, modelo_hipotensao=None) -> None:
        from monitor.vitals.predicao import PreditorHipotensao

        self.regras = DetectorRegras(paciente_id, regras)
        self.zscore = DetectorZScore(paciente_id)
        self.iforest = DetectorIForest(paciente_id, modelo_iforest) if modelo_iforest else None
        self.preditor = PreditorHipotensao(paciente_id, modelo_hipotensao) if modelo_hipotensao else None

    def atualizar(self, t: float, amostra: dict) -> list[Evento]:
        ev = self.regras.atualizar(t, amostra) + self.zscore.atualizar(t, amostra)
        if self.iforest:
            ev += self.iforest.atualizar(t, amostra)
        if self.preditor:
            ev += self.preditor.atualizar(t, amostra)
        return ev

    def processar_df(self, df: pd.DataFrame) -> list[Evento]:
        eventos: list[Evento] = []
        cols = ["t_s", *[c for c in CANAIS if c in df]]
        for linha in df[cols].itertuples(index=False):
            d = linha._asdict()
            eventos += self.atualizar(float(d.pop("t_s")), d)
        if len(df):
            self.regras.finalizar(float(df["t_s"].iloc[-1]))
        return eventos
