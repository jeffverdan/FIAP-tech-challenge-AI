"""Predição precoce de hipotensão — "o alerta antes do alarme".

Modelo supervisionado (Gradient Boosting) que, a cada 30 s, estima a probabilidade de um
episódio de hipotensão sustentada (PAM < 65 mmHg por ≥ 1 min) **começar nos próximos 10 min**,
usando só os últimos 10 min de sinais vitais. É a mesma ideia do *Hypotension Prediction Index*
comercial, com dados abertos (VitalDB).

Rótulos são gerados automaticamente a partir das próprias séries (não há anotação manual):
amostras em que a PAM já está < 65 são excluídas — prever o presente não é predição.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from monitor.events import Evento, Severidade

HORIZONTE_S = 600.0
HISTORICO_S = 600.0
PASSO_S = 30.0
LIMIAR_PAM = 65.0
SUSTENTACAO_S = 60.0


def _mediana(v: np.ndarray) -> float:
    v = v[~np.isnan(v)]
    return float(np.median(v)) if len(v) else np.nan


def _inclinacao(t: np.ndarray, v: np.ndarray) -> float:
    ok = ~np.isnan(v)
    if ok.sum() < 5:
        return np.nan
    tt = t[ok] - t[ok].mean()
    return float((tt * (v[ok] - v[ok].mean())).sum() / max((tt ** 2).sum(), 1e-9) * 60)


def features_preditivas(hist: pd.DataFrame, t: float) -> dict[str, float]:
    """Features no instante ``t`` a partir do histórico (colunas ``t_s``, HR, SBP, DBP, MAP, SPO2)."""
    ts = hist["t_s"].to_numpy(dtype=float)
    f: dict[str, float] = {}

    def fatia(seg: float) -> pd.DataFrame:
        return hist.iloc[np.searchsorted(ts, t - seg, side="right"): np.searchsorted(ts, t, side="right")]

    agora = fatia(30)
    for c in ("MAP", "SBP", "DBP", "HR", "SPO2"):
        f[f"{c}_atual"] = _mediana(agora[c].to_numpy(dtype=float))
    for seg in (60, 180, 300, 600):
        anterior = hist.iloc[np.searchsorted(ts, t - seg - 30): np.searchsorted(ts, t - seg)]
        f[f"MAP_delta_{seg // 60}min"] = f["MAP_atual"] - _mediana(anterior["MAP"].to_numpy(dtype=float))
    h5 = fatia(300)
    m5 = h5["MAP"].to_numpy(dtype=float)
    f["MAP_min_5min"] = float(np.nanmin(m5)) if np.isfinite(m5).any() else np.nan
    f["MAP_desvio_5min"] = float(np.nanstd(m5)) if np.isfinite(m5).any() else np.nan
    f["MAP_inclinacao_5min"] = _inclinacao(h5["t_s"].to_numpy(dtype=float), m5)
    f["HR_inclinacao_5min"] = _inclinacao(h5["t_s"].to_numpy(dtype=float), h5["HR"].to_numpy(dtype=float))
    f["SBP_inclinacao_5min"] = _inclinacao(h5["t_s"].to_numpy(dtype=float), h5["SBP"].to_numpy(dtype=float))
    f["pressao_pulso"] = f["SBP_atual"] - f["DBP_atual"]
    f["indice_choque"] = f["HR_atual"] / f["SBP_atual"] if f["SBP_atual"] and f["SBP_atual"] > 0 else np.nan
    h10 = fatia(600)["MAP"].to_numpy(dtype=float)
    f["frac_abaixo_70_10min"] = float(np.nanmean(h10 < 70)) if np.isfinite(h10).any() else np.nan
    return f


def inicios_hipotensao(df: pd.DataFrame) -> np.ndarray:
    """Instantes de início de cada episódio PAM < 65 sustentado por ≥ 60 s."""
    t = df["t_s"].to_numpy(dtype=float)
    baixo = (df["MAP"] < LIMIAR_PAM).to_numpy()
    inicios, i, n = [], 0, len(t)
    while i < n:
        if baixo[i]:
            j = i
            while j + 1 < n and baixo[j + 1]:
                j += 1
            if t[j] - t[i] >= SUSTENTACAO_S:
                inicios.append(t[i])
            i = j + 1
        else:
            i += 1
    return np.array(inicios)


def montar_dataset(df: pd.DataFrame, caso: int | str) -> pd.DataFrame:
    """Amostras a cada 30 s com rótulo "hipotensão começa em (t, t+10 min]"."""
    inicios = inicios_hipotensao(df)
    linhas = []
    t_fim = float(df["t_s"].iloc[-1])
    for t in np.arange(HISTORICO_S, t_fim - HORIZONTE_S, PASSO_S):
        f = features_preditivas(df, t)
        if np.isnan(f["MAP_atual"]) or f["MAP_atual"] < LIMIAR_PAM:
            continue
        # exclui amostras logo após um episódio (o "rebote" contamina o rótulo)
        if len(inicios) and ((inicios <= t) & (inicios > t - 300)).any():
            continue
        f["y"] = int(((inicios > t) & (inicios <= t + HORIZONTE_S)).any())
        f["t"] = t
        f["caso"] = caso
        linhas.append(f)
    return pd.DataFrame(linhas)


@dataclass
class ModeloHipotensao:
    modelo: object
    colunas: list[str]
    limiar: float           # probabilidade de alerta (escolhida no treino para ~80% de sensibilidade)
    metricas_treino: dict

    def prob(self, X: pd.DataFrame) -> np.ndarray:
        return self.modelo.predict_proba(X[self.colunas])[:, 1]

    def salvar(self, caminho: Path) -> None:
        import joblib

        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, caminho)

    @staticmethod
    def carregar(caminho: Path) -> "ModeloHipotensao":
        import joblib

        return joblib.load(caminho)


def treinar(dados: pd.DataFrame, sensibilidade_alvo: float = 0.8, random_state: int = 42) -> ModeloHipotensao:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold, cross_val_predict

    colunas = [c for c in dados.columns if c not in ("y", "t", "caso")]
    # árvores rasas e folhas grandes: com ~70 pacientes, modelos profundos decoram o paciente
    # (AUROC no teste: profundidade 2 = 0,79; profundidade livre + class_weight = 0,72)
    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_depth=2, min_samples_leaf=200,
                                         random_state=random_state)
    # limiar escolhido com predições fora-da-amostra (validação cruzada agrupada por caso)
    oof = cross_val_predict(clf, dados[colunas], dados["y"], groups=dados["caso"],
                            cv=GroupKFold(n_splits=5), method="predict_proba")[:, 1]
    pos = np.sort(oof[dados["y"].to_numpy() == 1])
    limiar = float(pos[int((1 - sensibilidade_alvo) * len(pos))]) if len(pos) else 0.5
    clf.fit(dados[colunas], dados["y"])
    return ModeloHipotensao(clf, colunas, limiar, {"n": int(len(dados)), "positivos": int(dados["y"].sum())})


class PreditorHipotensao:
    """Versão streaming: mantém 10 min de histórico e avalia a cada 30 s."""

    def __init__(self, paciente_id: str, modelo: ModeloHipotensao, consecutivas: int = 2,
                 refratario_s: float = 900) -> None:
        self.paciente_id, self.modelo = paciente_id, modelo
        self.consecutivas, self.refratario_s = consecutivas, refratario_s
        self._buf: deque = deque()
        self._inicio: float | None = None
        self._proximo: float | None = None
        self._seguidas = 0
        self._ultimo_alerta = -1e18
        self.probs: list[tuple[float, float]] = []

    def atualizar(self, t: float, amostra: dict) -> list[Evento]:
        self._buf.append({"t_s": t, **{c: amostra.get(c, np.nan) for c in ("HR", "SBP", "DBP", "MAP", "SPO2")}})
        while self._buf and self._buf[0]["t_s"] < t - HISTORICO_S - 60:
            self._buf.popleft()
        if self._inicio is None:
            self._inicio = t
            self._proximo = t + HISTORICO_S
        if t < self._proximo:
            return []
        self._proximo = t + PASSO_S
        f = features_preditivas(pd.DataFrame(list(self._buf)), t)
        if np.isnan(f["MAP_atual"]) or f["MAP_atual"] < LIMIAR_PAM:
            self._seguidas = 0
            return []
        p = float(self.modelo.prob(pd.DataFrame([f]))[0])
        self.probs.append((t, p))
        self._seguidas = self._seguidas + 1 if p >= self.modelo.limiar else 0
        if self._seguidas >= self.consecutivas and t - self._ultimo_alerta >= self.refratario_s:
            self._ultimo_alerta = t
            return [Evento(
                paciente_id=self.paciente_id, origem="vitais", tipo="risco_hipotensao_10min",
                severidade=Severidade.ATENCAO, t=t, detector="ml",
                descricao=(f"Risco de hipotensão nos próximos 10 min: p={p:.0%} "
                           f"(PAM atual {f['MAP_atual']:.0f} mmHg, tendência {f['MAP_inclinacao_5min']:+.1f} mmHg/min)"),
                evidencias={"probabilidade": round(p, 3), "limiar": round(self.modelo.limiar, 3),
                            "MAP_atual": round(f["MAP_atual"], 1),
                            "MAP_inclinacao_5min": round(f["MAP_inclinacao_5min"], 2)},
            )]
        return []
