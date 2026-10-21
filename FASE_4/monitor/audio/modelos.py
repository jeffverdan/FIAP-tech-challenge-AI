"""Classificadores de voz: disartria (TORGO), dificuldade respiratória e fadiga (Coswara).

Todos seguem o mesmo protocolo para não "vazar" a identidade do falante para o teste:
validação cruzada **agrupada por falante** (``GroupKFold``) — o modelo é sempre avaliado em
vozes que nunca ouviu. Sem isso, um classificador de disartria aprende a reconhecer as 15
pessoas do TORGO, não a disartria (o erro mais comum em trabalhos com esse dataset).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


def _pipeline(tipo: str, random_state: int = 42, C: float = 0.3):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if tipo == "rf":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                                    random_state=random_state, n_jobs=1))
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=5000))


@dataclass
class ModeloVoz:
    nome: str
    colunas: list[str]
    pipeline: object
    limiar: float
    metricas: dict = field(default_factory=dict)
    media_negativos: dict = field(default_factory=dict)
    #: domínio de gravação → deslocamento por feature (ver :meth:`calibrar_dominio`)
    deslocamentos: dict = field(default_factory=dict)

    @property
    def usa_wav2vec(self) -> bool:
        return any(c.startswith("w2v_") for c in self.colunas)

    def calibrar_dominio(self, dominio: str, X_saudaveis: pd.DataFrame) -> None:
        """Alinhamento de média: desloca as features de um domínio de gravação novo (outro
        microfone, outra sala, outro idioma) para que **falantes saudáveis desse domínio**
        fiquem, em média, onde estavam os controles do treino. Requer só gravações de pessoas
        sem a condição — o que um hospital consegue coletar (ex.: primeira consulta, equipe)."""
        ref = X_saudaveis.reindex(columns=self.colunas).mean()
        base = pd.Series(self.media_negativos).reindex(self.colunas)
        self.deslocamentos[dominio] = (base - ref).fillna(0.0).to_dict()

    def prob(self, feats: pd.DataFrame | dict, dominio: str | None = None) -> np.ndarray:
        X = pd.DataFrame([feats]) if isinstance(feats, dict) else feats
        X = X.reindex(columns=self.colunas)
        if dominio and dominio in self.deslocamentos:
            X = X + pd.Series(self.deslocamentos[dominio])
        return self.pipeline.predict_proba(X)[:, 1]

    def salvar(self, caminho: Path) -> None:
        import joblib

        Path(caminho).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, caminho)

    @staticmethod
    def carregar(caminho: Path) -> "ModeloVoz":
        import joblib

        return joblib.load(caminho)


def avaliar_e_treinar(nome: str, X: pd.DataFrame, y: np.ndarray, grupos: np.ndarray, n_splits: int = 5,
                      especificidade_alvo: float = 0.8, tipos: tuple[str, ...] = ("logistica", "rf"),
                      C: float = 0.3) -> tuple[ModeloVoz, dict, np.ndarray]:
    """Compara regressão logística × random forest com CV agrupada e treina o melhor em tudo.

    O limiar de alerta é escolhido nas predições fora-da-amostra para ~80% de especificidade
    (num monitor, falso alarme custa mais que no diagnóstico).
    """
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    n_splits = min(n_splits, len(np.unique(grupos)))
    cv = GroupKFold(n_splits=n_splits)
    resultados, oofs = {}, {}
    for tipo in tipos:
        oof = cross_val_predict(_pipeline(tipo, C=C), X, y, groups=grupos, cv=cv, method="predict_proba")[:, 1]
        oofs[tipo] = oof
        resultados[tipo] = {"auroc": round(float(roc_auc_score(y, oof)), 3),
                            "auprc": round(float(average_precision_score(y, oof)), 3)}
    melhor = max(resultados, key=lambda k: resultados[k]["auroc"])
    oof = oofs[melhor]
    neg = np.sort(oof[y == 0])
    limiar = float(neg[min(int(especificidade_alvo * len(neg)), len(neg) - 1)])
    pred = oof > limiar
    m = {
        "modelo_escolhido": melhor, "comparacao": resultados, "n": int(len(y)), "positivos": int(y.sum()),
        "grupos": int(len(np.unique(grupos))), "folds": n_splits, "limiar": round(limiar, 3),
        "sensibilidade": round(float(pred[y == 1].mean()), 3), "especificidade": round(float((~pred[y == 0]).mean()), 3),
        "acuracia_balanceada": round(float(balanced_accuracy_score(y, pred)), 3),
        **resultados[melhor],
    }
    pipe = _pipeline(melhor, C=C).fit(X, y)
    media_neg = X[y == 0].mean().to_dict()
    return ModeloVoz(nome, list(X.columns), pipe, limiar, m, media_neg), m, oof
