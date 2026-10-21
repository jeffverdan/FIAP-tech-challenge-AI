"""Experimento de detecção de anomalias em sinais vitais (VitalDB + MIMIC-IV demo).

    python scripts/run_vitals.py                  # tudo (~10 min em CPU)
    python scripts/run_vitals.py --rapido         # 8 casos de teste, sem injeção

Saídas em ``results/vitals/``:
  * ``metricas.json``               — todas as métricas citadas no relatório
  * ``eventos_caso_XXXX.json``      — eventos por caso de teste (o dashboard usa)
  * ``timeline_caso_XXXX.png``      — sinais vitais com os alertas sobrepostos
  * ``roc_hipotensao.png``          — curva ROC do preditor de hipotensão
  * ``injecao.csv``                 — resultado por anomalia sintética injetada
e os modelos em ``data/processed/models/``.

Protocolo
---------
* 100 casos VitalDB (anestesia geral, ≥ 2 h, PA invasiva). **Treino: 70 primeiros. Teste: 30
  últimos**, nunca vistos por nenhum modelo.
* **Preditor de hipotensão (supervisionado):** rótulo automático "PAM < 65 sustentada começa nos
  próximos 10 min"; AUROC/AUPRC no teste, comparado com o *baseline* "só a PAM atual".
* **Isolation Forest (não supervisionado):** treinado só em janelas sem violação de regra.
* **Antecipação em dados reais:** para cada episódio-índice confirmado por regra, verifica se
  alguma camada precoce alertou nos 15 min anteriores; conta também alertas "sem desfecho".
* **Injeção sintética:** anomalias com ground truth conhecido em trechos estáveis do teste;
  taxa de detecção e atraso por camada; trechos-controle medem falso alarme.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitor.config import CONFIG  # noqa: E402
from monitor.events import salvar_eventos  # noqa: E402
from monitor.vitals.canais import carregar_mimic_vitais, carregar_vitaldb, limpar  # noqa: E402
from monitor.vitals.detectores import (  # noqa: E402
    PERFIL_CONTINUO, DetectorNEWS2, ModeloIForest, MonitorSinaisVitais, janelas)
from monitor.vitals import predicao  # noqa: E402

RAW = CONFIG.raw_dir
OUT = CONFIG.results_dir / "vitals"
MODELOS = CONFIG.processed_dir / "models"
RNG = np.random.default_rng(42)

CASOS = json.loads((RAW / "vitaldb" / "casos.json").read_text())
CASOS_TREINO, CASOS_TESTE = CASOS[:70], CASOS[70:]
CASO_DEMO = 49   # caso com dessaturação + hipotensão, usado nas figuras e no dashboard


def camada(e) -> str:
    if e.detector == "ml":
        return "preditor" if e.tipo == "risco_hipotensao_10min" else "iforest"
    return {"estatistico": "zscore"}.get(e.detector, e.detector)


CAMADAS = ["regra", "zscore", "iforest", "preditor"]


def carregar_caso(caso: int) -> pd.DataFrame:
    df, _ = limpar(carregar_vitaldb(RAW / "vitaldb" / f"case_{caso:04d}.csv.gz"))
    ok = df[["HR", "MAP"]].notna().all(axis=1)
    if ok.any():   # descarta antes da entrada / após a saída da sala
        df = df.loc[ok.idxmax(): ok[::-1].idxmax()].reset_index(drop=True)
        df["t_s"] -= df["t_s"].iloc[0]
    return df


# --------------------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------------------
def janelas_normais(df: pd.DataFrame) -> pd.DataFrame:
    w = janelas(df)
    viol = np.zeros(len(w), dtype=bool)
    for r in PERFIL_CONTINUO:
        col = f"{r.canal}_media"
        if col in w:
            viol |= w[col].apply(lambda v: (not np.isnan(v)) and r.viola(v)).to_numpy()
    return w[~viol]


def treinar_iforest() -> ModeloIForest:
    X = pd.concat([janelas_normais(carregar_caso(c)) for c in CASOS_TREINO], ignore_index=True)
    modelo = ModeloIForest(quantil_alarme=0.01).fit(X)
    modelo.salvar(MODELOS / "iforest_vitais.joblib")
    print(f"  Isolation Forest: {modelo.n_treino} janelas normais de {len(CASOS_TREINO)} casos")
    return modelo


def dataset_hipotensao() -> pd.DataFrame:
    cache = CONFIG.processed_dir / "vitals_hipotensao_dataset.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    partes = [predicao.montar_dataset(carregar_caso(c), c) for c in CASOS]
    dados = pd.concat(partes, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    dados.to_csv(cache, index=False, compression="gzip")
    return dados


def avaliar_preditor(dados: pd.DataFrame) -> tuple[predicao.ModeloHipotensao, dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    tr = dados[dados["caso"].isin(CASOS_TREINO)]
    te = dados[dados["caso"].isin(CASOS_TESTE)]
    modelo = predicao.treinar(tr)
    modelo.salvar(MODELOS / "preditor_hipotensao.joblib")
    p = modelo.prob(te)
    y = te["y"].to_numpy()
    base = -te["MAP_atual"].to_numpy()             # baseline: quanto menor a PAM, maior o risco
    pred = p >= modelo.limiar
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    m = {
        "amostras_treino": int(len(tr)), "positivos_treino": int(tr["y"].sum()),
        "amostras_teste": int(len(te)), "positivos_teste": int(y.sum()),
        "prevalencia_teste": round(float(y.mean()), 4),
        "auroc": round(float(roc_auc_score(y, p)), 3),
        "auprc": round(float(average_precision_score(y, p)), 3),
        "auroc_baseline_pam_atual": round(float(roc_auc_score(y, base)), 3),
        "auprc_baseline_pam_atual": round(float(average_precision_score(y, base)), 3),
        "limiar": round(modelo.limiar, 3),
        "sensibilidade": round(tp / max(tp + fn, 1), 3), "especificidade": round(tn / max(tn + fp, 1), 3),
        "vpp": round(tp / max(tp + fp, 1), 3),
    }
    # importância por permutação (top 8) — explicabilidade do modelo
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(modelo.modelo, te[modelo.colunas], y, scoring="roc_auc", n_repeats=5,
                                 random_state=42)
    ordem = np.argsort(-imp.importances_mean)[:8]
    m["importancia_permutacao"] = {modelo.colunas[i]: round(float(imp.importances_mean[i]), 4) for i in ordem}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    for nome, s, cor in (("Gradient Boosting", p, "#2c3e50"), ("só PAM atual", base, "#95a5a6")):
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, color=cor, label=f"{nome} (AUROC {roc_auc_score(y, s):.2f})")
    ax.plot([0, 1], [0, 1], ls=":", color="gray")
    ax.scatter([1 - m["especificidade"]], [m["sensibilidade"]], color="#c0392b", zorder=5, label="limiar de alerta")
    ax.set_xlabel("1 − especificidade")
    ax.set_ylabel("sensibilidade")
    ax.set_title("Hipotensão nos próximos 10 min\n(30 casos de teste)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "roc_hipotensao.png", dpi=110)
    plt.close(fig)
    return modelo, m


# --------------------------------------------------------------------------------------
# Antecipação em dados reais
# --------------------------------------------------------------------------------------
def episodios_indice(episodios: list[dict]) -> list[dict]:
    """Episódios urgentes/críticos sem outro episódio relevante nos 30 min anteriores."""
    graves = sorted([e for e in episodios if e["severidade"] >= 2], key=lambda e: e["inicio"])
    out, ultimo_fim = [], -1e18
    for ep in graves:
        if ep["inicio"] > ultimo_fim + 1800:
            out.append(ep)
        ultimo_fim = max(ultimo_fim, ep["fim"])
    return out


def avaliar_antecipacao(eventos, episodios, horas: float, horizonte_s: float = 900) -> dict:
    res = {"horas": horas, "episodios_indice": 0, "episodios_hipotensao": 0}
    indice = episodios_indice(episodios)
    res["episodios_indice"] = len(indice)
    for cam in ("zscore", "iforest", "preditor", "qualquer"):
        res[f"{cam}_antecipados"] = 0
        res[f"{cam}_antecedencias_s"] = []
        precoces = [e for e in eventos if camada(e) != "regra" and (cam == "qualquer" or camada(e) == cam)]
        res[f"{cam}_alertas"] = len(precoces)
        alvo = [ep for ep in indice if cam != "preditor" or ep["tipo"].startswith("hipotensao")]
        for ep in alvo:
            janela = [e for e in precoces if ep["alerta"] - horizonte_s <= e.t < ep["alerta"]]
            if janela:
                res[f"{cam}_antecipados"] += 1
                res[f"{cam}_antecedencias_s"].append(ep["alerta"] - min(e.t for e in janela))
        sem = 0
        for e in precoces:
            explicado = any(ep["inicio"] - horizonte_s <= e.t <= ep["fim"] for ep in episodios if ep["severidade"] >= 2)
            sem += 0 if explicado else 1
        res[f"{cam}_sem_desfecho"] = sem
    res["episodios_hipotensao"] = sum(ep["tipo"].startswith("hipotensao") for ep in indice)
    return res


# --------------------------------------------------------------------------------------
# Injeção sintética
# --------------------------------------------------------------------------------------
def _rampa(n: int, n_rampa: int) -> np.ndarray:
    return np.clip(np.arange(n) / max(n_rampa, 1), 0, 1)


def injetar(seg: pd.DataFrame, i0: int, tipo: str) -> pd.DataFrame:
    s = seg.copy()
    n = len(s) - i0
    dt = float(np.median(np.diff(s["t_s"])))
    idx = s.index[i0:]
    if tipo == "dessaturacao_progressiva":         # −12 pontos de SpO2 em 5 min
        s.loc[idx, "SPO2"] = s.loc[idx, "SPO2"] - 12 * _rampa(n, int(300 / dt))
    elif tipo == "taquicardia_subita":             # +35 bpm em 20 s
        s.loc[idx, "HR"] = s.loc[idx, "HR"] + 35 * _rampa(n, int(20 / dt))
    elif tipo == "hipotensao_progressiva":         # −35% da PA em 8 min
        f = 1 - 0.35 * _rampa(n, int(480 / dt))
        for c in ("SBP", "DBP", "MAP"):
            s.loc[idx, c] = s.loc[idx, c] * f
    elif tipo == "choque_compensado":              # FC +25 e PAS −18% em 5 min, sem cruzar limiar isolado
        r = _rampa(n, int(300 / dt))
        s.loc[idx, "HR"] = s.loc[idx, "HR"] + 25 * r
        s.loc[idx, "SBP"] = s.loc[idx, "SBP"] * (1 - 0.18 * r)
        s.loc[idx, "MAP"] = s.loc[idx, "MAP"] * (1 - 0.10 * r)
    elif tipo != "controle":
        raise ValueError(tipo)
    return s


TIPOS_INJECAO = ["dessaturacao_progressiva", "taquicardia_subita", "hipotensao_progressiva",
                 "choque_compensado", "controle"]


def experimento_injecao(casos, iforest, preditor, por_caso: int = 5) -> pd.DataFrame:
    linhas = []
    aquec_s, obs_s = 900, 600
    for caso in casos:
        df = carregar_caso(caso)
        ref = MonitorSinaisVitais("ref")
        ref.processar_df(df)
        ocupado = [(ep["inicio"] - 1200, ep["fim"] + 600) for ep in ref.regras.episodios]
        candidatos = [t for t in np.arange(aquec_s + 300, df["t_s"].iloc[-1] - obs_s, 60)
                      if not any(a <= t <= b for a, b in ocupado)]
        if not candidatos:
            continue
        escolhidos = sorted(RNG.choice(candidatos, size=min(por_caso, len(candidatos)), replace=False))
        for k, t0 in enumerate(escolhidos):
            tipo = TIPOS_INJECAO[(k + caso) % len(TIPOS_INJECAO)]
            seg = df[(df["t_s"] >= t0 - aquec_s) & (df["t_s"] < t0 + obs_s)].reset_index(drop=True)
            seg = injetar(seg, int(np.searchsorted(seg["t_s"], t0)), tipo)
            mon = MonitorSinaisVitais(f"INJ-{caso}", iforest, modelo_hipotensao=preditor)
            evs = [e for e in mon.processar_df(seg) if e.t >= t0]
            linha = {"caso": caso, "t0": t0, "tipo": tipo}
            for cam in CAMADAS + ["qualquer"]:
                ts = [e.t - t0 for e in evs if cam == "qualquer" or camada(e) == cam]
                linha[f"{cam}_detectou"] = bool(ts)
                linha[f"{cam}_atraso_s"] = min(ts) if ts else np.nan
            linhas.append(linha)
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------------------
def figura_caso(caso: int, df: pd.DataFrame, eventos, destino: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cores = {"regra": "#c0392b", "zscore": "#e67e22", "iforest": "#8e44ad", "preditor": "#16a085"}
    rotulos = {"regra": "regra clínica", "zscore": "z-score robusto", "iforest": "Isolation Forest",
               "preditor": "preditor de hipotensão"}
    fig, eixos = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    t_min = df["t_s"] / 60
    for ax, (c, nome) in zip(eixos, [("HR", "FC (bpm)"), ("MAP", "PAM (mmHg)"), ("SPO2", "SpO₂ (%)"),
                                       ("RR", "FR (irpm)")]):
        ax.plot(t_min, df[c], lw=0.7, color="#34495e")
        ax.set_ylabel(nome)
        ax.grid(alpha=0.3)
    eixos[1].axhline(65, ls="--", color="#c0392b", lw=0.8)
    eixos[2].axhline(92, ls="--", color="#c0392b", lw=0.8)
    for e in eventos:
        for ax in eixos:
            ax.axvline(e.t / 60, color=cores[camada(e)], alpha=0.4, lw=1)
    eixos[0].legend([Line2D([0], [0], color=v) for v in cores.values()], list(rotulos.values()),
                    loc="upper right", fontsize=8, ncol=2)
    eixos[-1].set_xlabel("minutos")
    fig.suptitle(f"VitalDB caso {caso} — sinais vitais e alertas por camada")
    fig.tight_layout()
    fig.savefig(destino, dpi=110)
    plt.close(fig)


def rodar_mimic() -> dict:
    """NEWS2 por internação de UTI + validação contra mortalidade hospitalar (AUROC)."""
    from sklearn.metrics import roc_auc_score

    base = RAW / "mimic-iv-demo"
    vit = carregar_mimic_vitais(base / "icu" / "chartevents.csv.gz")
    stays = pd.read_csv(base / "icu" / "icustays.csv.gz")
    adm = pd.read_csv(base / "hosp" / "admissions.csv.gz")
    stays = stays.merge(adm[["hadm_id", "hospital_expire_flag"]], on="hadm_id", how="left")
    linhas, todos = [], []
    for stay_id, g in vit.groupby("stay_id"):
        det = DetectorNEWS2(f"MIMIC-{stay_id}")
        evs = []
        for r in g.itertuples(index=False):
            evs += det.atualizar(float(r.t_s), r._asdict())
        todos += evs
        s24 = [s for t, s in det.serie if t <= 24 * 3600]
        linhas.append({"stay_id": stay_id, "news2_max_24h": max(s24) if s24 else np.nan,
                       "alertas": len(evs), "criticos": sum(int(e.severidade) == 3 for e in evs)})
    res = pd.DataFrame(linhas).merge(stays[["stay_id", "hospital_expire_flag", "los"]], on="stay_id")
    ok = res.dropna(subset=["news2_max_24h", "hospital_expire_flag"])
    auc = roc_auc_score(ok["hospital_expire_flag"], ok["news2_max_24h"])
    res.to_csv(OUT / "mimic_news2_por_internacao.csv", index=False)
    salvar_eventos(todos, OUT / "eventos_mimic_news2.json")
    dias = float(res["los"].sum())
    return {"internacoes": int(len(res)), "pacientes_dia": round(dias, 1), "obitos": int(ok["hospital_expire_flag"].sum()),
            "auroc_news2_max_24h_mortalidade": round(float(auc), 3),
            "alertas_total": int(res["alertas"].sum()), "alertas_por_paciente_dia": round(res["alertas"].sum() / dias, 2),
            "alertas_criticos": int(res["criticos"].sum()),
            "news2_max_24h_mediana_obitos": float(ok.loc[ok.hospital_expire_flag == 1, "news2_max_24h"].median()),
            "news2_max_24h_mediana_sobreviventes": float(ok.loc[ok.hospital_expire_flag == 0, "news2_max_24h"].median())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rapido", action="store_true", help="8 casos de teste, sem injeção")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    teste = CASOS_TESTE[:8] if a.rapido else CASOS_TESTE
    metricas: dict = {"n_casos_treino": len(CASOS_TREINO), "n_casos_teste": len(CASOS_TESTE),
                      "casos_teste": CASOS_TESTE}

    print("[1/5] preditor de hipotensão")
    dados = dataset_hipotensao()
    preditor, metricas["preditor_hipotensao"] = avaliar_preditor(dados)
    print("  ", {k: v for k, v in metricas["preditor_hipotensao"].items() if k != "importancia_permutacao"})

    print("[2/5] Isolation Forest")
    iforest = treinar_iforest()

    print(f"[3/5] monitor completo em {len(teste)} casos de teste")
    agreg: dict = {}
    contagem: dict = {}
    for caso in teste + ([CASO_DEMO] if CASO_DEMO not in teste else []):
        df = carregar_caso(caso)
        mon = MonitorSinaisVitais(f"VDB-{caso:04d}", iforest, modelo_hipotensao=preditor)
        eventos = mon.processar_df(df)
        salvar_eventos(eventos, OUT / f"eventos_caso_{caso:04d}.json")
        if caso == CASO_DEMO:
            figura_caso(caso, df, eventos, OUT / f"timeline_caso_{caso:04d}.png")
        if caso not in teste:
            continue
        av = avaliar_antecipacao(eventos, mon.regras.episodios, float(df["t_s"].iloc[-1]) / 3600)
        for k, v in av.items():
            agreg[k] = agreg.get(k, [] if isinstance(v, list) else 0) + v
        for e in eventos:
            contagem[camada(e)] = contagem.get(camada(e), 0) + 1
    resumo = {"horas": round(agreg["horas"], 1), "episodios_indice": agreg["episodios_indice"],
              "episodios_indice_hipotensao": agreg["episodios_hipotensao"], "alertas_por_camada": contagem}
    for cam in ("zscore", "iforest", "preditor", "qualquer"):
        denom = agreg["episodios_hipotensao"] if cam == "preditor" else agreg["episodios_indice"]
        ant = agreg[f"{cam}_antecedencias_s"]
        resumo[cam] = {"antecipados": agreg[f"{cam}_antecipados"], "de": denom,
                       "taxa": round(agreg[f"{cam}_antecipados"] / denom, 3) if denom else None,
                       "antecedencia_mediana_min": round(float(np.median(ant)) / 60, 1) if ant else None,
                       "alertas_por_hora": round(agreg[f"{cam}_alertas"] / agreg["horas"], 2),
                       "sem_desfecho_por_hora": round(agreg[f"{cam}_sem_desfecho"] / agreg["horas"], 2)}
    metricas["antecipacao_dados_reais"] = resumo
    print("  ", json.dumps(resumo, ensure_ascii=False))

    if not a.rapido:
        print("[4/5] injeção sintética")
        inj = experimento_injecao(teste, iforest, preditor)
        inj.to_csv(OUT / "injecao.csv", index=False)
        tab = {}
        for tipo, g in inj.groupby("tipo"):
            tab[tipo] = {"n": int(len(g))}
            for cam in CAMADAS + ["qualquer"]:
                tab[tipo][f"{cam}_taxa"] = round(float(g[f"{cam}_detectou"].mean()), 2)
                at = g[f"{cam}_atraso_s"].median()
                tab[tipo][f"{cam}_atraso_mediano_s"] = None if np.isnan(at) else round(float(at))
        metricas["injecao"] = tab
        print(pd.DataFrame(tab).T[["n"] + [f"{c}_taxa" for c in CAMADAS + ["qualquer"]]].to_string())

    print("[5/5] MIMIC-IV demo — NEWS2")
    metricas["mimic_news2"] = rodar_mimic()
    print("  ", metricas["mimic_news2"])

    (OUT / "metricas.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False, default=str),
                                      encoding="utf-8")
    print("ok →", OUT)


if __name__ == "__main__":
    main()
