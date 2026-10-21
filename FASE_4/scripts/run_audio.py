"""Experimento de análise de áudio.

    python scripts/run_audio.py                 # tudo
    python scripts/run_audio.py --so-consultas  # só reprocessa as consultas (ex.: depois de configurar o Azure)

Etapas
------
1. **TORGO → classificador de disartria** (validação cruzada agrupada por falante).
2. **Coswara → classificadores de dificuldade respiratória e de fadiga** a partir da contagem
   de 1 a 20 (sintomas autorrelatados; controles = saudáveis sem nenhum sintoma).
3. **PriMock57 → pipeline completo de consulta** (Azure Speech → Azure Language → acústica →
   modelos de voz → eventos). Se o Azure estiver ativo, mede também o WER contra a transcrição
   manual do dataset.
4. **Demos de voz alterada:** concatena gravações de um falante com disartria (TORGO) e de um
   participante com dispneia (Coswara) e passa pelo mesmo pipeline — é o que aparece no vídeo.

Saídas: ``results/audio/metricas.json``, ``results/audio/consultas/*.json``,
``results/audio/consultas_resumo.csv`` e modelos em ``data/processed/models/voz_*.joblib``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitor.audio import acustica  # noqa: E402
from monitor.audio.consulta import ModelosVoz, analisar_consulta  # noqa: E402
from monitor.audio.modelos import avaliar_e_treinar  # noqa: E402
from monitor.azure.speech import ler_textgrid  # noqa: E402
from monitor.config import CONFIG  # noqa: E402
from monitor.events import salvar_eventos  # noqa: E402

RAW = CONFIG.raw_dir
PROC = CONFIG.processed_dir
OUT = CONFIG.results_dir / "audio"
MODELOS = PROC / "models"


def _feats_arquivos(arquivos: list[Path], cache: Path, chave) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache)
    linhas = []
    for i, arq in enumerate(arquivos, 1):
        try:
            f = acustica.features_voz(acustica.carregar(arq))
        except Exception as e:  # noqa: BLE001 — arquivo corrompido não derruba o experimento
            print(f"  ! {arq.name}: {e}")
            continue
        linhas.append({**chave(arq), **f})
        if i % 50 == 0:
            print(f"  {i}/{len(arquivos)}")
    df = pd.DataFrame(linhas)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def _w2v_arquivos(arquivos: list[Path], cache: Path, chave, max_s: float = 30.0) -> pd.DataFrame | None:
    """Embeddings wav2vec 2.0 (768-d) por arquivo, com cache em CSV. ``None`` sem torchaudio."""
    from monitor.audio import embeddings

    if cache.exists():
        return pd.read_csv(cache)
    if not embeddings.disponivel():
        print("  (torchaudio ausente — pulando wav2vec 2.0)")
        return None
    linhas = []
    for i, arq in enumerate(arquivos, 1):
        try:
            e = embeddings.embedding(acustica.carregar(arq), max_s=max_s)
        except Exception as ex:  # noqa: BLE001
            print(f"  ! {arq.name}: {ex}")
            continue
        linhas.append({**chave(arq), **{f"w2v_{j}": round(float(v), 5) for j, v in enumerate(e)}})
        if i % 100 == 0:
            print(f"  wav2vec {i}/{len(arquivos)}")
    df = pd.DataFrame(linhas)
    df.to_csv(cache, index=False)
    return df


def treinar_variantes(nome: str, df: pd.DataFrame, y: np.ndarray, grupos: np.ndarray) -> tuple[dict, np.ndarray]:
    """Compara marcadores clássicos × wav2vec 2.0 × ambos; salva o melhor como ``voz_<nome>`` e o de
    marcadores como ``voz_<nome>_marcadores`` (fallback sem torchaudio)."""
    marc = colunas_modelo(df)
    w2v = [c for c in df.columns if c.startswith("w2v_")]
    variantes = {"marcadores": (marc, ("logistica", "rf"), 0.3)}
    if w2v:
        variantes["wav2vec2"] = (w2v, ("logistica",), 0.01)
        variantes["marcadores+wav2vec2"] = (marc + w2v, ("logistica",), 0.01)
    res, melhores = {}, {}
    for var, (cols, tipos, C) in variantes.items():
        modelo, m, oof = avaliar_e_treinar(nome, df[cols], y, grupos, tipos=tipos, C=C)
        res[var], melhores[var] = m, (modelo, oof)
    melhores["marcadores"][0].salvar(MODELOS / f"voz_{nome}_marcadores.joblib")
    escolhida = max(res, key=lambda k: res[k]["auroc"])
    melhores[escolhida][0].salvar(MODELOS / f"voz_{nome}.joblib")
    return {"variante_escolhida": escolhida, **res[escolhida],
            "variantes": {k: {kk: v[kk] for kk in ("auroc", "auprc", "sensibilidade", "especificidade", "modelo_escolhido")}
                          for k, v in res.items()}}, melhores[escolhida][1]


def colunas_modelo(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c in acustica.MARCADORES or c.startswith("mfcc") or c == "centroide_espectral_hz"]


# --------------------------------------------------------------------------------------
def etapa_torgo() -> dict:
    arquivos = sorted((RAW / "torgo").glob("*/*.wav"))
    df = _feats_arquivos(arquivos, PROC / "audio_torgo_features.csv",
                         lambda a: {"arquivo": a.name, "falante": a.name.split("_")[0],
                                    "disartria": int(not a.parent.name.startswith("non_"))})
    w2v = _w2v_arquivos(arquivos, PROC / "audio_torgo_w2v.csv", lambda a: {"arquivo": a.name})
    if w2v is not None:
        df = df.merge(w2v, on="arquivo", how="inner")
    df = df[df["duracao_s"] >= 0.5].reset_index(drop=True)
    m, oof = treinar_variantes("disartria", df, df["disartria"].to_numpy(), df["falante"].to_numpy())
    limiar = m["limiar"]
    # nível de falante: média das probabilidades fora-da-amostra das gravações de cada pessoa
    df["oof"] = oof
    por_falante = df.groupby("falante").agg(y=("disartria", "first"), p=("oof", "mean"), n=("oof", "size"))
    por_falante["pred"] = por_falante["p"] > limiar
    from sklearn.metrics import roc_auc_score
    m["nivel_falante"] = {"auroc": round(float(roc_auc_score(por_falante["y"], por_falante["p"])), 3),
                          "acertos": int((por_falante["pred"] == por_falante["y"].astype(bool)).sum()),
                          "falantes": int(len(por_falante)),
                          "detalhe": por_falante.round(3).reset_index().to_dict(orient="records")}
    # marcadores interpretáveis: mediana por grupo (para o relatório)
    m["marcadores_mediana"] = df.groupby("disartria")[[c for c in acustica.MARCADORES if c in df]].median().round(3) \
        .rename(index={0: "controle", 1: "disartria"}).to_dict(orient="index")
    return m


def etapa_coswara() -> dict:
    """Dificuldade respiratória (bd) e fadiga (ftg) a partir da contagem de 1 a 20.

    **Armadilha de confusão por lote.** Os participantes com sintomas vêm dos lotes de 2021 e os
    saudáveis, majoritariamente, de abril/2020 — e o *lote* é previsível pelo áudio (aparelho,
    versão do app). Comparar "sintomático 2021 × saudável 2020" mede o lote, não o sintoma.
    Por isso o protocolo oficial usa **só os lotes de 2021** e, como controle, **todos os
    participantes desses lotes que não relataram o sintoma** (inclui quem tem outros sintomas —
    é o cenário real de um hospital). O protocolo ingênuo fica registrado para comparação.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    from monitor.audio.modelos import _pipeline

    meta = pd.read_csv(RAW / "coswara" / "combined_data.csv")
    sintomas = ["bd", "ftg", "cough", "fever", "st", "cold", "loss_of_smell", "diarrhoea", "mp", "others_resp"]
    for c in sintomas:
        meta[c] = meta[c].fillna(False).astype(str).str.lower().isin({"true", "1", "yes"}) if c in meta else False
    arquivos = sorted((RAW / "coswara").glob("2*/*/counting-normal.wav"))
    lote = {a.parent.name: a.parent.parent.name for a in arquivos}
    df = _feats_arquivos(arquivos, PROC / "audio_coswara_features.csv", lambda a: {"id": a.parent.name})
    w2v = _w2v_arquivos([a for a in arquivos if lote[a.parent.name] >= "2021"], PROC / "audio_coswara_w2v.csv",
                        lambda a: {"id": a.parent.name}, max_s=20)
    df["lote"] = df["id"].map(lote)
    df = df.merge(meta, on="id", how="left")
    df = df[df["duracao_s"] >= 3].reset_index(drop=True)
    saudavel = (df["covid_status"] == "healthy") & ~df[sintomas].any(axis=1)
    cols = colunas_modelo(df)

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    y_lote = (df["lote"] < "2021").astype(int)
    p_lote = cross_val_predict(_pipeline("rf"), df[cols], y_lote, cv=cv, method="predict_proba")[:, 1]
    res = {"confusao_por_lote": {"auroc_prever_lote_2020_vs_2021_pelo_audio": round(float(roc_auc_score(y_lote, p_lote)), 3),
                                 "participantes": int(len(df))}}
    d21 = df[df["lote"] >= "2021"].reset_index(drop=True)
    if w2v is not None:
        d21 = d21.merge(w2v, on="id", how="inner")
    for alvo, nome in (("bd", "respiratorio"), ("ftg", "fadiga")):
        ingenuo = df[df[alvo] | saudavel]
        y_i = ingenuo[alvo].astype(int)
        p_i = cross_val_predict(_pipeline("rf"), ingenuo[cols], y_i, cv=cv, method="predict_proba")[:, 1]
        y = d21[alvo].astype(int).to_numpy()
        m, _ = treinar_variantes(nome, d21, y, d21["id"].to_numpy())
        m["protocolo"] = "lotes 2021; controle = participantes sem o sintoma"
        m["protocolo_ingenuo_auroc"] = round(float(roc_auc_score(y_i, p_i)), 3)
        m["marcadores_mediana"] = d21.groupby(alvo)[[c for c in acustica.MARCADORES if c in d21]].median().round(3) \
            .rename(index={False: "sem_sintoma", True: nome}).to_dict(orient="index")
        res[nome] = m
    # o modelo de disartria (treinado no TORGO) também precisa de calibração para o domínio Coswara
    mods = ModelosVoz.carregar()
    if mods.disartria is not None:
        saud21 = d21[~d21[sintomas].any(axis=1) & (d21["covid_status"] == "healthy")].set_index("id")
        res["disartria_no_dominio_coswara"] = calibrar_dominio_loo(ModelosVoz(disartria=mods.disartria),
                                                                  saud21, "coswara")["disartria"]
    return res


# --------------------------------------------------------------------------------------
def _normalizar(txt: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", txt.lower()).split()


def wer(ref: str, hip: str) -> float:
    r, h = _normalizar(ref), _normalizar(hip)
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        ant, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(ant[j] + 1, d[j - 1] + 1, ant[j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(h)] / max(len(r), 1))


#: Mapeamento manual queixa principal (nota do médico) → sinal de alarme esperado.
ESPERADO = {
    "day1_consultation06": {"dispneia"}, "day2_consultation04": {"dispneia"}, "day2_consultation06": {"dispneia"},
    "day2_consultation07": {"dor_toracica"}, "day2_consultation09": {"deficit_neurologico"},
    "day3_consultation03": {"fadiga"},
    # ERRO NO METADADO DO DATASET: o campo presenting_complaint diz "my chest hurts and it's been hard
    # to breathe", mas o áudio, a transcrição manual e a nota do próprio médico tratam de cefaleia
    # com alteração visual. O esperado aqui é "nenhum sinal de alarme torácico/respiratório".
    "day4_consultation06": set(),
    "day5_consultation09": {"fadiga"}, "day1_consultation02": set(), "day1_consultation09": set(),
    "day5_consultation03": set(),
}


def _salvar_modelos(modelos: ModelosVoz) -> None:
    from monitor.audio import embeddings

    for nome in ("disartria", "respiratorio", "fadiga"):
        m = getattr(modelos, nome)
        if m is not None:
            m.salvar(MODELOS / (f"voz_{nome}.joblib" if m.usa_wav2vec or not embeddings.disponivel()
                                else f"voz_{nome}_marcadores.joblib"))


def calibrar_dominio_loo(modelos: ModelosVoz, X: pd.DataFrame, dominio: str) -> dict:
    """Falsos positivos dos modelos de voz em falantes saudáveis de um domínio novo, antes e
    depois do alinhamento de média (calibração deixa-um-fora: o falante avaliado nunca entra
    na própria referência). Ao final, grava a calibração com todos os falantes."""
    res = {}
    for nome in ("disartria", "respiratorio", "fadiga"):
        m = getattr(modelos, nome)
        if m is None or not m.media_negativos:
            continue
        cols = m.colunas
        antes = m.prob(X)
        depois = []
        base = pd.Series(m.media_negativos).reindex(cols)
        for i in range(len(X)):
            outros = X.drop(index=X.index[i]).reindex(columns=cols)
            delta = (base - outros.mean()).fillna(0.0)
            depois.append(float(m.pipeline.predict_proba(X.iloc[[i]].reindex(columns=cols) + delta)[:, 1][0]))
        m.calibrar_dominio(dominio, X)
        res[nome] = {"falantes_saudaveis": int(len(X)), "alertas_sem_calibracao": int((antes >= m.limiar).sum()),
                     "alertas_com_calibracao_loo": int((np.array(depois) >= m.limiar).sum())}
    _salvar_modelos(modelos)
    return res


def etapa_consultas() -> dict:
    from monitor.audio.consulta import features_completas
    from monitor.azure import speech

    (OUT / "consultas").mkdir(parents=True, exist_ok=True)
    modelos = ModelosVoz.carregar()
    linhas, todos = [], []
    base = RAW / "primock57"
    # 1) features de voz de cada consulta (trechos de fala do paciente, segundo o STT)
    cache_f = PROC / "audio_primock_features.csv"
    if cache_f.exists():
        feats_df = pd.read_csv(cache_f, index_col=0)
    else:
        feats = {}
        for cid in sorted(ESPERADO):
            stt = speech.transcrever(base / "audio" / f"{cid}_patient.wav", idioma="en-GB",
                                     referencia=base / "transcripts" / f"{cid}_patient.TextGrid")
            feats[cid] = features_completas(base / "audio" / f"{cid}_patient.wav",
                                            [(x["inicio"], x["fim"]) for x in stt["segmentos"]], modelos.precisa_wav2vec)
        feats_df = pd.DataFrame(feats).T
        feats_df.to_csv(cache_f)
    # 2) calibração de domínio (todos os pacientes do PriMock57 são atores saudáveis)
    calib = calibrar_dominio_loo(modelos, feats_df, "primock")
    print("  calibração de domínio PriMock57:", calib)
    # a variante só com marcadores (usada sem torchaudio) recebe a mesma calibração
    from monitor.audio.modelos import ModeloVoz
    for nome in ("disartria", "respiratorio", "fadiga"):
        arq = MODELOS / f"voz_{nome}_marcadores.joblib"
        if arq.exists():
            mm = ModeloVoz.carregar(arq)
            mm.calibrar_dominio("primock", feats_df)
            mm.salvar(arq)
    for i, cid in enumerate(sorted(ESPERADO)):
        audio = base / "audio" / f"{cid}_patient.wav"
        ref = base / "transcripts" / f"{cid}_patient.TextGrid"
        nota = json.loads((base / "notes" / f"{cid}.json").read_text(encoding="utf-8"))
        pid = f"PRIMOCK-{cid}"
        rel, eventos = analisar_consulta(audio, pid, referencia=ref, modelos=modelos, idioma_fala="en-GB",
                                         idioma_texto="en", dominio="primock", feats=feats_df.loc[cid].to_dict())
        rel["queixa_principal"] = nota.get("presenting_complaint")
        if rel["stt"]["fonte"] in ("azure", "azure_cache"):
            texto_ref = " ".join(s["texto"] for s in ler_textgrid(ref))
            rel["stt"]["wer_vs_manual"] = round(wer(texto_ref, rel["stt"]["texto"]), 3)
        (OUT / "consultas" / f"{cid}.json").write_text(json.dumps(rel, indent=1, ensure_ascii=False, default=str),
                                                       encoding="utf-8")
        todos += eventos
        detectados = set(rel["sinais_alarme"])
        esperado = ESPERADO[cid]
        linhas.append({
            "consulta": cid, "queixa": rel["queixa_principal"], "stt": rel["stt"]["fonte"],
            "linguagem": rel["linguagem"]["fonte"], "wer": rel["stt"].get("wer_vs_manual"),
            "esperado": ",".join(sorted(esperado)) or "-", "detectado": ",".join(sorted(detectados)) or "-",
            "acertou_esperado": esperado <= detectados,
            "sentimento": rel["linguagem"]["sentimento"],
            **{f"p_{k}": v["prob"] for k, v in rel["modelos_voz"].items()},
            "eventos": len(eventos), "severidade_max": max((int(e.severidade) for e in eventos), default=0),
        })
        print(f"  {cid}: {rel['queixa_principal'][:40]!r} → {sorted(detectados)} [{rel['stt']['fonte']}/{rel['linguagem']['fonte']}]")
    tab = pd.DataFrame(linhas)
    tab.to_csv(OUT / "consultas_resumo.csv", index=False)
    salvar_eventos(todos, OUT / "eventos_consultas.json")
    com_queixa = tab[tab["esperado"] != "-"]
    vozes = {k: int((tab[f"p_{k}"] >= getattr(modelos, k).limiar).sum()) for k in ("disartria", "respiratorio", "fadiga")
             if f"p_{k}" in tab and getattr(modelos, k) is not None}
    vozes["consultas"] = int(len(tab))
    return {"consultas": int(len(tab)), "calibracao_dominio_voz": calib, "fonte_stt": tab["stt"].value_counts().to_dict(),
            "fonte_linguagem": tab["linguagem"].value_counts().to_dict(),
            "queixa_principal_detectada": f"{int(com_queixa['acertou_esperado'].sum())}/{len(com_queixa)}",
            "controles_sem_alarme": f"{int((tab[tab['esperado'] == '-']['detectado'] == '-').sum())}/{int((tab['esperado'] == '-').sum())}",
            "wer_medio": round(float(tab["wer"].dropna().mean()), 3) if tab["wer"].notna().any() else None,
            "alertas_de_voz_calibrados_em_falantes_saudaveis": vozes}


def _concatenar(arquivos: list[Path], destino: Path, pausa_s: float = 0.6) -> Path:
    import soundfile as sf

    sr = acustica.SR
    partes = []
    for a in arquivos:
        partes += [acustica.carregar(a), np.zeros(int(pausa_s * sr), dtype=np.float32)]
    destino.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destino, np.concatenate(partes), sr)
    return destino


def etapa_demos() -> dict:
    """Áudios de demonstração com voz alterada, passados pelo mesmo pipeline."""
    modelos = ModelosVoz.carregar()
    demos = PROC / "audio_demos"
    res = {}
    # os modelos respiratório/fadiga (Coswara) também saem do domínio no TORGO → calibra com os controles
    f = pd.read_csv(PROC / "audio_torgo_features.csv")
    w = PROC / "audio_torgo_w2v.csv"
    if w.exists():
        f = f.merge(pd.read_csv(w), on="arquivo")
    controles = f[f["disartria"] == 0].set_index("arquivo")
    res["calibracao_torgo"] = calibrar_dominio_loo(ModelosVoz(None, modelos.respiratorio, modelos.fadiga),
                                                   controles, "torgo")
    casos = {
        "disartria_M05": sorted((RAW / "torgo" / "dysarthria_male").glob("M05_*.wav"))[:15],
        "controle_MC01": sorted((RAW / "torgo" / "non_dysarthria_male").glob("MC01_*.wav"))[:15],
    }
    meta = pd.read_csv(RAW / "coswara" / "combined_data.csv")
    bd = meta[meta["bd"].astype(str).str.lower() == "true"]["id"]
    sadios = meta[(meta["covid_status"] == "healthy") & meta[["bd", "ftg", "cough"]].isna().all(axis=1)]["id"]
    for nome, ids in (("dispneia_coswara", bd), ("saudavel_coswara", sadios)):
        for pid in ids:
            arq = next((RAW / "coswara").glob(f"2*/{pid}/counting-normal.wav"), None)
            if arq is not None and arq.stat().st_size > 50_000:
                casos[nome] = [arq]
                break
    for nome, arquivos in casos.items():
        if not arquivos:
            continue
        wav = _concatenar(arquivos, demos / f"{nome}.wav")
        dominio = "coswara" if "coswara" in nome else "torgo"
        rel, eventos = analisar_consulta(wav, f"DEMO-{nome}", modelos=modelos, dominio=dominio)
        res[nome] = {"modelos_voz": rel["modelos_voz"], "eventos": [e.descricao for e in eventos],
                     "acustica": {k: rel["acustica"].get(k) for k in ("taxa_fala_sil_s", "pausas_por_min",
                                                                      "frase_media_s", "f0_desvio_st", "hnr_db",
                                                                      "cpps_db")}}
        print(f"  demo {nome}: {res[nome]['modelos_voz']}")
    return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--so-consultas", action="store_true")
    p.add_argument("--so-demos", action="store_true")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arq_m = OUT / "metricas.json"
    metricas = json.loads(arq_m.read_text(encoding="utf-8")) if arq_m.exists() else {}
    metricas["azure"] = CONFIG.resumo_azure()
    if not (a.so_consultas or a.so_demos):
        print("[1/4] TORGO — disartria")
        metricas["disartria_torgo"] = etapa_torgo()
        print("  ", {k: v for k, v in metricas["disartria_torgo"].items()
                     if k not in ("nivel_falante", "marcadores_mediana", "comparacao")})
        print("[2/4] Coswara — respiratório e fadiga")
        metricas["coswara"] = etapa_coswara()
        for k, v in metricas["coswara"].items():
            print("  ", k, {kk: vv for kk, vv in v.items() if kk not in ("marcadores_mediana", "comparacao", "variantes")},
                  v.get("variantes"))
    if not a.so_demos:
        print("[3/4] PriMock57 — consultas")
        metricas["consultas_primock"] = etapa_consultas()
        print("  ", metricas["consultas_primock"])
        arq_m.write_text(json.dumps(metricas, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("[4/4] demos de voz alterada")
    if not ((RAW / "torgo").exists() and (RAW / "coswara" / "combined_data.csv").exists()):
        print("  (pulado: precisa de TORGO e Coswara — python scripts/download_data.py --only torgo coswara)")
        print("ok →", OUT)
        return
    metricas["demos"] = etapa_demos()
    arq_m.write_text(json.dumps(metricas, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("ok →", OUT)


if __name__ == "__main__":
    main()
