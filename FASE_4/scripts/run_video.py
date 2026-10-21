"""Experimento de análise de vídeo (REHAB24-6 + UR Fall).

    python scripts/run_video.py                    # tudo (~40 min em CPU; a extração de pose fica em cache)
    python scripts/run_video.py --sem-openpose     # pula a comparação com OpenPose (a parte mais lenta)
    python scripts/run_video.py --so-demos         # só renderiza os vídeos anotados

Etapas
------
1. **Pose (YOLOv8-pose)** de todos os vídeos de fisioterapia (exercícios 1 e 6) e das 70
   gravações do UR Fall → ``data/processed/video/trilhas/*.npz`` (cache).
2. **Acurácia da pose** contra o *motion capture* do REHAB24-6 (ground truth 2D): YOLOv8-pose ×
   OpenPose — PCK@0,2, erro por articulação e erro do **ângulo de abdução do ombro**.
3. **Fisioterapia:** features por repetição, padrão esperado aprendido das repetições corretas
   de *outros* pacientes (deixa-um-paciente-fora) e classificador supervisionado; contagem
   automática de repetições; relatório Markdown por vídeo.
4. **Quedas (UR Fall):** detector de queda + imobilidade, com zona segura do leito pelo YOLOv8;
   sensibilidade, especificidade e latência contra os rótulos por quadro do dataset.
5. **Vídeos anotados** para a demonstração.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitor.config import CONFIG  # noqa: E402
from monitor.events import salvar_eventos  # noqa: E402
from monitor.video import reabilitacao as RB  # noqa: E402
from monitor.video.cinematica import serie_cinematica  # noqa: E402
from monitor.video.pose import (KP, DetectorObjetos, TrilhaPose, criar_backend, extrair_trilha,  # noqa: E402
                                info_video, ler_video)
from monitor.video.quedas import DetectorQuedas, zonas_de_objetos  # noqa: E402

RAW = CONFIG.raw_dir
TRILHAS = CONFIG.processed_dir / "video" / "trilhas"
OUT = CONFIG.results_dir / "video"
FPS_REHAB = 10.0
FPS_QUEDA = 15.0


def _yolo():
    return criar_backend("yolo", pesos=CONFIG.yolo_pose_weights)


# --------------------------------------------------------------------------------------
# 1. Trilhas de pose
# --------------------------------------------------------------------------------------
def videos_rehab() -> list[Path]:
    return sorted((RAW / "rehab24-6" / "videos").glob("Ex*/*.mp4"))


def trilha_rehab(video: Path, backend=None) -> TrilhaPose:
    arq = TRILHAS / f"{video.stem}_yolo.npz"
    if arq.exists():
        return TrilhaPose.carregar(arq)
    t0 = time.time()
    tr = extrair_trilha(video, backend or _yolo(), fps_alvo=FPS_REHAB)
    tr.salvar(arq)
    print(f"  pose {video.name}: {len(tr.t)} quadros em {time.time() - t0:.0f} s")
    return tr


def videos_queda() -> list[Path]:
    return sorted((RAW / "urfall").glob("*-cam0.mp4"))


def trilha_queda(video: Path, backend=None) -> TrilhaPose:
    arq = TRILHAS / f"{video.stem}_yolo.npz"
    if arq.exists():
        return TrilhaPose.carregar(arq)
    tr = extrair_trilha(video, backend or _yolo(), fps_alvo=FPS_QUEDA, recorte_x=(320, 640))
    tr.salvar(arq)
    return tr


# --------------------------------------------------------------------------------------
# 2. Acurácia da pose vs. mocap
# --------------------------------------------------------------------------------------
#: articulação do esqueleto mocap (joints_names.txt) → keypoint COCO
MOCAP_PARA_COCO = {7: "ombro_e", 8: "cotovelo_e", 9: "punho_e", 12: "ombro_d", 13: "cotovelo_d", 14: "punho_d",
                   16: "quadril_e", 17: "joelho_e", 18: "tornozelo_e", 21: "quadril_d", 22: "joelho_d",
                   23: "tornozelo_d"}


def gt_para_coco(gt: np.ndarray) -> np.ndarray:
    kps = np.zeros((len(gt), 17, 3))
    for j, nome in MOCAP_PARA_COCO.items():
        kps[:, KP[nome], :2] = gt[:, j]
        kps[:, KP[nome], 2] = 1.0
    return kps


def acuracia_pose(com_openpose: bool, fps: float = 2.0, max_s: float = 60.0) -> dict:
    """Compara backends com o ground truth em 2 vídeos (um de cada exercício)."""
    from monitor.video.cinematica import angulo

    alvos = [RAW / "rehab24-6" / "videos" / "Ex1" / "PM_001-Camera17-30fps.mp4",
             RAW / "rehab24-6" / "videos" / "Ex6" / "PM_022-Camera17-30fps.mp4"]
    backends = {"yolov8n-pose": _yolo()}
    if com_openpose:
        backends["openpose"] = criar_backend("openpose", lado_max=512)
    juntas = [KP[n] for n in MOCAP_PARA_COCO.values()]
    res: dict = {}
    for video in alvos:
        gt_arq = RAW / "rehab24-6" / "joints2d" / video.parent.name / f"{video.stem.split('-')[0]}-c17-30fps.npy"
        if not gt_arq.exists():
            print(f"  sem ground truth para {video.name}")
            continue
        gt = np.load(gt_arq)
        for nome, be in backends.items():
            erros, pck, ang_err, tempos = [], [], [], []
            for q in ler_video(video, fps, fim_s=max_s):
                if q.indice >= len(gt):
                    break
                t0 = time.time()
                pessoas = be.estimar(q.imagem)
                tempos.append(time.time() - t0)
                g = gt_para_coco(gt[[q.indice]])[0]
                tronco = np.linalg.norm((g[KP["ombro_e"], :2] + g[KP["ombro_d"], :2]) / 2 -
                                        (g[KP["quadril_e"], :2] + g[KP["quadril_d"], :2]) / 2)
                if not pessoas:
                    pck.append(0.0)
                    continue
                # pessoa estimada mais próxima do ground truth (pode haver outra pessoa em cena)
                p = min(pessoas, key=lambda p: np.nanmean(np.linalg.norm(p.keypoints[juntas, :2] - g[juntas, :2], axis=1)))
                d = np.linalg.norm(p.keypoints[juntas, :2] - g[juntas, :2], axis=1) / tronco
                d[p.keypoints[juntas, 2] < 0.1] = np.inf
                erros.append(d)
                pck.append(float(np.mean(d < 0.2)))
                a_est = angulo(p.keypoints[None, KP["quadril_d"], :2], p.keypoints[None, KP["ombro_d"], :2],
                               p.keypoints[None, KP["cotovelo_d"], :2])[0]
                a_gt = angulo(g[None, KP["quadril_d"], :2], g[None, KP["ombro_d"], :2], g[None, KP["cotovelo_d"], :2])[0]
                ang_err.append(abs(a_est - a_gt))
            E = np.array(erros)
            E[np.isinf(E)] = np.nan
            por_junta = {n: round(float(np.nanmedian(E[:, i])), 3) for i, n in enumerate(MOCAP_PARA_COCO.values())} if len(E) else {}
            res.setdefault(video.stem, {})[nome] = {
                "quadros": len(pck), "pck@0.2": round(float(np.mean(pck)), 3),
                "erro_mediano_tronco": round(float(np.nanmedian(E)), 3) if len(E) else None,
                "erro_por_articulacao": por_junta,
                "erro_angulo_abducao_ombro_mediano_graus": round(float(np.median(ang_err)), 1) if ang_err else None,
                "segundos_por_quadro": round(float(np.median(tempos)), 3)}
            print(f"  {video.stem} {nome}: {res[video.stem][nome]['pck@0.2']} PCK, "
                  f"{res[video.stem][nome]['erro_angulo_abducao_ombro_mediano_graus']}° ângulo, "
                  f"{res[video.stem][nome]['segundos_por_quadro']} s/quadro")
    return res


# --------------------------------------------------------------------------------------
# 3. Fisioterapia
# --------------------------------------------------------------------------------------
def repeticoes_rotuladas() -> pd.DataFrame:
    seg = pd.read_csv(RAW / "rehab24-6" / "Segmentation.csv", sep=";")
    linhas = []
    for video in videos_rehab():
        ex = int(video.parent.name.removeprefix("Ex"))
        if ex not in RB.EXERCICIOS:
            continue
        vid = video.stem.split("-")[0]
        cin = RB.preparar_serie(serie_cinematica(*_tk(trilha_rehab(video))))
        for r in seg[(seg["video_id"] == vid) & (seg["exercise_id"] == ex)].itertuples(index=False):
            a, b = r.first_frame / 30.0, r.last_frame / 30.0
            f = RB.features_repeticao(cin[(cin["t"] >= a) & (cin["t"] <= b)], RB.EXERCICIOS[ex])
            linhas.append({"video": vid, "arquivo": video.name, "exercicio": ex, "pessoa": r.person_id,
                           "repeticao": r.repetition_number, "inicio": a, "fim": b, "correta": int(r.correctness),
                           "vista": r.cam17_orientation, "luz": r.lights_on, **f})
    return pd.DataFrame(linhas)


def _tk(tr: TrilhaPose):
    return tr.t, tr.kps


def avaliar_reabilitacao(reps: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    res, saida = {}, []
    for ex_id, g in reps.groupby("exercicio"):
        ex = RB.EXERCICIOS[ex_id]
        g = g.copy()
        feats = list(ex.features)
        g["escore"] = np.nan
        g["prob_sup"] = np.nan
        g["desvios"] = [[] for _ in range(len(g))]
        for pessoa in g["pessoa"].unique():
            teste = g["pessoa"] == pessoa
            treino = g[~teste]
            # padrão esperado: repetições corretas dos outros, mesma vista se houver amostra suficiente
            for vista in g.loc[teste, "vista"].unique():
                ref = treino[(treino["correta"] == 1) & (treino["vista"] == vista)]
                if len(ref) < 8:
                    ref = treino[treino["correta"] == 1]
                padrao = RB.PadraoEsperado.de_repeticoes(ref, ex)
                idx = g.index[teste & (g["vista"] == vista)]
                for i in idx:
                    f = g.loc[i, feats].to_dict()
                    g.at[i, "escore"] = padrao.escore(f, ex)
                    g.at[i, "desvios"] = padrao.desvios(f, ex)
            X = pd.get_dummies(g[feats + ["vista"]], columns=["vista"], dtype=float)
            clf = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                LogisticRegression(C=0.5, class_weight="balanced", max_iter=5000))
            clf.fit(X[~teste], 1 - g.loc[~teste, "correta"])
            g.loc[teste, "prob_sup"] = clf.predict_proba(X[teste])[:, 1]
        y = 1 - g["correta"].to_numpy()            # positivo = execução incorreta
        g["fora_padrao"] = g["escore"] >= 2.5
        pred = g["fora_padrao"].to_numpy()
        res[ex.nome] = {
            "repeticoes": int(len(g)), "incorretas": int(y.sum()), "pacientes": int(g["pessoa"].nunique()),
            "padrao_esperado": {"auroc": round(float(roc_auc_score(y, g["escore"])), 3),
                                "limiar_z": 2.5,
                                "sensibilidade": round(float(pred[y == 1].mean()), 3),
                                "especificidade": round(float((~pred[y == 0]).mean()), 3),
                                "acuracia_balanceada": round(float(balanced_accuracy_score(y, pred)), 3)},
            "supervisionado_lopo": {"auroc": round(float(roc_auc_score(y, g["prob_sup"])), 3),
                                    "acuracia_balanceada": round(float(balanced_accuracy_score(y, g["prob_sup"] >= 0.5)), 3)},
            "desvios_mais_frequentes": pd.Series([d["feature"] for ds in g.loc[g["fora_padrao"], "desvios"] for d in ds[:1]])
                                          .value_counts().head(5).to_dict(),
            "features_mediana": g.groupby("correta")[feats].median().round(2).rename(index={0: "incorreta", 1: "correta"})
                                 .to_dict(orient="index"),
        }
        saida.append(g)
    return res, pd.concat(saida, ignore_index=True)


def contagem_automatica(reps: pd.DataFrame) -> dict:
    out = {}
    for video in videos_rehab():
        ex_id = int(video.parent.name.removeprefix("Ex"))
        ex = RB.EXERCICIOS[ex_id]
        cin = RB.preparar_serie(serie_cinematica(*_tk(trilha_rehab(video))))
        auto = RB.segmentar_repeticoes(cin, ex)
        gt = int((reps["arquivo"] == video.name).sum())
        out[video.stem.split("-")[0]] = {"exercicio": ex_id, "gt": gt, "auto": len(auto)}
    erros = [abs(v["auto"] - v["gt"]) for v in out.values()]
    return {"por_video": out, "erro_absoluto_medio": round(float(np.mean(erros)), 2),
            "videos_contagem_exata": f"{sum(e == 0 for e in erros)}/{len(erros)}"}


def relatorio_sessao(video_id: str, g: pd.DataFrame, destino: Path) -> list:
    ex = RB.EXERCICIOS[int(g["exercicio"].iloc[0])]
    g = g.sort_values("repeticao")
    eventos = RB.eventos_sessao(f"REHAB-{video_id}", g)
    linhas = [f"# Relatório automático — sessão {video_id} ({ex.nome})", "",
              f"* Repetições analisadas: **{len(g)}** · fora do padrão: **{int(g['fora_padrao'].sum())}** · "
              f"rótulo do fisioterapeuta (dataset): {int((g['correta'] == 0).sum())} incorretas",
              f"* Vista da câmera: {g['vista'].iloc[0]} · alertas gerados: {len(eventos)}", "",
              "| Rep. | Tempo (s) | " + " | ".join(ex.features[f][0] for f in list(ex.features)[:4]) +
              " | Escore | Situação | Rótulo |", "|" + "---|" * 9]
    for r in g.itertuples(index=False):
        vals = " | ".join(f"{getattr(r, f):.1f}" if pd.notna(getattr(r, f)) else "–" for f in list(ex.features)[:4])
        sit = "⚠️ fora do padrão" if r.fora_padrao else "ok"
        linhas.append(f"| {r.repeticao} | {r.inicio:.1f}–{r.fim:.1f} | {vals} | {r.escore:.1f} | {sit} | "
                      f"{'correta' if r.correta else 'incorreta'} |")
    linhas += ["", "## Desvios detectados", ""]
    for r in g[g["fora_padrao"]].itertuples(index=False):
        linhas.append(f"* **Repetição {r.repeticao}** ({r.inicio:.1f}–{r.fim:.1f} s): " +
                      "; ".join(RB.texto_desvio(d) for d in r.desvios[:3]))
    if not g["fora_padrao"].any():
        linhas.append("Nenhum desvio relevante em relação ao padrão esperado.")
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return eventos


# --------------------------------------------------------------------------------------
# 4. Quedas
# --------------------------------------------------------------------------------------
def rotulos_urfall() -> pd.DataFrame:
    cols = ["seq", "quadro", "rotulo"]
    partes = [pd.read_csv(RAW / "urfall" / n, header=None, usecols=[0, 1, 2], names=cols)
              for n in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")]
    return pd.concat(partes, ignore_index=True)


def zonas_video(video: Path, det: DetectorObjetos) -> list:
    """Detecta leito/sofá nos primeiros quadros (a mobília não se move)."""
    zonas = []
    for k, q in enumerate(ler_video(video, 2.0, recorte_x=(320, 640))):
        zonas += zonas_de_objetos(det.detectar(q.imagem))
        if k >= 2:
            break
    return zonas[:2]


def avaliar_quedas() -> tuple[dict, dict]:
    det = DetectorObjetos(CONFIG.yolo_det_weights, conf=0.35)
    rot = rotulos_urfall()
    linhas, eventos_por_video = [], {}
    for video in videos_queda():
        seq = video.stem.replace("-cam0", "")
        tr = trilha_queda(video)
        zonas = zonas_video(video, det)
        dq = DetectorQuedas(f"URFALL-{seq}", zonas_seguras=zonas)
        evs = []
        for i in range(len(tr.t)):
            evs += dq.atualizar(float(tr.t[i]), tr.kps[i], tr.caixas[i])
        eventos_por_video[seq] = (evs, zonas)
        r = rot[rot["seq"] == seq]
        inicio_queda = r.loc[r["rotulo"] >= 0, "quadro"].min() / 30.0 if (r["rotulo"] >= 0).any() else np.nan
        queda_ev = [e for e in evs if e.tipo == "queda"]
        linhas.append({"seq": seq, "queda_real": seq.startswith("fall"), "detectou": bool(queda_ev),
                       "adl_deita": bool(seq.startswith("adl") and (r["rotulo"] == 1).any()),
                       "fim_s": float(tr.t[-1]) if len(tr.t) else 0.0,
                       "t_queda": queda_ev[0].t if queda_ev else np.nan,
                       "latencia_s": (queda_ev[0].t - inicio_queda) if queda_ev and not np.isnan(inicio_queda) else np.nan,
                       "imobilidade": any(e.tipo == "imobilidade_pos_queda" for e in evs),
                       "leito": any(e.tipo == "deitou_no_leito" for e in evs), "zonas": len(zonas)})
    tab = pd.DataFrame(linhas)
    tab.to_csv(OUT / "quedas_por_video.csv", index=False)
    q, a = tab[tab["queda_real"]], tab[~tab["queda_real"]]
    tp, fn = int(q["detectou"].sum()), int((~q["detectou"]).sum())
    fp, tn = int(a["detectou"].sum()), int((~a["detectou"]).sum())
    m = {"videos_queda": len(q), "videos_atividade_normal": len(a), "vp": tp, "fn": fn, "fp": fp, "vn": tn,
         "sensibilidade": round(tp / max(tp + fn, 1), 3), "especificidade": round(tn / max(tn + fp, 1), 3),
         "latencia_mediana_s": round(float(q["latencia_s"].median()), 2) if q["latencia_s"].notna().any() else None,
         "imobilidade_pos_queda_detectada": int(q["imobilidade"].sum()),
         "especificidade_adl_sem_deitar": f"{int((~a.loc[~a['adl_deita'], 'detectou']).sum())}/{int((~a['adl_deita']).sum())}",
         "alertas_em_adl_deitando_no_chao_ou_cama": f"{int(a.loc[a['adl_deita'], 'detectou'].sum())}/{int(a['adl_deita'].sum())}",
         "segundos_de_video_apos_a_queda_mediana": round(float((q["fim_s"] - q["t_queda"]).median()), 2),
         "adl_com_leito_detectado": int(a["zonas"].gt(0).sum()), "adl_deitou_no_leito": int(a["leito"].sum())}
    return m, eventos_por_video


# --------------------------------------------------------------------------------------
# 5. Demos
# --------------------------------------------------------------------------------------
def _h264(arq: Path) -> Path:
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return arq
    saida = arq.with_name(arq.stem + "_h264.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(arq), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "26", str(saida)], check=False)
    if saida.exists():
        arq.unlink()
        saida.rename(arq)
    return arq


def demos(reps_aval: pd.DataFrame, eventos_quedas: dict) -> list[str]:
    from monitor.video.anotar import renderizar

    feitos = []
    # sessão de fisioterapia com mais repetições fora do padrão (exercício 1)
    g1 = reps_aval[reps_aval["exercicio"] == 1]
    vid = g1.groupby("video")["fora_padrao"].sum().idxmax()
    sub = g1[g1["video"] == vid]
    video = RAW / "rehab24-6" / "videos" / "Ex1" / f"{vid}-Camera17-30fps.mp4"
    tr = trilha_rehab(video)
    cin = RB.preparar_serie(serie_cinematica(tr.t, tr.kps))
    evs = RB.eventos_sessao(f"REHAB-{vid}", sub.sort_values("repeticao"))

    def info(t):
        i = int(np.argmin(np.abs(cin["t"] - t)))
        rep = sub[(sub["inicio"] <= t) & (sub["fim"] >= t)]
        txt = f"abducao ombro D: {cin['abducao_ombro_d'].iloc[i]:5.0f} graus | cotovelo D: {cin['cotovelo_d'].iloc[i]:4.0f} | tronco: {cin['inclinacao_tronco'].iloc[i]:3.0f}"
        return [txt + (f" | rep {int(rep['repeticao'].iloc[0])}" if len(rep) else "")]
    fim = float(sub["fim"].max()) + 2
    tr_corte = TrilhaPose(tr.video, tr.backend, tr.fps_amostrado, tr.t[tr.t <= fim], tr.quadro[tr.t <= fim],
                          tr.kps[tr.t <= fim], tr.caixas[tr.t <= fim], tr.n_pessoas[tr.t <= fim], tr.largura, tr.altura)
    arq = renderizar(video, tr_corte, evs, OUT / "demo" / f"fisioterapia_{vid}.mp4",
                     f"Fisioterapia {vid} - abducao do braco (YOLOv8-pose)", info_por_t=info)
    feitos.append(str(_h264(arq)))
    # queda real com mais tempo de vídeo depois do alerta (prefere a que também teve imobilidade)
    quedas = []
    for seq, (evs, zonas) in eventos_quedas.items():
        q = [e for e in evs if e.tipo == "queda"]
        if seq.startswith("fall") and q:
            fim = float(trilha_queda(RAW / "urfall" / f"{seq}-cam0.mp4").t[-1])
            quedas.append((any(e.tipo == "imobilidade_pos_queda" for e in evs), fim - q[0].t, seq))
    if quedas:
        seq = max(quedas)[2]
        evs, zonas = eventos_quedas[seq]
        video = RAW / "urfall" / f"{seq}-cam0.mp4"
        arq = renderizar(video, trilha_queda(video), evs, OUT / "demo" / f"queda_{seq}.mp4",
                         f"Quarto - {seq}", recorte_x=(320, 640), largura_saida=640, zonas=zonas)
        feitos.append(str(_h264(arq)))
    # atividade normal: deitar no leito (não deve alertar)
    for seq, (evs, zonas) in eventos_quedas.items():
        if seq.startswith("adl") and zonas and not evs:
            video = RAW / "urfall" / f"{seq}-cam0.mp4"
            arq = renderizar(video, trilha_queda(video), evs, OUT / "demo" / f"normal_{seq}.mp4",
                             f"Quarto - {seq} (atividade normal)", recorte_x=(320, 640), largura_saida=640, zonas=zonas)
            feitos.append(str(_h264(arq)))
            break
    return feitos


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sem-openpose", action="store_true")
    p.add_argument("--so-demos", action="store_true")
    p.add_argument("--pular-acuracia", action="store_true", help="mantém a comparação de pose já salva")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arq_m = OUT / "metricas.json"
    metricas = json.loads(arq_m.read_text(encoding="utf-8")) if arq_m.exists() else {}

    print("[1/5] trilhas de pose (YOLOv8-pose)")
    be = _yolo()
    for v in videos_rehab():
        trilha_rehab(v, be)
    for v in videos_queda():
        trilha_queda(v, be)

    if not (a.so_demos or a.pular_acuracia):
        print("[2/5] acurácia da pose vs. motion capture")
        metricas["acuracia_pose"] = acuracia_pose(com_openpose=not a.sem_openpose)

    print("[3/5] fisioterapia")
    reps = repeticoes_rotuladas()
    res, reps_aval = avaliar_reabilitacao(reps)
    reps_aval.drop(columns=["desvios"]).to_csv(OUT / "repeticoes.csv", index=False)
    if not a.so_demos:
        metricas["fisioterapia"] = res
        metricas["contagem_repeticoes"] = contagem_automatica(reps)
        for k, v in res.items():
            print(f"  {k}: padrão esperado {v['padrao_esperado']} | supervisionado {v['supervisionado_lopo']}")
        print("  contagem:", {k: v for k, v in metricas["contagem_repeticoes"].items() if k != "por_video"})
    todos = []
    for vid, g in reps_aval.groupby("video"):
        todos += relatorio_sessao(vid, g, OUT / "relatorios" / f"{vid}.md")
    salvar_eventos(todos, OUT / "eventos_fisioterapia.json")

    print("[4/5] quedas")
    mq, eventos_quedas = avaliar_quedas()
    metricas["quedas"] = mq
    print("  ", mq)
    salvar_eventos([e for evs, _ in eventos_quedas.values() for e in evs], OUT / "eventos_quedas.json")

    print("[5/5] vídeos anotados")
    metricas["demos"] = demos(reps_aval, eventos_quedas)
    print("  ", metricas["demos"])
    arq_m.write_text(json.dumps(metricas, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("ok →", OUT)


if __name__ == "__main__":
    main()
