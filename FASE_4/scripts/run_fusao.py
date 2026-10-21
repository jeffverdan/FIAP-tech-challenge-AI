"""Fusão multimodal: (1) dados reais do MIMIC-IV demo e (2) cenário composto do Hospital Aurora.

    python scripts/run_fusao.py

Pré-requisito: ``run_vitals.py``, ``run_audio.py`` e ``run_video.py`` já executados (os eventos de
cada modalidade são lidos de ``results/``).

1. **MIMIC-IV demo (real, mesmo paciente):** sinais vitais horários (regras + NEWS2) e prescrições
   de cada internação entram no motor de fusão. Mede quantos alertas cada regra de correlação
   gera e mostra exemplos — é a única fonte pública com duas modalidades do *mesmo* paciente.
2. **Cenário "Leito 12" (composto):** um paciente do Hospital Aurora cuja monitorização junta
   eventos reais de datasets diferentes, posicionados numa linha do tempo plausível — sinais
   vitais do VitalDB, consulta do PriMock57, sessão de fisioterapia do REHAB24-6, queda do UR
   Fall e uma prescrição do MIMIC. É o cenário que o dashboard reproduz em "tempo real".
   **Deixamos explícito no relatório que as modalidades vêm de pessoas diferentes.**
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitor import prescricoes  # noqa: E402
from monitor.config import CONFIG  # noqa: E402
from monitor.events import Evento, carregar_eventos, salvar_eventos  # noqa: E402
from monitor.fusion.motor import MotorFusao  # noqa: E402
from monitor.vitals.canais import carregar_mimic_vitais  # noqa: E402
from monitor.vitals.detectores import PERFIL_HORARIO, DetectorNEWS2, DetectorRegras  # noqa: E402

RAW = CONFIG.raw_dir
RES = CONFIG.results_dir
OUT = RES / "fusao"
PACIENTE = "AURORA-LEITO-12"
CASO_VITAIS = 49


# --------------------------------------------------------------------------------------
# 1. MIMIC-IV demo
# --------------------------------------------------------------------------------------
def eventos_mimic() -> tuple[list[Evento], pd.DataFrame]:
    base = RAW / "mimic-iv-demo"
    vit = carregar_mimic_vitais(base / "icu" / "chartevents.csv.gz")
    stays = pd.read_csv(base / "icu" / "icustays.csv.gz")[["stay_id", "hadm_id"]]
    adm = pd.read_csv(base / "hosp" / "admissions.csv.gz", parse_dates=["admittime", "dischtime"])
    vit = vit.merge(stays, on="stay_id").merge(adm[["hadm_id", "admittime"]], on="hadm_id")
    vit["t_adm"] = (vit["charttime"] - vit["admittime"]).dt.total_seconds()
    eventos: list[Evento] = []
    for hadm, g in vit.sort_values("t_adm").groupby("hadm_id"):
        pid = f"MIMIC-H{int(hadm)}"
        regras = DetectorRegras(pid, PERFIL_HORARIO, max_gap_s=4 * 3600)
        news = DetectorNEWS2(pid)
        for r in g.itertuples(index=False):
            amostra = r._asdict()
            eventos += regras.atualizar(float(r.t_adm), amostra) + news.atualizar(float(r.t_adm), amostra)
    _, ev_rx = prescricoes.analisar(base)
    adm["dias"] = (adm["dischtime"] - adm["admittime"]).dt.total_seconds() / 86400
    return eventos + ev_rx, adm


def fusao_mimic() -> dict:
    eventos, adm = eventos_mimic()
    motor = MotorFusao()
    alertas = motor.processar_lote(eventos)
    com_vitais = {e.paciente_id for e in eventos if e.origem == "vitais"}
    pacientes_dia = float(adm[adm["hadm_id"].map(lambda h: f"MIMIC-H{int(h)}").isin(com_vitais)]["dias"].sum())
    fusao = [a for a in alertas if a.regra.startswith("fusao:")]
    por_regra = Counter(a.regra.removeprefix("fusao:") for a in fusao)
    exemplos = []
    for regra in por_regra:
        a = next(x for x in fusao if x.regra == f"fusao:{regra}")
        exemplos.append({"regra": regra, "paciente": a.paciente_id, "t_h": round(a.t / 3600, 1),
                         "severidade": a.severidade.rotulo, "titulo": a.titulo, "justificativa": a.justificativa})
    salvar_eventos(eventos, OUT / "mimic_eventos.json")
    (OUT / "mimic_alertas.json").write_text(json.dumps([a.to_dict() for a in alertas], ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    sev = Counter(a.severidade.rotulo for a in alertas)
    return {
        "internacoes_com_vitais_e_prescricao": len(com_vitais & {e.paciente_id for e in eventos if e.origem == "prescricao"}),
        "pacientes_dia": round(pacientes_dia, 1),
        "eventos": dict(Counter(e.origem for e in eventos)),
        "alertas_total": len(alertas), "alertas_por_severidade": dict(sev),
        "alertas_por_paciente_dia": round(len(alertas) / max(pacientes_dia, 1e-9), 2),
        "eventos_suprimidos_por_repeticao": int(sum(a.repeticoes - 1 for a in alertas)),
        "alertas_fusao_por_regra": dict(por_regra),
        "internacoes_com_alerta_de_fusao": len({a.paciente_id for a in fusao}),
        "exemplos": exemplos,
    }


# --------------------------------------------------------------------------------------
# 2. Cenário composto
# --------------------------------------------------------------------------------------
def inserir(eventos: list[Evento], t_inicio: float, fonte: str) -> list[Evento]:
    out = []
    for e in eventos:
        d = e.to_dict()
        d.update(paciente_id=PACIENTE, t=round(t_inicio + e.t, 1), id=f"EVT-{uuid.uuid4().hex[:10].upper()}")
        d["evidencias"] = {**d.get("evidencias", {}), "fonte_dataset": fonte}
        out.append(Evento.from_dict(d))
    return out


def montar_cenario() -> dict:
    vit = carregar_eventos(RES / "vitals" / f"eventos_caso_{CASO_VITAIS:04d}.json")
    t_dessat = min(e.t for e in vit if e.tipo.startswith("dessaturacao"))
    t_hipo = min(e.t for e in vit if e.tipo.startswith("hipotensao"))

    # consulta com queixa respiratória termina 10 min antes da primeira dessaturação
    consulta = json.loads((RES / "audio" / "consultas" / "day2_consultation04.json").read_text(encoding="utf-8"))
    ev_audio = [Evento.from_dict(d) for d in consulta["eventos"]]
    dur_consulta = max((e.t for e in ev_audio), default=0.0)
    t_consulta = max(t_dessat - 600 - dur_consulta, 0)

    # prescrição real do MIMIC: associação opioide + benzodiazepínico, 25 min antes
    _, rx = prescricoes.analisar(RAW / "mimic-iv-demo")
    rx_ex = next(e for e in rx if e.tipo == "opioide_benzodiazepinico")
    rx_ex.t = 0.0

    # sessão de fisioterapia logo no início do turno; queda 5 min após a hipotensão
    fisio = [e for e in carregar_eventos(RES / "video" / "eventos_fisioterapia.json")]
    sessao = Counter(e.paciente_id for e in fisio).most_common(1)[0][0]
    ev_fisio = [e for e in fisio if e.paciente_id == sessao]
    quedas = carregar_eventos(RES / "video" / "eventos_quedas.json")
    demo_queda = next((RES / "video" / "demo").glob("queda_fall-*.mp4"), None)
    seq = f"URFALL-{demo_queda.stem.removeprefix('queda_')}" if demo_queda else \
        next(e.paciente_id for e in quedas if e.tipo == "queda" and e.paciente_id.startswith("URFALL-fall"))
    ev_queda = [e for e in quedas if e.paciente_id == seq]
    t0_queda = min(e.t for e in ev_queda)

    blocos = [
        {"modalidade": "prescricao", "fonte": f"MIMIC-IV demo ({rx_ex.paciente_id})", "t_inicio": t_dessat - 1500,
         "eventos": inserir([rx_ex], t_dessat - 1500, "MIMIC-IV demo")},
        {"modalidade": "audio", "fonte": "PriMock57 day2_consultation04 (\"I feel breathless\")", "t_inicio": t_consulta,
         "midia": "results/audio/consultas/day2_consultation04.json",
         "eventos": inserir(ev_audio, t_consulta, "PriMock57")},
        {"modalidade": "video", "fonte": f"REHAB24-6 {sessao.removeprefix('REHAB-')} (abdução do braço)", "t_inicio": 300.0,
         "midia": f"results/video/demo/fisioterapia_{sessao.removeprefix('REHAB-')}.mp4",
         "eventos": inserir(ev_fisio, 300.0, "REHAB24-6")},
        {"modalidade": "movimentacao", "fonte": f"UR Fall {seq.removeprefix('URFALL-')}", "t_inicio": t_hipo + 300 - t0_queda,
         "midia": f"results/video/demo/queda_{seq.removeprefix('URFALL-')}.mp4",
         "eventos": inserir(ev_queda, t_hipo + 300 - t0_queda, "UR Fall")},
    ]
    demo = CONFIG.processed_dir / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    shutil.copy(RAW / "vitaldb" / f"case_{CASO_VITAIS:04d}.csv.gz", demo / f"vitaldb_case_{CASO_VITAIS:04d}.csv.gz")
    cenario = {
        "paciente_id": PACIENTE, "descricao": "Paciente composto do Hospital Aurora (modalidades de datasets públicos distintos)",
        "vitais": {"fonte": f"VitalDB caso {CASO_VITAIS}", "arquivo": f"data/processed/demo/vitaldb_case_{CASO_VITAIS:04d}.csv.gz",
                   "modelos": ["data/processed/models/iforest_vitais.joblib", "data/processed/models/preditor_hipotensao.joblib"]},
        "blocos": [{**b, "eventos": [e.to_dict() for e in b["eventos"]]} for b in blocos],
    }
    (CONFIG.processed_dir / "cenario_aurora.json").write_text(json.dumps(cenario, ensure_ascii=False, indent=1,
                                                                         default=str), encoding="utf-8")
    # roda o cenário inteiro (vitais pré-computados + blocos) no motor, para o relatório
    todos = inserir(vit, 0.0, f"VitalDB {CASO_VITAIS}") + [e for b in blocos for e in b["eventos"]]
    motor = MotorFusao()
    alertas = motor.processar_lote(todos)
    (OUT / "cenario_alertas.json").write_text(json.dumps([a.to_dict() for a in alertas], ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    figura_cenario(todos, alertas, OUT / "cenario_timeline.png")
    curva = [{"t_min": round(t / 60, 1), **motor.escore(PACIENTE, t)} for t in np.arange(0, max(e.t for e in todos) + 600, 300)]
    return {"eventos": len(todos), "eventos_por_modalidade": dict(Counter(e.origem for e in todos)),
            "alertas": len(alertas), "alertas_por_severidade": dict(Counter(a.severidade.rotulo for a in alertas)),
            "alertas_fusao": [{"t_min": round(a.t / 60, 1), "severidade": a.severidade.rotulo, "titulo": a.titulo,
                               "modalidades": a.modalidades, "justificativa": a.justificativa}
                              for a in alertas if a.regra.startswith("fusao:")],
            "escore_maximo": max(curva, key=lambda c: c["escore"])}


def figura_cenario(eventos, alertas, destino: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mods = ["vitais", "prescricao", "audio", "texto", "video", "movimentacao"]
    nomes = {"vitais": "sinais vitais", "prescricao": "prescrição", "audio": "áudio (consulta)", "texto": "texto",
             "video": "vídeo (fisioterapia)", "movimentacao": "vídeo (quarto)"}
    cores = {0: "#bdc3c7", 1: "#f1c40f", 2: "#e67e22", 3: "#c0392b"}
    fusao = [a for a in alertas if a.regra.startswith("fusao:")]
    fig, ax = plt.subplots(figsize=(12, 4.6 + 0.22 * len(fusao)))
    for e in eventos:
        ax.scatter(e.t / 60, mods.index(e.origem), color=cores[int(e.severidade)], s=30 + 20 * int(e.severidade),
                   edgecolor="k", lw=0.3, zorder=3)
    legenda = []
    for i, a in enumerate(fusao, 1):
        ax.axvline(a.t / 60, color="#c0392b", lw=1.2, ls="--", zorder=1)
        ax.text(a.t / 60, len(mods) - 0.5 + 0.25 * (i % 2), f" {i}", fontsize=9, color="#c0392b", fontweight="bold")
        legenda.append(f"{i}. {a.t / 60:.0f} min — [{a.severidade.rotulo}] {a.titulo}  ({' + '.join(a.modalidades)})")
    ax.set_yticks(range(len(mods)), [nomes[m] for m in mods])
    ax.set_ylim(-0.6, len(mods) + 0.1)
    ax.set_xlabel("minutos desde o início do turno")
    ax.set_title(f"{PACIENTE} — eventos por modalidade (cor = severidade) e alertas de fusão (tracejado)")
    ax.grid(alpha=0.3)
    fig.text(0.07, 0.01, "\n".join(legenda), fontsize=8.5, color="#7b241c", va="bottom")
    fig.tight_layout(rect=(0, 0.035 + 0.04 * len(fusao), 1, 1))
    fig.savefig(destino, dpi=110)
    plt.close(fig)


#: Leitos da Central (vista de vários pacientes). O Leito 12 é o cenário multimodal completo; os demais
#: são cirurgias do conjunto de TESTE do VitalDB (nunca vistas pelos modelos), só com sinais vitais.
LEITOS_CENTRAL = [
    ("Leito 03", 166, "dessaturação e hipotensão"),
    ("Leito 05", 161, "hipotensão tardia"),
    ("Leito 07", 128, "dessaturação leve"),
    ("Leito 08", 177, "bradicardia e dessaturação"),
    ("Leito 10", 153, "hipotensão isolada"),
    ("Leito 11", 132, "estável"),
    ("Leito 14", 167, "estável"),
]


def montar_central() -> dict:
    demo = CONFIG.processed_dir / "demo"
    demo.mkdir(parents=True, exist_ok=True)
    leitos = [{"leito": "Leito 12", "paciente_id": PACIENTE, "cenario": True,
               "fonte": "cenário multimodal composto", "arquivo": f"data/processed/demo/vitaldb_case_{CASO_VITAIS:04d}.csv.gz"}]
    for nome, caso, perfil in LEITOS_CENTRAL:
        shutil.copy(RAW / "vitaldb" / f"case_{caso:04d}.csv.gz", demo / f"vitaldb_case_{caso:04d}.csv.gz")
        leitos.append({"leito": nome, "paciente_id": f"VDB-{caso:04d}", "cenario": False,
                       "fonte": f"VitalDB caso {caso} (teste) — {perfil}",
                       "arquivo": f"data/processed/demo/vitaldb_case_{caso:04d}.csv.gz"})
    (CONFIG.processed_dir / "central_leitos.json").write_text(json.dumps(leitos, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
    return {"leitos": len(leitos)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/2] MIMIC-IV demo")
    m = {"mimic": fusao_mimic()}
    print("  ", {k: v for k, v in m["mimic"].items() if k != "exemplos"})
    print("[2/2] cenário composto Leito 12")
    m["cenario"] = montar_cenario()
    print("  ", {k: v for k, v in m["cenario"].items() if k != "alertas_fusao"})
    for a in m["cenario"]["alertas_fusao"]:
        print(f"   {a['t_min']:6.1f} min [{a['severidade']}] {a['titulo']} ← {a['modalidades']}")
    m["central"] = montar_central()
    (OUT / "metricas.json").write_text(json.dumps(m, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("ok →", OUT)


if __name__ == "__main__":
    main()
