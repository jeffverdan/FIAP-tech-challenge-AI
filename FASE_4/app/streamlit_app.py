"""Central de Alertas — Hospital Aurora (FASE 4).

    streamlit run app/streamlit_app.py
    # no Windows, abra http://127.0.0.1:8501 (não "localhost")

Abas:
  * **Tempo real** — replay acelerado do paciente "Leito 12": os sinais vitais passam pelo
    monitor *streaming* de verdade; consulta, fisioterapia, queda e prescrição chegam como eventos
    nos instantes do cenário; o motor de fusão decide os alertas ao vivo.
  * **Áudio**, **Vídeo**, **Sinais vitais & prescrições** — resultados de cada modalidade.
  * **Azure & sistema** — configuração ativa, cache e teste das credenciais.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from monitor.config import CONFIG  # noqa: E402
from monitor.events import Evento, Severidade  # noqa: E402
from monitor.fusion.motor import MotorFusao  # noqa: E402
from monitor.fusion.notificacao import Notificador  # noqa: E402
from monitor.vitals.canais import carregar_vitaldb, limpar  # noqa: E402

st.set_page_config(page_title="Aurora · Central de Alertas", page_icon="🩺", layout="wide")

COR = {0: "#7f8c8d", 1: "#d4ac0d", 2: "#e67e22", 3: "#c0392b"}
ICONE = {"vitais": "💓", "audio": "🎙️", "texto": "💬", "video": "🎥", "movimentacao": "🚶", "prescricao": "💊"}
NIVEL_COR = {"verde": "#27ae60", "amarelo": "#d4ac0d", "laranja": "#e67e22", "vermelho": "#c0392b"}
RES = CONFIG.results_dir


def _json(caminho: Path, padrao=None):
    try:
        return json.loads(Path(caminho).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return padrao


# ======================================================================================
# Tempo real
# ======================================================================================
@st.cache_resource
def _modelos():
    from monitor.vitals.detectores import ModeloIForest
    from monitor.vitals.predicao import ModeloHipotensao

    pasta = CONFIG.processed_dir / "models"
    ifo = pasta / "iforest_vitais.joblib"
    pre = pasta / "preditor_hipotensao.joblib"
    return (ModeloIForest.carregar(ifo) if ifo.exists() else None,
            ModeloHipotensao.carregar(pre) if pre.exists() else None)


@st.cache_data
def _cenario():
    c = _json(CONFIG.processed_dir / "cenario_aurora.json")
    if c is None:
        return None, None
    df, _ = limpar(carregar_vitaldb(RAIZ / c["vitais"]["arquivo"]))
    ok = df[["HR", "MAP"]].notna().all(axis=1)
    df = df.loc[ok.idxmax(): ok[::-1].idxmax()].reset_index(drop=True)
    df["t_s"] -= df["t_s"].iloc[0]
    return c, df


def _reiniciar(t_inicial: float = 0.0):
    from monitor.vitals.detectores import MonitorSinaisVitais

    c, df = _cenario()
    ifo, pre = _modelos()
    ss = st.session_state
    ss.t_sim = 0.0
    ss.rodando = False
    ss.monitor = MonitorSinaisVitais(c["paciente_id"], ifo, modelo_hipotensao=pre)
    ss.motor = MotorFusao()
    ss.notificador = Notificador(sessao=f"replay-{int(time.time())}")
    ss.eventos, ss.alertas, ss.entregas = [], [], {}
    ss.pendentes = sorted((Evento.from_dict(e) for b in c["blocos"] for e in b["eventos"]), key=lambda e: e.t)
    ss.idx = 0
    if t_inicial > 0:
        _avancar(t_inicial)


def _avancar(ate: float):
    ss = st.session_state
    c, df = _cenario()
    ts = df["t_s"].to_numpy()
    fim = int(np.searchsorted(ts, ate, side="right"))
    novos_ev = []
    for i in range(ss.idx, fim):
        linha = df.iloc[i]
        t = float(linha["t_s"])
        while ss.pendentes and ss.pendentes[0].t <= t:
            novos_ev.append(ss.pendentes.pop(0))
        novos_ev += ss.monitor.atualizar(t, {k: linha[k] for k in df.columns if k != "t_s"})
    while ss.pendentes and ss.pendentes[0].t <= ate:
        novos_ev.append(ss.pendentes.pop(0))
    ss.idx = fim
    ss.t_sim = ate
    for ev in sorted(novos_ev, key=lambda e: e.t):
        ss.eventos.append(ev)
        for a in ss.motor.processar(ev):
            ss.alertas.append(a)
            ss.entregas[a.id] = ss.notificador.entregar(a)


def _cartao_alerta(a, entrega=None):
    cor = COR[int(a.severidade)]
    # alerta direto: a justificativa é o próprio evento (já está no título)
    just = "" if a.regra.startswith("direto:") else "".join(f"<li>{j}</li>" for j in a.justificativa)
    fusao = "🔗 FUSÃO MULTIMODAL · " if a.regra.startswith("fusao:") else ""
    canais = f" · canais: {', '.join(entrega['canais'])}" if entrega else ""
    rep = f" · repetiu {a.repeticoes}×" if a.repeticoes > 1 else ""
    st.markdown(
        f"""<div style="border-left:6px solid {cor};padding:6px 10px;margin:6px 0;background:rgba(127,127,127,.07);border-radius:4px">
        <div style="font-size:.8em;color:{cor};font-weight:700">{fusao}{a.severidade.rotulo.upper()} · t = {a.t / 60:.1f} min · {' '.join(ICONE.get(m, '') for m in a.modalidades)}</div>
        <div style="font-weight:600">{a.titulo}</div>
        <ul style="margin:2px 0 2px 16px;font-size:.85em">{just}</ul>
        <div style="font-size:.8em">➡️ {a.acao_sugerida}<br>👥 {', '.join(a.destinatarios)} — {a.prazo}{canais}{rep}</div>
        </div>""", unsafe_allow_html=True)


def aba_tempo_real():
    c, df = _cenario()
    if c is None:
        st.warning("Cenário não encontrado. Rode `python scripts/run_fusao.py`.")
        return
    ss = st.session_state
    if "monitor" not in ss:
        _reiniciar()
    t_max = float(df["t_s"].iloc[-1])

    col = st.columns([1.2, 1, 1, 1, 2.5])
    vel = col[0].select_slider("Velocidade", [1, 10, 30, 60, 120, 300], value=60, format_func=lambda v: f"{v}×")
    if col[1].button("▶️ Iniciar" if not ss.rodando else "⏸️ Pausar", width="stretch"):
        ss.rodando = not ss.rodando
    if col[2].button("⏮️ Reiniciar", width="stretch"):
        _reiniciar()
    # próximo evento de outra modalidade (consulta, fisioterapia, queda, prescrição)
    proximo = next((e.t for e in ss.pendentes if e.t > ss.t_sim + 60), ss.pendentes[0].t if ss.pendentes else None)
    if col[3].button("⏭️ Próximo evento", width="stretch", disabled=proximo is None,
                     help="Avança até logo depois do próximo evento de áudio, vídeo ou prescrição"):
        _avancar(min(proximo + 5, t_max))
    col[4].caption(f"**{c['paciente_id']}** · {c['descricao']}. Sinais vitais: {c['vitais']['fonte']}. "
                   + " · ".join(f"{ICONE[b['modalidade']]} {b['fonte']}" for b in c["blocos"]))

    @st.fragment(run_every=1.0)
    def painel():
        if ss.rodando and ss.t_sim < t_max:
            _avancar(min(ss.t_sim + vel, t_max))
        t = ss.t_sim
        esc = ss.motor.escore(c["paciente_id"], t)
        k = st.columns(6)
        atual = df[df["t_s"] <= t].tail(15)
        def val(cn, f="{:.0f}"):
            v = atual[cn].median() if len(atual) else np.nan
            return "–" if np.isnan(v) else f.format(v)
        k[0].metric("Tempo", f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}")
        k[1].metric("FC (bpm)", val("HR"))
        k[2].metric("PAM (mmHg)", val("MAP"))
        k[3].metric("SpO₂ (%)", val("SPO2"))
        k[4].metric("FR (irpm)", val("RR"))
        k[5].markdown(f"<div style='background:{NIVEL_COR[esc['nivel']]};color:white;border-radius:6px;padding:8px;"
                      f"text-align:center'><b>Risco {esc['nivel'].upper()}</b><br>escore {esc['escore']}<br>"
                      f"<span style='font-size:.8em'>{' '.join(ICONE.get(m, '') for m in esc['modalidades_ativas'])}</span></div>",
                      unsafe_allow_html=True)

        esq, dir_ = st.columns([2.2, 1])
        with esq:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            jan = df[(df["t_s"] <= t) & (df["t_s"] >= t - 1800)]
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04)
            for i, (cn, nome, lim) in enumerate([("HR", "FC", None), ("MAP", "PAM", 65), ("SPO2", "SpO₂", 92)], 1):
                fig.add_trace(go.Scatter(x=jan["t_s"] / 60, y=jan[cn], name=nome, line=dict(width=1.5)), row=i, col=1)
                if lim:
                    fig.add_hline(y=lim, line_dash="dot", line_color="#c0392b", row=i, col=1)
                fig.update_yaxes(title_text=nome, row=i, col=1)
            for e in ss.eventos:
                if t - 1800 <= e.t <= t and e.severidade >= 1:
                    fig.add_vline(x=e.t / 60, line_color=COR[int(e.severidade)], opacity=0.5)
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            fig.update_xaxes(title_text="minutos", row=3, col=1)
            st.plotly_chart(fig, width="stretch", key=f"vit")

            evs = pd.DataFrame([{"t_min": e.t / 60, "modalidade": e.origem, "sev": int(e.severidade),
                                 "descricao": e.descricao[:120]} for e in ss.eventos])
            if len(evs):
                import plotly.express as px
                f2 = px.scatter(evs, x="t_min", y="modalidade", color=evs["sev"].map(lambda s: Severidade(s).rotulo),
                                color_discrete_map={Severidade(s).rotulo: COR[s] for s in COR}, hover_data=["descricao"],
                                height=230)
                for a in ss.alertas:
                    if a.regra.startswith("fusao:"):
                        f2.add_vline(x=a.t / 60, line_dash="dash", line_color="#c0392b")
                f2.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="severidade")
                st.plotly_chart(f2, width="stretch", key="tl")
        with dir_:
            st.markdown(f"**Alertas** ({len(ss.alertas)})")
            caixa = st.container(height=640)
            with caixa:
                for a in sorted(ss.alertas, key=lambda a: (-a.t))[:40]:
                    _cartao_alerta(a, ss.entregas.get(a.id))
    painel()


# ======================================================================================
# Áudio
# ======================================================================================
def aba_audio():
    m = _json(RES / "audio" / "metricas.json", {})
    consultas = sorted((RES / "audio" / "consultas").glob("*.json"))
    if not consultas:
        st.info("Rode `python scripts/run_audio.py`.")
        return
    nomes = {p.stem: _json(p) for p in consultas}
    esc = st.selectbox("Consulta (PriMock57 — canal do paciente)", list(nomes),
                       format_func=lambda k: f"{k} — {nomes[k].get('queixa_principal', '')}")
    r = nomes[esc]
    a, b, c_ = st.columns(3)
    a.metric("Transcrição", r["stt"]["fonte"], r["stt"].get("provedor") or "")
    b.metric("Análise de texto", r["linguagem"]["fonte"], r["linguagem"].get("provedor") or "")
    c_.metric("Sentimento geral", r["linguagem"]["sentimento"])
    if r["stt"].get("wer_vs_manual") is not None:
        st.caption(f"WER do Azure Speech contra a transcrição manual do dataset: **{r['stt']['wer_vs_manual']:.1%}**")
    esq, dir_ = st.columns([1.6, 1])
    with esq:
        texto = r["stt"]["texto"]
        for e in sorted(r["linguagem"]["entidades"], key=lambda e: -len(e["texto"] or "")):
            if e["texto"] and e["categoria"] in ("SymptomOrSign", "MedicationName", "Diagnosis", "Dosage"):
                cor = "#95a5a6" if e["negada"] else {"SymptomOrSign": "#e67e22", "MedicationName": "#2980b9"}.get(e["categoria"], "#8e44ad")
                texto = texto.replace(e["texto"], f"<span style='background:{cor}33;border-bottom:2px solid {cor}'>{e['texto']}</span>", 1)
        st.markdown("**Transcrição** (laranja = sintoma, azul = medicamento, cinza = negado)")
        st.markdown(f"<div style='max-height:360px;overflow:auto;font-size:.9em'>{texto}</div>", unsafe_allow_html=True)
    with dir_:
        st.markdown("**Sinais de alarme**")
        for g, s in r["sinais_alarme"].items():
            st.markdown(f"<span style='color:{COR[s['severidade']]}'>●</span> **{g.replace('_', ' ')}** — "
                        f"{', '.join(sorted(set(s['mencoes']))[:4])}", unsafe_allow_html=True)
        if not r["sinais_alarme"]:
            st.caption("nenhum")
        st.markdown("**Modelos de voz**")
        for k, v in r["modelos_voz"].items():
            st.progress(min(v["prob"], 1.0), text=f"{k}: {v['prob']:.0%} (alerta ≥ {v['limiar']:.0%})")
        st.markdown("**Marcadores acústicos**")
        st.dataframe(pd.Series(r["acustica"]).round(3).rename("valor"), width="stretch", height=260)
    with st.expander("Desempenho dos modelos de voz (validação cruzada por falante)"):
        linhas = []
        if "disartria_torgo" in m:
            linhas.append({"modelo": "disartria (TORGO)", **{k: m["disartria_torgo"].get(k) for k in ("auroc", "sensibilidade", "especificidade", "n", "grupos")}})
        for k, v in m.get("coswara", {}).items():
            linhas.append({"modelo": f"{k} (Coswara)", **{kk: v.get(kk) for kk in ("auroc", "sensibilidade", "especificidade", "n", "grupos")}})
        st.dataframe(pd.DataFrame(linhas), width="stretch")
        if "consultas_primock" in m:
            st.json(m["consultas_primock"])


# ======================================================================================
# Vídeo
# ======================================================================================
def aba_video():
    m = _json(RES / "video" / "metricas.json", {})
    demos = sorted((RES / "video" / "demo").glob("*.mp4"))
    if demos:
        cols = st.columns(len(demos))
        for c_, d in zip(cols, demos):
            c_.markdown(f"**{d.stem}**")
            c_.video(str(d))
    reps = RES / "video" / "repeticoes.csv"
    if reps.exists():
        df = pd.read_csv(reps)
        st.markdown("**Repetições de fisioterapia** — escore = maior desvio (em desvios robustos) do padrão esperado")
        vid = st.selectbox("Sessão", sorted(df["video"].unique()))
        st.dataframe(df[df["video"] == vid].drop(columns=["arquivo"]).round(2), width="stretch", height=280)
        rel = RES / "video" / "relatorios" / f"{vid}.md"
        if rel.exists():
            with st.expander("Relatório automático da sessão"):
                st.markdown(rel.read_text(encoding="utf-8"))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fisioterapia — detecção de execução incorreta**")
        st.json(m.get("fisioterapia", {}), expanded=False)
        st.json(m.get("acuracia_pose", {}), expanded=False)
    with c2:
        st.markdown("**Quedas (UR Fall)**")
        st.json(m.get("quedas", {}), expanded=True)


# ======================================================================================
# Sinais vitais & prescrições
# ======================================================================================
def aba_vitais():
    m = _json(RES / "vitals" / "metricas.json", {})
    f = _json(RES / "fusao" / "metricas.json", {})
    c1, c2 = st.columns([1.4, 1])
    with c1:
        for img in sorted((RES / "vitals").glob("timeline_caso_*.png")):
            st.image(str(img))
    with c2:
        roc = RES / "vitals" / "roc_hipotensao.png"
        if roc.exists():
            st.image(str(roc))
        st.json(m.get("preditor_hipotensao", {}), expanded=False)
    st.markdown("**Antecipação em dados reais (30 cirurgias de teste)**")
    st.json(m.get("antecipacao_dados_reais", {}), expanded=False)
    if "injecao" in m:
        st.markdown("**Anomalias injetadas — taxa de detecção por camada**")
        st.dataframe(pd.DataFrame(m["injecao"]).T, width="stretch")
    st.markdown("**MIMIC-IV demo — fusão real de sinais vitais + prescrições do mesmo paciente**")
    if "mimic" in f:
        st.json({k: v for k, v in f["mimic"].items() if k != "exemplos"}, expanded=False)
        for ex in f["mimic"].get("exemplos", []):
            st.markdown(f"* **{ex['titulo']}** ({ex['severidade']}, {ex['paciente']}, {ex['t_h']} h): " + " | ".join(ex["justificativa"]))


# ======================================================================================
# Azure
# ======================================================================================
def aba_azure():
    st.json(CONFIG.resumo_azure())
    n = sum(1 for _ in CONFIG.azure_cache_dir.rglob("*.json")) if CONFIG.azure_cache_dir.exists() else 0
    st.metric("Respostas Azure em cache", n)
    st.markdown("""
**Trocar de conta Azure** (sem mexer em código):
1. `az login` na conta nova
2. `.\\scripts\\azure_setup.ps1` — cria Speech + Language e grava as chaves no `.env`
3. `python scripts/check_azure.py`

O cache é indexado pelo **conteúdo** (hash do áudio/texto), não pela conta: nada precisa ser reprocessado.
Com `AZURE_MODE=replay` a demo roda só com as respostas já obtidas.""")
    if st.button("Testar credenciais agora"):
        import subprocess
        r = subprocess.run([sys.executable, str(RAIZ / "scripts" / "check_azure.py")], capture_output=True, text=True)
        st.code(r.stdout + r.stderr)


st.title("🩺 Hospital Aurora — Monitoramento multimodal")
abas = st.tabs(["⏱️ Tempo real", "🎙️ Áudio", "🎥 Vídeo", "💓 Sinais vitais & prescrições", "☁️ Azure & sistema"])
with abas[0]:
    aba_tempo_real()
with abas[1]:
    aba_audio()
with abas[2]:
    aba_video()
with abas[3]:
    aba_vitais()
with abas[4]:
    aba_azure()
