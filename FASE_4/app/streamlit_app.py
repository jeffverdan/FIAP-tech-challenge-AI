"""Central de Alertas — Hospital Aurora (FASE 4).

    streamlit run app/streamlit_app.py
    # no Windows, abra http://127.0.0.1:8501 (não "localhost")

Abas:
  * **Central** — todos os leitos no mesmo relógio simulado, cada um com nível de risco, sinais
    vitais atuais, mini-gráficos e último alerta (ordenados pelo risco, como numa central de
    enfermagem). O Leito 12 é o cenário multimodal completo; os demais são cirurgias de teste do
    VitalDB, só com sinais vitais.
  * **Leito (detalhe)** — gráficos, linha do tempo multimodal e alertas do leito escolhido. Os
    sinais vitais passam pelo monitor *streaming* de verdade; consulta, fisioterapia, queda e
    prescrição chegam como eventos nos instantes do cenário; o motor de fusão decide os alertas.
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
# Tempo real — Central de leitos + detalhe por leito
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


def _carregar_vitais(arquivo: str) -> pd.DataFrame:
    df, _ = limpar(carregar_vitaldb(RAIZ / arquivo))
    ok = df[["HR", "MAP"]].notna().all(axis=1)
    df = df.loc[ok.idxmax(): ok[::-1].idxmax()].reset_index(drop=True)
    df["t_s"] -= df["t_s"].iloc[0]
    return df


@st.cache_data
def _cenario():
    c = _json(CONFIG.processed_dir / "cenario_aurora.json")
    if c is None:
        return None, None
    return c, _carregar_vitais(c["vitais"]["arquivo"])


@st.cache_data
def _leitos():
    """Configuração da Central. Sem ``central_leitos.json``, mostra só o Leito 12."""
    c, _ = _cenario()
    cfg = _json(CONFIG.processed_dir / "central_leitos.json")
    if not cfg:
        cfg = [{"leito": "Leito 12", "paciente_id": c["paciente_id"], "cenario": True,
                "fonte": "cenário multimodal composto", "arquivo": c["vitais"]["arquivo"]}]
    out = []
    for item in cfg:
        if (RAIZ / item["arquivo"]).exists():
            out.append({**item, "df": _carregar_vitais(item["arquivo"])})
    return out


def _reiniciar():
    from monitor.vitals.detectores import MonitorSinaisVitais

    c, _ = _cenario()
    ifo, pre = _modelos()
    ss = st.session_state
    ss.t_sim, ss.rodando = 0.0, False
    ss.motor = MotorFusao()
    ss.notificador = Notificador(sessao=f"replay-{int(time.time())}")
    ss.alertas, ss.entregas = [], {}
    ss.leitos = {}
    for L in _leitos():
        df = L["df"]
        pend = sorted((Evento.from_dict(e) for b in c["blocos"] for e in b["eventos"]), key=lambda e: e.t) \
            if L["cenario"] else []
        ss.leitos[L["paciente_id"]] = {
            "cfg": L, "monitor": MonitorSinaisVitais(L["paciente_id"], ifo, modelo_hipotensao=pre),
            "t": df["t_s"].to_numpy(), "valores": df[[k for k in df.columns if k != "t_s"]].to_numpy(),
            "colunas": [k for k in df.columns if k != "t_s"], "idx": 0, "pendentes": pend, "eventos": []}


def _avancar(ate: float):
    ss = st.session_state
    novos = []
    for pid, L in ss.leitos.items():
        fim = int(np.searchsorted(L["t"], ate, side="right"))
        for i in range(L["idx"], fim):
            t = float(L["t"][i])
            while L["pendentes"] and L["pendentes"][0].t <= t:
                novos.append(L["pendentes"].pop(0))
            novos += L["monitor"].atualizar(t, dict(zip(L["colunas"], L["valores"][i])))
        while L["pendentes"] and L["pendentes"][0].t <= ate:
            novos.append(L["pendentes"].pop(0))
        L["idx"] = fim
    ss.t_sim = ate
    for ev in sorted(novos, key=lambda e: e.t):
        ss.leitos[ev.paciente_id]["eventos"].append(ev)
        for a in ss.motor.processar(ev):
            ss.alertas.append(a)
            ss.entregas[a.id] = ss.notificador.entregar(a)


def _nome_leito(pid: str) -> str:
    L = st.session_state.leitos.get(pid)
    return L["cfg"]["leito"] if L else pid


def _cartao_alerta(a, entrega=None, com_leito: bool = False):
    cor = COR[int(a.severidade)]
    # alerta direto: a justificativa é o próprio evento (já está no título)
    just = "" if a.regra.startswith("direto:") else "".join(f"<li>{j}</li>" for j in a.justificativa)
    fusao = "🔗 FUSÃO MULTIMODAL · " if a.regra.startswith("fusao:") else ""
    leito = f"🛏️ {_nome_leito(a.paciente_id)} · " if com_leito else ""
    canais = f" · canais: {', '.join(entrega['canais'])}" if entrega else ""
    rep = f" · repetiu {a.repeticoes}×" if a.repeticoes > 1 else ""
    st.markdown(
        f"""<div style="border-left:6px solid {cor};padding:6px 10px;margin:6px 0;background:rgba(127,127,127,.07);border-radius:4px">
        <div style="font-size:.8em;color:{cor};font-weight:700">{leito}{fusao}{a.severidade.rotulo.upper()} · t = {a.t / 60:.1f} min · {' '.join(ICONE.get(m, '') for m in a.modalidades)}</div>
        <div style="font-weight:600">{a.titulo}</div>
        <ul style="margin:2px 0 2px 16px;font-size:.85em">{just}</ul>
        <div style="font-size:.8em">➡️ {a.acao_sugerida}<br>👥 {', '.join(a.destinatarios)} — {a.prazo}{canais}{rep}</div>
        </div>""", unsafe_allow_html=True)


def _controles():
    """Relógio único da simulação, na barra lateral (vale para a Central e para o detalhe)."""
    ss = st.session_state
    t_max = max(float(L["t"][-1]) for L in ss.leitos.values())
    with st.sidebar:
        st.markdown("### ⏱️ Simulação")
        st.select_slider("Velocidade", [1, 10, 30, 60, 120, 300], value=60, format_func=lambda v: f"{v}×", key="vel")
        c1, c2 = st.columns(2)
        if c1.button("▶️ Iniciar" if not ss.rodando else "⏸️ Pausar", width="stretch"):
            ss.rodando = not ss.rodando
        if c2.button("⏮️ Reiniciar", width="stretch"):
            _reiniciar()
        pend = [e for L in ss.leitos.values() for e in L["pendentes"]]
        proximo = min((e.t for e in pend if e.t > ss.t_sim + 60), default=min((e.t for e in pend), default=None))
        c3, c4 = st.columns(2)
        if c3.button("⏭️ Próx. evento", width="stretch", disabled=proximo is None,
                     help="Avança até logo depois do próximo evento de áudio, vídeo ou prescrição"):
            _avancar(min(proximo + 5, t_max))
        if c4.button("⏩ +5 min", width="stretch"):
            _avancar(min(ss.t_sim + 300, t_max))

        @st.fragment(run_every=1.0)
        def relogio():
            if ss.rodando and ss.t_sim < t_max:
                _avancar(min(ss.t_sim + st.session_state.vel, t_max))
            t = ss.t_sim
            st.metric("Tempo simulado", f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}")
            st.caption(f"{len(ss.leitos)} leitos · {len(ss.alertas)} alertas")
        relogio()


def _atual(L: dict, t: float) -> dict:
    i = int(np.searchsorted(L["t"], t, side="right"))
    jan = L["valores"][max(0, i - 15):i]
    if not len(jan):
        return {}
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # canal sem dado na janela → NaN
        med = np.nanmedian(jan, axis=0)
    return dict(zip(L["colunas"], med))


def _sparkline(L: dict, t: float, canal: str, lim: float, cor: str, janela_s: float = 1800) -> str:
    """Mini gráfico SVG dos últimos 30 min (linha pontilhada = limiar clínico)."""
    i = int(np.searchsorted(L["t"], t, side="right"))
    a = int(np.searchsorted(L["t"], t - janela_s))
    tt, vv = L["t"][a:i], L["valores"][a:i, L["colunas"].index(canal)]
    ok = ~np.isnan(vv)
    if ok.sum() < 3:
        return "<div style='height:34px'></div>"
    tt, vv = tt[ok], vv[ok]
    lo, hi = min(vv.min(), lim) - 3, max(vv.max(), lim) + 3
    w, h = 220, 34
    x = (tt - (t - janela_s)) / janela_s * w
    y = h - (vv - lo) / max(hi - lo, 1e-6) * h
    pts = " ".join(f"{a_:.1f},{b_:.1f}" for a_, b_ in zip(x, y))
    yl = h - (lim - lo) / max(hi - lo, 1e-6) * h
    return (f"<svg width='100%' height='{h}' viewBox='0 0 {w} {h}' preserveAspectRatio='none'>"
            f"<line x1='0' x2='{w}' y1='{yl:.1f}' y2='{yl:.1f}' stroke='#c0392b' stroke-dasharray='3,3' stroke-width='0.8'/>"
            f"<polyline points='{pts}' fill='none' stroke='{cor}' stroke-width='1.4'/></svg>")


def _cartao_leito(pid: str, L: dict, t: float) -> str:
    ss = st.session_state
    esc = ss.motor.escore(pid, t)
    cor = NIVEL_COR[esc["nivel"]]
    v = _atual(L, t)
    def fmt(c, lo=None, hi=None):
        x = v.get(c, np.nan)
        if x is None or np.isnan(x):
            return "–"
        ruim = (lo is not None and x < lo) or (hi is not None and x > hi)
        return f"<b style='color:{'#c0392b' if ruim else 'inherit'}'>{x:.0f}</b>"
    meus = [a for a in ss.alertas if a.paciente_id == pid]
    criticos = sum(int(a.severidade) == 3 for a in meus)
    ultimo = max(meus, key=lambda a: a.t) if meus else None
    ult = (f"<div style='font-size:.75em;border-left:3px solid {COR[int(ultimo.severidade)]};padding-left:5px;"
           f"height:2.4em;overflow:hidden'>{'🔗 ' if ultimo.regra.startswith('fusao:') else ''}{ultimo.titulo}</div>"
           if ultimo else "<div style='font-size:.75em;color:#7f8c8d;height:2.4em'>sem alertas</div>")
    fim = " · <i>fim do registro</i>" if t > L["t"][-1] else ""
    mods = " ".join(ICONE.get(m, "") for m in esc["modalidades_ativas"])
    return f"""<div style="border:2px solid {cor};border-radius:8px;padding:8px 10px;margin-bottom:10px;background:rgba(127,127,127,.04)">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <b style="font-size:1.05em">🛏️ {L['cfg']['leito']}</b>
        <span style="background:{cor};color:white;border-radius:4px;padding:1px 7px;font-size:.75em;font-weight:700">
          {esc['nivel'].upper()} · {esc['escore']:.0f}</span></div>
      <div style="font-size:.7em;color:#7f8c8d;height:1.3em;overflow:hidden">{L['cfg']['fonte']}{fim}</div>
      <div style="font-size:.85em;margin:4px 0">FC {fmt('HR', 45, 120)} · PAM {fmt('MAP', 65)} · SpO₂ {fmt('SPO2', 92)} · FR {fmt('RR', 8, 30)}</div>
      <div style="font-size:.65em;color:#7f8c8d">PAM (30 min)</div>{_sparkline(L, t, 'MAP', 65, '#2c3e50')}
      <div style="font-size:.65em;color:#7f8c8d">SpO₂ (30 min)</div>{_sparkline(L, t, 'SPO2', 92, '#2980b9')}
      <div style="font-size:.75em;margin:4px 0">🔔 {len(meus)} alertas · <b style="color:#c0392b">{criticos} críticos</b> {mods}</div>
      {ult}
    </div>"""


def aba_central():
    ss = st.session_state
    st.caption("Todos os leitos no mesmo relógio. A borda e o selo mostram o nível de risco do motor de fusão "
               "(verde → vermelho). O Leito 12 tem todas as modalidades; os demais, só sinais vitais.")
    ordenar = st.toggle("Ordenar por risco", value=True)

    @st.fragment(run_every=1.0)
    def grade():
        t = ss.t_sim
        pids = list(ss.leitos)
        if ordenar:
            pids.sort(key=lambda p: -ss.motor.escore(p, t)["escore"])
        n_col = 4
        for i in range(0, len(pids), n_col):
            cols = st.columns(n_col)
            for c, pid in zip(cols, pids[i:i + n_col]):
                c.markdown(_cartao_leito(pid, ss.leitos[pid], t), unsafe_allow_html=True)
        st.markdown(f"**Últimos alertas — todos os leitos** ({len(ss.alertas)})")
        recentes = sorted(ss.alertas, key=lambda a: (-int(a.severidade) if t - a.t < 600 else 0, -a.t))[:12]
        cols = st.columns(2)
        for k, a in enumerate(recentes):
            with cols[k % 2]:
                _cartao_alerta(a, ss.entregas.get(a.id), com_leito=True)
    grade()


def aba_tempo_real():
    ss = st.session_state
    rotulos = {f"{L['cfg']['leito']} — {L['cfg']['fonte']}": p for p, L in ss.leitos.items()}
    pid = rotulos[st.selectbox("Leito", list(rotulos), key="leito_sel")]
    L = ss.leitos[pid]
    if L["cfg"]["cenario"]:
        c, _ = _cenario()
        st.caption(f"**{c['paciente_id']}** · {c['descricao']}. Sinais vitais: {c['vitais']['fonte']}. "
                   + " · ".join(f"{ICONE[b['modalidade']]} {b['fonte']}" for b in c["blocos"]))

    @st.fragment(run_every=1.0)
    def painel():
        t = ss.t_sim
        esc = ss.motor.escore(pid, t)
        v = _atual(L, t)
        val = lambda cn: "–" if cn not in v or np.isnan(v[cn]) else f"{v[cn]:.0f}"
        k = st.columns(5)
        k[0].metric("FC (bpm)", val("HR"))
        k[1].metric("PAM (mmHg)", val("MAP"))
        k[2].metric("SpO₂ (%)", val("SPO2"))
        k[3].metric("FR (irpm)", val("RR"))
        k[4].markdown(f"<div style='background:{NIVEL_COR[esc['nivel']]};color:white;border-radius:6px;padding:8px;"
                      f"text-align:center'><b>Risco {esc['nivel'].upper()}</b><br>escore {esc['escore']}<br>"
                      f"<span style='font-size:.8em'>{' '.join(ICONE.get(m, '') for m in esc['modalidades_ativas'])}</span></div>",
                      unsafe_allow_html=True)

        esq, dir_ = st.columns([2.2, 1])
        with esq:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            i = int(np.searchsorted(L["t"], t, side="right"))
            a0 = int(np.searchsorted(L["t"], t - 1800))
            tt = L["t"][a0:i] / 60
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04)
            for r, (cn, nome, lim) in enumerate([("HR", "FC", None), ("MAP", "PAM", 65), ("SPO2", "SpO₂", 92)], 1):
                fig.add_trace(go.Scatter(x=tt, y=L["valores"][a0:i, L["colunas"].index(cn)], name=nome,
                                         line=dict(width=1.5)), row=r, col=1)
                if lim:
                    fig.add_hline(y=lim, line_dash="dot", line_color="#c0392b", row=r, col=1)
                fig.update_yaxes(title_text=nome, row=r, col=1)
            for e in L["eventos"]:
                if t - 1800 <= e.t <= t and e.severidade >= 1:
                    fig.add_vline(x=e.t / 60, line_color=COR[int(e.severidade)], opacity=0.5)
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
            fig.update_xaxes(title_text="minutos", row=3, col=1)
            st.plotly_chart(fig, width="stretch", key="vit")

            evs = pd.DataFrame([{"t_min": e.t / 60, "modalidade": e.origem, "sev": int(e.severidade),
                                 "descricao": e.descricao[:120]} for e in L["eventos"]])
            if len(evs):
                import plotly.express as px
                f2 = px.scatter(evs, x="t_min", y="modalidade", color=evs["sev"].map(lambda s: Severidade(s).rotulo),
                                color_discrete_map={Severidade(s).rotulo: COR[s] for s in COR}, hover_data=["descricao"],
                                height=230)
                for a in ss.alertas:
                    if a.paciente_id == pid and a.regra.startswith("fusao:"):
                        f2.add_vline(x=a.t / 60, line_dash="dash", line_color="#c0392b")
                f2.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title_text="severidade")
                st.plotly_chart(f2, width="stretch", key="tl")
        with dir_:
            meus = [a for a in ss.alertas if a.paciente_id == pid]
            st.markdown(f"**Alertas** ({len(meus)})")
            with st.container(height=640):
                for a in sorted(meus, key=lambda a: -a.t)[:40]:
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
    demos = {k: v for k, v in m.get("demos", {}).items() if "modelos_voz" in v}
    if demos:
        st.markdown("**Teste com voz alterada** — mesmo pipeline, áudios de pacientes reais com e sem a condição")
        nomes_demo = {"disartria_M05": "🗣️ Paciente com disartria (TORGO M05)",
                      "controle_MC01": "🙂 Controle sem disartria (TORGO MC01)",
                      "dispneia_coswara": "😮‍💨 Paciente com falta de ar (Coswara)",
                      "saudavel_coswara": "🙂 Participante saudável (Coswara)"}
        cols = st.columns(len(demos))
        for c_, (k, v) in zip(cols, demos.items()):
            with c_:
                st.markdown(f"**{nomes_demo.get(k, k)}**")
                mod = "disartria" if k.endswith(("M05", "MC01")) else "respiratorio"
                p_ = v["modelos_voz"].get(mod)
                if p_:
                    alerta = "🔴" if p_["prob"] >= p_["limiar"] else "⚪"
                    st.progress(min(p_["prob"], 1.0), text=f"{alerta} {mod}: {p_['prob']:.0%} (alerta ≥ {p_['limiar']:.0%})")
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
if _cenario()[0] is None:
    st.warning("Cenário não encontrado. Rode `python scripts/run_fusao.py`.")
    st.stop()
if "leitos" not in st.session_state:
    _reiniciar()
_controles()
abas = st.tabs(["🏥 Central", "⏱️ Leito (detalhe)", "🎙️ Áudio", "🎥 Vídeo", "💓 Sinais vitais & prescrições",
                "☁️ Azure & sistema"])
with abas[0]:
    aba_central()
with abas[1]:
    aba_tempo_real()
with abas[2]:
    aba_audio()
with abas[3]:
    aba_video()
with abas[4]:
    aba_vitais()
with abas[5]:
    aba_azure()
