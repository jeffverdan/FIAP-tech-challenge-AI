"""Detecção de alterações inesperadas na evolução das prescrições (MIMIC-IV demo).

Camadas, como nos sinais vitais:
  * **regra** — salto de dose entre prescrições consecutivas do mesmo fármaco/via;
    duplicidade terapêutica perigosa; associação opioide + benzodiazepínico.
  * **estatístico** — dose diária atípica em relação à distribuição populacional do mesmo
    fármaco/unidade/via (z robusto em escala log), típico de erro de digitação (10× a dose).
  * **regra de dinâmica** — instabilidade: ≥ 4 mudanças de dose do mesmo fármaco em 48 h.

Medicamentos de alta vigilância (lista ISMP) sobem um nível de severidade.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from monitor.events import Evento, Severidade

#: Fármacos de alta vigilância (ISMP High-Alert Medications, recorte hospitalar).
ALTA_VIGILANCIA = re.compile(
    r"insulin|heparin|enoxaparin|warfarin|apixaban|rivaroxaban|morphine|hydromorphone|fentanyl|"
    r"oxycodone|methadone|potassium chloride|digoxin|amiodarone|norepinephrine|epinephrine|"
    r"vasopressin|phenylephrine|propofol|midazolam|magnesium sulfate", re.I)

CLASSES = {
    "opioide": r"morphine|hydromorphone|fentanyl|oxycodone|methadone|tramadol|hydrocodone",
    "benzodiazepinico": r"lorazepam|midazolam|diazepam|alprazolam|clonazepam",
    "anticoagulante": r"^heparin$|heparin sodium|enoxaparin|apixaban|rivaroxaban|dabigatran|fondaparinux",
    "betabloqueador": r"metoprolol|carvedilol|atenolol|labetalol|propranolol|esmolol",
}

#: Itens que não são terapêutica (diluentes, flush, soros de manutenção).
NAO_TERAPEUTICO = re.compile(r"flush|dwell|sodium chloride|dextrose|sterile water|^bag$|lactated ringer|"
                             r"syringe|iso-osmotic|vial", re.I)
#: Protocolos *reativos por desenho* (reposição de eletrólitos guiada por exame, insulina por
#: escala): a dose muda sempre, e isso é o esperado — não entram em salto/instabilidade.
PROTOCOLO_REATIVO = re.compile(r"replacement|sliding|scale|^insulin$|potassium|magnesium|phosph|calcium", re.I)


def classe(droga: str) -> str | None:
    for nome, pad in CLASSES.items():
        if re.search(pad, droga, re.I):
            return nome
    return None


def _dose(valor) -> float:
    """'1-2' → 2 (pior caso); '0.5' → 0.5; inválido → NaN."""
    if pd.isna(valor):
        return np.nan
    nums = re.findall(r"\d+(?:\.\d+)?", str(valor))
    return max(float(n) for n in nums) if nums else np.nan


def preparar(prescricoes: pd.DataFrame, admissoes: pd.DataFrame) -> pd.DataFrame:
    p = prescricoes[prescricoes["drug_type"] == "MAIN"].copy()
    p = p[~p["drug"].fillna("").str.contains(NAO_TERAPEUTICO)]
    p["droga"] = p["drug"].str.strip().str.lower()
    p["principio"] = p["droga"].str.extract(r"([a-z]+)", expand=False)
    p["dose"] = p["dose_val_rx"].map(_dose)
    # frequência ausente → compara dose por administração (não inventa frequência)
    p["freq_conhecida"] = p["doses_per_24_hrs"] > 0
    p["dose_dia"] = np.where(p["freq_conhecida"], p["dose"] * p["doses_per_24_hrs"], p["dose"])
    p = p[p["dose_dia"] > 0]
    p["reativo"] = p["droga"].str.contains(PROTOCOLO_REATIVO)
    p["starttime"] = pd.to_datetime(p["starttime"])
    p["stoptime"] = pd.to_datetime(p["stoptime"])
    p = p.merge(admissoes[["hadm_id", "admittime"]], on="hadm_id", how="left")
    p["admittime"] = pd.to_datetime(p["admittime"])
    p["t"] = (p["starttime"] - p["admittime"]).dt.total_seconds()
    p["alta_vigilancia"] = p["droga"].str.contains(ALTA_VIGILANCIA)
    p["classe"] = p["droga"].map(classe)
    return p.sort_values(["hadm_id", "starttime"]).reset_index(drop=True)


def _pid(hadm_id) -> str:
    return f"MIMIC-H{int(hadm_id)}"


def _sobe(sev: Severidade, alta: bool) -> Severidade:
    return Severidade(min(int(sev) + (1 if alta else 0), 3))


def severidade_salto(r: float, alta: bool) -> Severidade | None:
    """Matriz razão × alta vigilância. Reduções só importam em alta vigilância."""
    if r >= 10:
        return Severidade.URGENTE if not alta else Severidade.CRITICO     # provável erro de digitação
    if r > 1:
        if alta:
            return Severidade.CRITICO if r >= 5 else Severidade.URGENTE if r >= 3 else \
                Severidade.ATENCAO if r >= 2 else None
        return Severidade.ATENCAO if r >= 3 else None
    if alta and r <= 1 / 3:
        return Severidade.ATENCAO
    return None


def detectar_saltos(p: pd.DataFrame) -> list[Evento]:
    eventos = []
    q = p[~p["reativo"]]
    for (hadm, droga, via, unid, freq), g in q.groupby(["hadm_id", "droga", "route", "dose_unit_rx", "freq_conhecida"],
                                                       dropna=False):
        if len(g) < 2:
            continue
        ant = g.iloc[0]
        for _, atual in g.iloc[1:].iterrows():
            r = atual["dose_dia"] / ant["dose_dia"] if ant["dose_dia"] else np.nan
            sev = severidade_salto(r, bool(atual["alta_vigilancia"])) if np.isfinite(r) else None
            if sev is not None:
                direcao = "aumento" if r > 1 else "redução"
                por = "/dia" if freq else "/dose"
                eventos.append(Evento(
                    paciente_id=_pid(hadm), origem="prescricao", tipo=f"salto_dose_{'aumento' if r > 1 else 'reducao'}",
                    severidade=sev, t=float(atual["t"]), detector="regra",
                    descricao=(f"{direcao.capitalize()} de {r:.1f}× na dose de {atual['drug']} ({via}): "
                               f"{ant['dose_dia']:g} → {atual['dose_dia']:g} {unid}{por}"
                               + (" — medicamento de alta vigilância" if atual["alta_vigilancia"] else "")),
                    evidencias={"droga": atual["drug"], "via": via, "unidade": unid, "razao": round(float(r), 2),
                                "dose_anterior": float(ant["dose_dia"]), "dose_nova": float(atual["dose_dia"]),
                                "base": por, "alta_vigilancia": bool(atual["alta_vigilancia"]),
                                "classe": atual["classe"]},
                ))
            ant = atual
    return eventos


def detectar_doses_atipicas(p: pd.DataFrame, z_limiar: float = 4.0, min_prescricoes: int = 15) -> list[Evento]:
    """Dose diária fora da distribuição populacional do mesmo fármaco/unidade/via."""
    eventos = []
    for (droga, via, unid), g in p[~p["reativo"]].groupby(["droga", "route", "dose_unit_rx"], dropna=False):
        if len(g) < min_prescricoes:
            continue
        lg = np.log10(g["dose_dia"])
        med = lg.median()
        mad = 1.4826 * (lg - med).abs().median()
        z = (lg - med) / max(mad, 0.15)          # piso: ~1,4× de variação já é "normal"
        for idx in g.index[z.abs() >= z_limiar]:
            linha = p.loc[idx]
            sev = _sobe(Severidade.ATENCAO, bool(linha["alta_vigilancia"]) and z[idx] > 0)
            eventos.append(Evento(
                paciente_id=_pid(linha["hadm_id"]), origem="prescricao", tipo="dose_atipica",
                severidade=sev, t=float(linha["t"]), detector="estatistico",
                descricao=(f"Dose diária de {linha['drug']} ({linha['dose_dia']:g} {unid}/dia) é "
                           f"{10 ** (lg[idx] - med):.1f}× a mediana hospitalar ({10 ** med:g}) — possível erro de dose"),
                evidencias={"droga": linha["drug"], "dose_dia": float(linha["dose_dia"]), "mediana_pop": float(10 ** med),
                            "z_log": round(float(z[idx]), 2), "n_referencia": int(len(g))},
            ))
    return eventos


def detectar_instabilidade(p: pd.DataFrame, janela_h: float = 48, min_mudancas: int = 4) -> list[Evento]:
    eventos = []
    for (hadm, droga), g in p[~p["reativo"]].groupby(["hadm_id", "droga"]):
        g = g.sort_values("t")
        mudou = g["dose_dia"].ne(g["dose_dia"].shift()) & g["dose_dia"].shift().notna()
        tm = g.loc[mudou, "t"].to_numpy()
        ultimo = -1e18
        for i in range(len(tm)):
            n = int(((tm >= tm[i] - janela_h * 3600) & (tm <= tm[i])).sum())
            if n >= min_mudancas and tm[i] - ultimo > janela_h * 3600:
                ultimo = tm[i]
                linha = g[g["t"] == tm[i]].iloc[0]
                eventos.append(Evento(
                    paciente_id=_pid(hadm), origem="prescricao", tipo="instabilidade_tratamento",
                    severidade=_sobe(Severidade.ATENCAO, bool(linha["alta_vigilancia"])), t=float(tm[i]),
                    detector="regra",
                    descricao=f"{n} mudanças de dose de {linha['drug']} em {janela_h:.0f} h — tratamento instável",
                    evidencias={"droga": linha["drug"], "mudancas": n, "janela_h": janela_h}))
    return eventos


def _sobrepoe(a: pd.Series, b: pd.Series, minimo_h: float = 6) -> bool:
    fim_a = a["stoptime"] if pd.notna(a["stoptime"]) else a["starttime"] + pd.Timedelta(hours=24)
    fim_b = b["stoptime"] if pd.notna(b["stoptime"]) else b["starttime"] + pd.Timedelta(hours=24)
    sob = (min(fim_a, fim_b) - max(a["starttime"], b["starttime"])).total_seconds() / 3600
    return sob >= minimo_h


def detectar_combinacoes(p: pd.DataFrame) -> list[Evento]:
    """Duplicidade de anticoagulante e opioide + benzodiazepínico ativos ao mesmo tempo."""
    eventos = []
    for hadm, g in p[p["classe"].notna()].groupby("hadm_id"):
        vistos: set = set()
        linhas = list(g.itertuples(index=False))
        for i, a in enumerate(linhas):
            for b in linhas[i + 1:]:
                if (b.starttime - a.starttime).total_seconds() > 72 * 3600:
                    break
                par = None
                if a.classe == b.classe == "anticoagulante" and a.principio != b.principio:
                    par = ("duplicidade_anticoagulante", Severidade.URGENTE,
                           f"Dois anticoagulantes ativos: {a.drug} + {b.drug} — risco de sangramento")
                elif {a.classe, b.classe} == {"opioide", "benzodiazepinico"}:
                    par = ("opioide_benzodiazepinico", Severidade.ATENCAO,
                           f"Opioide + benzodiazepínico ativos: {a.drug} + {b.drug} — risco de depressão respiratória")
                if par and par[0] not in vistos and _sobrepoe(pd.Series(a._asdict()), pd.Series(b._asdict())):
                    vistos.add(par[0])
                    eventos.append(Evento(
                        paciente_id=_pid(hadm), origem="prescricao", tipo=par[0], severidade=par[1],
                        t=float(max(a.t, b.t)), detector="regra", descricao=par[2],
                        evidencias={"drogas": [a.drug, b.drug], "classes": [a.classe, b.classe]}))
    return eventos


def analisar(dir_mimic: Path) -> tuple[pd.DataFrame, list[Evento]]:
    pres = pd.read_csv(Path(dir_mimic) / "hosp" / "prescriptions.csv.gz")
    adm = pd.read_csv(Path(dir_mimic) / "hosp" / "admissions.csv.gz")
    p = preparar(pres, adm)
    eventos = detectar_saltos(p) + detectar_doses_atipicas(p) + detectar_instabilidade(p) + detectar_combinacoes(p)
    return p, sorted(eventos, key=lambda e: (e.paciente_id, e.t))
