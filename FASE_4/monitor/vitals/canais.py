"""Canais canônicos de sinais vitais, leitura dos datasets e limpeza de artefatos.

Canais: ``HR`` (bpm), ``SPO2`` (%), ``SBP``/``DBP``/``MAP`` (mmHg), ``RR`` (irpm), ``TEMP`` (°C).
Todos os detectores trabalham sobre um ``DataFrame`` com a coluna ``t_s`` (segundos desde o
início) e esses canais — de onde quer que os dados venham.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANAIS = ["HR", "SPO2", "SBP", "DBP", "MAP", "RR", "TEMP"]

UNIDADES = {"HR": "bpm", "SPO2": "%", "SBP": "mmHg", "DBP": "mmHg", "MAP": "mmHg", "RR": "irpm", "TEMP": "°C"}
NOMES = {"HR": "frequência cardíaca", "SPO2": "saturação de O₂", "SBP": "PA sistólica",
         "DBP": "PA diastólica", "MAP": "PA média", "RR": "frequência respiratória", "TEMP": "temperatura"}

#: Faixas fisiologicamente possíveis. Fora delas é artefato de sensor (cabo solto, flush da
#: linha arterial, oxímetro deslocado) — vira NaN, não alerta.
FAIXAS_FISIOLOGICAS = {
    "HR": (20, 250), "SPO2": (50, 100), "SBP": (30, 260), "DBP": (10, 180),
    "MAP": (20, 200), "RR": (3, 60), "TEMP": (30, 43),
}


def limpar(df: pd.DataFrame, max_gap_s: float = 30.0) -> tuple[pd.DataFrame, dict[str, float]]:
    """Remove artefatos e preenche lacunas curtas.

    Retorna o DataFrame limpo e a fração de amostras descartadas como artefato por canal.
    Lacunas maiores que ``max_gap_s`` continuam NaN — o detector não deve inventar dado.
    """
    df = df.copy()
    artefatos: dict[str, float] = {}
    dt = float(np.nanmedian(np.diff(df["t_s"].to_numpy()))) if len(df) > 1 else 1.0
    limite = max(1, int(round(max_gap_s / max(dt, 1e-6))))
    for c in CANAIS:
        if c not in df:
            df[c] = np.nan
            continue
        lo, hi = FAIXAS_FISIOLOGICAS[c]
        serie = pd.to_numeric(df[c], errors="coerce")
        presentes = serie.notna().sum()
        fora = (serie < lo) | (serie > hi)
        serie = serie.mask(fora)
        artefatos[c] = float(fora.sum() / presentes) if presentes else 0.0
        df[c] = serie.ffill(limit=limite)
    # PA sistólica < diastólica é fisicamente impossível → artefato nos dois
    inval = df["SBP"] < df["DBP"]
    df.loc[inval, ["SBP", "DBP"]] = np.nan
    return df[["t_s", *CANAIS]], artefatos


# --------------------------------------------------------------------------------------
# VitalDB — monitorização intraoperatória a cada 2 s
# --------------------------------------------------------------------------------------
def carregar_vitaldb(caminho: Path) -> pd.DataFrame:
    """Lê um ``case_XXXX.csv.gz`` baixado por ``scripts/download_data.py``.

    Usa a PA invasiva (linha arterial) e, onde ela falta, a não invasiva.
    """
    bruto = pd.read_csv(caminho)
    out = pd.DataFrame({"t_s": bruto["t_s"]})
    out["HR"] = bruto.get("HR")
    out["SPO2"] = bruto.get("PLETH_SPO2")
    for alvo, art, nibp in (("SBP", "ART_SBP", "NIBP_SBP"), ("DBP", "ART_DBP", "NIBP_DBP"),
                            ("MAP", "ART_MBP", "NIBP_MBP")):
        a = bruto.get(art, pd.Series(np.nan, index=bruto.index))
        n = bruto.get(nibp, pd.Series(np.nan, index=bruto.index))
        out[alvo] = a.where(a.notna(), n)
    out["RR"] = bruto.get("RR_CO2")
    out["TEMP"] = bruto.get("BT")
    return out


# --------------------------------------------------------------------------------------
# MIMIC-IV demo — sinais vitais horários de UTI (chartevents)
# --------------------------------------------------------------------------------------
MIMIC_ITENS = {
    220045: "HR", 220277: "SPO2", 220210: "RR",
    220179: "SBP", 220180: "DBP", 220181: "MAP",          # não invasiva
    220050: "SBP_ART", 220051: "DBP_ART", 220052: "MAP_ART",  # invasiva
    223762: "TEMP", 223761: "TEMP_F",
    226732: "O2_DEVICE", 220739: "GCS_EYE", 223900: "GCS_VERBAL", 223901: "GCS_MOTOR",
}


def carregar_mimic_vitais(caminho_chartevents: Path) -> pd.DataFrame:
    """Pivota ``chartevents`` em uma linha por (stay_id, charttime) com os canais canônicos.

    Colunas extras: ``O2`` (1 se em oxigênio suplementar) e ``ALERTA`` (1 se GCS < 15 — usado
    como proxy de nível de consciência no NEWS2).
    """
    ce = pd.read_csv(caminho_chartevents,
                     usecols=["subject_id", "stay_id", "charttime", "itemid", "value", "valuenum"])
    ce = ce[ce["itemid"].isin(MIMIC_ITENS)].copy()
    ce["canal"] = ce["itemid"].map(MIMIC_ITENS)

    o2 = ce[ce["canal"] == "O2_DEVICE"].copy()
    o2["O2"] = (~o2["value"].fillna("None").str.lower().isin({"none", "room air"})).astype(float)

    num = ce[ce["canal"] != "O2_DEVICE"]
    tab = num.pivot_table(index=["subject_id", "stay_id", "charttime"], columns="canal",
                          values="valuenum", aggfunc="mean").reset_index()
    if "TEMP_F" in tab:
        tab["TEMP"] = tab.get("TEMP", pd.Series(np.nan, index=tab.index)).fillna((tab["TEMP_F"] - 32) / 1.8)
    for c in ("SBP", "DBP", "MAP"):
        art = f"{c}_ART"
        if art in tab:
            tab[c] = tab.get(c, pd.Series(np.nan, index=tab.index)).fillna(tab[art])
    gcs = [c for c in ("GCS_EYE", "GCS_VERBAL", "GCS_MOTOR") if c in tab]
    if gcs:
        total = tab[gcs].sum(axis=1, min_count=3)
        tab["CONSC_ALTERADA"] = (total < 15).astype(float).where(total.notna())

    tab = tab.merge(o2[["stay_id", "charttime", "O2"]].drop_duplicates(["stay_id", "charttime"]),
                    on=["stay_id", "charttime"], how="left")
    tab["charttime"] = pd.to_datetime(tab["charttime"])
    tab = tab.sort_values(["stay_id", "charttime"])
    # o dispositivo de O2 é registrado esparsamente: vale até a próxima anotação
    tab["O2"] = tab.groupby("stay_id")["O2"].ffill().fillna(0.0)
    tab["CONSC_ALTERADA"] = tab.groupby("stay_id")["CONSC_ALTERADA"].ffill() if "CONSC_ALTERADA" in tab else 0.0
    tab["t_s"] = (tab["charttime"] - tab.groupby("stay_id")["charttime"].transform("min")).dt.total_seconds()
    for c in CANAIS:
        if c not in tab:
            tab[c] = np.nan
    return tab[["subject_id", "stay_id", "charttime", "t_s", *CANAIS, "O2", "CONSC_ALTERADA"]]
