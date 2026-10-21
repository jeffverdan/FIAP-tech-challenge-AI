"""Testes unitários da FASE 4 — rodam sem dados brutos, sem Azure e sem GPU (``pytest -q``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitor.events import Evento, EventStore, Severidade  # noqa: E402
from monitor.fusion.motor import MotorFusao  # noqa: E402
from monitor.vitals.detectores import (DetectorNEWS2, DetectorRegras, DetectorZScore,  # noqa: E402
                                       MonitorSinaisVitais)
from monitor.vitals.news2 import calcular_news2  # noqa: E402


# ---------------------------------------------------------------- eventos
def test_evento_roundtrip(tmp_path):
    e = Evento("P1", "vitais", "hipotensao", Severidade.URGENTE, 10.0, "x", "regra", {"v": np.float64(1.5)})
    st = EventStore(tmp_path / "e.jsonl")
    st.gravar([e])
    lido = st.ler()[0]
    assert lido.severidade == Severidade.URGENTE and lido.evidencias["v"] == 1.5


def test_evento_origem_invalida():
    with pytest.raises(ValueError):
        Evento("P1", "radar", "x", 1, 0.0, "x", "regra")


# ---------------------------------------------------------------- NEWS2
def test_news2_paciente_normal():
    r = calcular_news2(rr=16, spo2=98, sbp=120, hr=70, temp=36.8)
    assert r.total == 0 and r.risco == "baixo"


def test_news2_alto_risco():
    r = calcular_news2(rr=26, spo2=90, sbp=88, hr=135, temp=39.5, em_oxigenio=True)
    assert r.total >= 7 and r.risco == "alto"


def test_news2_parametro_extremo_isolado():
    r = calcular_news2(rr=16, spo2=98, sbp=120, hr=70, temp=36.8, consciencia_alterada=True)
    assert r.total == 3 and r.risco == "baixo-médio"


# ---------------------------------------------------------------- sinais vitais
def _serie(n=900, dt=2.0, **mudancas):
    t = np.arange(n) * dt
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"t_s": t, "HR": 72 + rng.normal(0, 1, n), "SPO2": 98 + rng.normal(0, 0.3, n),
                       "SBP": 120 + rng.normal(0, 2, n), "DBP": 70 + rng.normal(0, 1.5, n),
                       "MAP": 85 + rng.normal(0, 1.5, n), "RR": 14 + rng.normal(0, 0.5, n), "TEMP": 36.8})
    for canal, (inicio, valor) in mudancas.items():
        df.loc[df["t_s"] >= inicio, canal] = valor
    return df


def test_regra_exige_sustentacao():
    det = DetectorRegras("P")
    assert det.atualizar(0, {"MAP": 60}) == []
    assert det.atualizar(30, {"MAP": 60}) == []          # 30 s < 60 s exigidos
    ev = det.atualizar(62, {"MAP": 58})
    assert len(ev) == 1 and ev[0].tipo == "hipotensao"
    assert det.atualizar(64, {"MAP": 58}) == []           # um alerta por episódio


def test_artefato_curto_nao_alerta():
    det = DetectorRegras("P")
    eventos = []
    for i, v in enumerate([90, 90, 50, 90, 90]):           # 1 amostra baixa (artefato)
        eventos += det.atualizar(i * 2, {"MAP": v})
    assert eventos == []


def test_zscore_detecta_mudanca_abrupta():
    df = _serie(HR=(1400, 110))
    mon = MonitorSinaisVitais("P")
    ev = [e for e in mon.processar_df(df) if e.detector == "estatistico"]
    assert any(e.tipo == "desvio_hr" and e.t >= 1400 for e in ev)


def test_serie_estavel_sem_alertas():
    mon = MonitorSinaisVitais("P")
    assert mon.processar_df(_serie()) == []


def test_news2_detector_escalonamento_com_histerese():
    det = DetectorNEWS2("P")
    ok = {"RR": 16, "SPO2": 98, "SBP": 120, "HR": 70, "TEMP": 36.8}
    ruim = {"RR": 26, "SPO2": 90, "SBP": 88, "HR": 135, "TEMP": 39.5}
    assert det.atualizar(0, ok) == []
    assert det.atualizar(3600, ruim)[0].severidade == Severidade.CRITICO
    det.atualizar(7200, ok)
    assert all(e.tipo != "news2_alto" for e in det.atualizar(10800, ruim))   # < 6 h: não repete


# ---------------------------------------------------------------- prescrições
def test_matriz_salto_de_dose():
    from monitor.prescricoes import severidade_salto

    assert severidade_salto(2.0, alta=False) is None
    assert severidade_salto(3.0, alta=False) == Severidade.ATENCAO
    assert severidade_salto(2.0, alta=True) == Severidade.ATENCAO
    assert severidade_salto(5.0, alta=True) == Severidade.CRITICO
    assert severidade_salto(10.0, alta=False) == Severidade.URGENTE
    assert severidade_salto(0.25, alta=True) == Severidade.ATENCAO
    assert severidade_salto(0.25, alta=False) is None


def test_deteccao_salto_em_tabela():
    from monitor import prescricoes as P

    pres = pd.DataFrame({
        "subject_id": 1, "hadm_id": 10, "drug_type": "MAIN", "drug": ["Morphine Sulfate"] * 2,
        "dose_val_rx": ["2", "10"], "dose_unit_rx": "mg", "doses_per_24_hrs": [4, 4], "route": "IV",
        "starttime": ["2100-01-01 08:00", "2100-01-02 08:00"], "stoptime": [None, None]})
    adm = pd.DataFrame({"hadm_id": [10], "admittime": ["2100-01-01 00:00"]})
    ev = P.detectar_saltos(P.preparar(pres, adm))
    assert len(ev) == 1 and ev[0].severidade == Severidade.CRITICO and ev[0].t == 32 * 3600


# ---------------------------------------------------------------- fusão
def _ev(origem, tipo, sev, t, pid="P"):
    return Evento(pid, origem, tipo, sev, t, f"{origem}:{tipo}", "teste")


def test_fusao_respiratoria_multifonte():
    m = MotorFusao()
    m.processar(_ev("audio", "sintoma_relatado_dispneia", 2, 0))
    alertas = m.processar(_ev("vitais", "dessaturacao", 2, 600))
    assert any(a.regra == "fusao:respiratoria_multifonte" and a.severidade == Severidade.CRITICO for a in alertas)


def test_fusao_exige_janela_temporal():
    m = MotorFusao()
    m.processar(_ev("audio", "sintoma_relatado_dispneia", 2, 0))
    alertas = m.processar(_ev("vitais", "dessaturacao", 2, 4 * 3600))
    assert not any(a.regra.startswith("fusao:") for a in alertas)


def test_fusao_nao_mistura_pacientes():
    m = MotorFusao()
    m.processar(_ev("audio", "sintoma_relatado_dispneia", 2, 0, pid="A"))
    alertas = m.processar(_ev("vitais", "dessaturacao", 2, 60, pid="B"))
    assert not any(a.regra.startswith("fusao:") for a in alertas)


def test_supressao_de_repeticao():
    m = MotorFusao(janela_supressao_s=900)
    assert len(m.processar(_ev("vitais", "taquicardia", 2, 0))) == 1
    assert m.processar(_ev("vitais", "taquicardia", 2, 300)) == []
    assert m.alertas[0].repeticoes == 2
    assert len(m.processar(_ev("vitais", "taquicardia", 2, 1000))) == 1


def test_info_nao_vira_alerta_mas_entra_no_escore():
    m = MotorFusao()
    assert m.processar(_ev("movimentacao", "deitou_no_leito", 0, 0)) == []
    assert m.escore("P", 10)["nivel"] == "verde"
    m.processar(_ev("vitais", "hipotensao_grave", 3, 20))
    m.processar(_ev("movimentacao", "queda", 2, 30))
    e = m.escore("P", 40)
    assert e["nivel"] == "vermelho" and set(e["modalidades_ativas"]) == {"vitais", "movimentacao"}


# ---------------------------------------------------------------- Azure: cache e modos
def test_cache_modos(tmp_path, monkeypatch):
    from monitor.azure import cache
    from monitor.config import CONFIG

    monkeypatch.setattr(CONFIG, "azure_cache_dir", tmp_path)
    chamadas = []
    vivo = lambda: (chamadas.append(1), {"x": 1})[1]
    fb = lambda: {"x": 0}
    monkeypatch.setattr(CONFIG, "azure_mode", "offline")
    assert cache.executar("s", "k", vivo, True, fb)["fonte"] == "fallback_local"
    monkeypatch.setattr(CONFIG, "azure_mode", "auto")
    assert cache.executar("s", "k", vivo, True, fb)["fonte"] == "azure"
    assert cache.executar("s", "k", vivo, True, fb)["fonte"] == "azure_cache"
    assert len(chamadas) == 1
    monkeypatch.setattr(CONFIG, "azure_mode", "replay")
    assert cache.executar("s", "k", vivo, False, fb)["x"] == 1
    with pytest.raises(cache.AzureIndisponivel):
        cache.executar("s", "outra", vivo, True, None)


def test_cache_nao_depende_da_conta():
    from monitor.azure import cache

    assert cache.chave("speech", "stt", "abc", idioma="en") == cache.chave("speech", "stt", "abc", idioma="en")
    assert cache.chave("speech", "stt", "abc", idioma="en") != cache.chave("speech", "stt", "abd", idioma="en")


def test_fallback_linguagem_com_negacao():
    from monitor.azure.language import _fallback_analisar
    from monitor.audio.consulta import sinais_de_alarme

    r = _fallback_analisar("I have chest pain since this morning. I don't have any shortness of breath.", "en")
    achados = sinais_de_alarme(r["entidades_saude"])
    assert "dor_toracica" in achados and "dispneia" not in achados


# ---------------------------------------------------------------- vídeo
def _kps_em_pe(x=100.0, y_quadril=200.0, tronco=60.0, deitado=False):
    k = np.zeros((17, 3))
    k[:, 2] = 0.9
    if deitado:
        k[[5, 6], :2] = [[x + tronco, y_quadril], [x + tronco, y_quadril + 2]]
        k[[11, 12], :2] = [[x, y_quadril], [x, y_quadril + 2]]
    else:
        k[[5, 6], :2] = [[x - 10, y_quadril - tronco], [x + 10, y_quadril - tronco]]
        k[[11, 12], :2] = [[x - 8, y_quadril], [x + 8, y_quadril]]
    return k


def test_queda_e_imobilidade():
    from monitor.video.quedas import DetectorQuedas

    det = DetectorQuedas("P")
    ev, t = [], 0.0
    for _ in range(20):                                            # em pé
        ev += det.atualizar(t, _kps_em_pe(), np.array([80, 130, 120, 210]))
        t += 1 / 15
    for i in range(6):                                             # cai: quadril desce rápido
        ev += det.atualizar(t, _kps_em_pe(y_quadril=200 + 25 * (i + 1)), np.array([80, 130, 120, 210]))
        t += 1 / 15
    for _ in range(80):                                            # deitado e parado
        ev += det.atualizar(t, _kps_em_pe(y_quadril=350, deitado=True), np.array([60, 330, 180, 360]))
        t += 1 / 15
    tipos = [e.tipo for e in ev]
    assert "queda" in tipos and "imobilidade_pos_queda" in tipos


def test_deitar_devagar_nao_e_queda():
    from monitor.video.quedas import DetectorQuedas

    det = DetectorQuedas("P")
    ev, t = [], 0.0
    for i in range(60):                                            # desce em 4 s
        ev += det.atualizar(t, _kps_em_pe(y_quadril=200 + i * 2), np.array([80, 130, 120, 210]))
        t += 1 / 15
    for _ in range(40):
        ev += det.atualizar(t, _kps_em_pe(y_quadril=320, deitado=True), np.array([60, 300, 180, 330]))
        t += 1 / 15
    assert not any(e.tipo == "queda" for e in ev)


def test_angulos_cinematica():
    from monitor.video.cinematica import angulo

    a = angulo(np.array([[0.0, 1.0]]), np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))
    assert abs(a[0] - 90) < 1e-6


def test_padrao_esperado_explica_desvio():
    from monitor.video.reabilitacao import EXERCICIOS, PadraoEsperado

    ex = EXERCICIOS[1]
    ref = pd.DataFrame({"amplitude_abducao": [150, 155, 148, 152, 151], "cotovelo_min": [170, 168, 172, 171, 169],
                        "tronco_desvio": [3, 4, 2, 3, 5], "ombros_desvio": [2, 3, 2, 2, 3],
                        "braco_contralateral": [5, 6, 4, 5, 5], "punho_altura_max": [0.8, 0.9, 0.85, 0.8, 0.9],
                        "duracao_s": [3, 3.2, 2.9, 3.1, 3], "picos_extras": [0, 0, 0, 0, 0]})
    p = PadraoEsperado.de_repeticoes(ref, ex)
    ruim = {**ref.iloc[0].to_dict(), "amplitude_abducao": 95.0, "tronco_desvio": 20.0}
    nomes = [d["feature"] for d in p.desvios(ruim, ex)]
    assert "amplitude_abducao" in nomes and "tronco_desvio" in nomes
    assert p.escore(ref.iloc[1].to_dict(), ex) < 2.5


def test_fusao_ordenada_causa_antes_do_efeito():
    def rx(t):
        return Evento("P", "prescricao", "salto_dose_aumento", 1, t, "metoprolol", "regra", {"droga": "Metoprolol Tartrate"})
    m = MotorFusao()
    m.processar(_ev("vitais", "hipotensao", 2, 0))
    assert not any(a.regra.startswith("fusao:") for a in m.processar(rx(600)))      # efeito antes da causa
    m2 = MotorFusao()
    m2.processar(rx(0))
    alertas = m2.processar(_ev("vitais", "hipotensao", 2, 3600))
    assert any(a.regra == "fusao:hemodinamica_medicamentosa" for a in alertas)


def test_fusao_filtra_classe_do_farmaco():
    m = MotorFusao()
    m.processar(Evento("P", "prescricao", "salto_dose_aumento", 1, 0, "vitamina D", "regra", {"droga": "Vitamin D"}))
    assert not any(a.regra.startswith("fusao:") for a in m.processar(_ev("vitais", "hipotensao", 2, 600)))
