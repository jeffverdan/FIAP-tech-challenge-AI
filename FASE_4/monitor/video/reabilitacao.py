"""Análise de sessões de fisioterapia: repetições, desvios do padrão esperado e relatório.

Fluxo por vídeo:  trilha de pose → série cinemática → repetições → features por repetição →
comparação com o **padrão esperado** (repetições corretas de *outros* pacientes, mesma vista
de câmera) → desvios explicados em linguagem clínica → eventos.

Exercícios suportados (REHAB24-6):
  * ``1`` abdução do braço direito — amplitude, cotovelo estendido, tronco e ombros parados;
  * ``6`` agachamento — profundidade (joelho/quadril), inclinação do tronco, simetria.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from monitor.events import Evento, Severidade


@dataclass(frozen=True)
class Exercicio:
    id: int
    nome: str
    sinal_primario: str          # coluna da série cinemática que oscila a cada repetição
    pico_e_maximo: bool          # repetição = pico (True) ou vale (False) do sinal primário
    features: dict               # feature → (descrição clínica, unidade, sentido "ruim": +1 maior, -1 menor, 0 ambos)
    sinal_segmentacao: str = ""  # sinal usado para contar repetições (padrão: o primário)


EXERCICIOS = {
    1: Exercicio(1, "abdução do braço direito", "abducao_ombro_d", True, {
        "amplitude_abducao": ("amplitude de abdução do ombro", "°", -1),
        "cotovelo_min": ("extensão do cotovelo no topo", "°", -1),
        "tronco_desvio": ("inclinação compensatória do tronco", "°", +1),
        "ombros_desvio": ("elevação/inclinação dos ombros", "°", +1),
        "braco_contralateral": ("movimento do braço esquerdo", "°", +1),
        "punho_altura_max": ("altura do punho acima do ombro (× tronco)", "", -1),
        "duracao_s": ("duração da repetição", "s", 0),
        "picos_extras": ("irregularidade do movimento (picos extras)", "", +1),
    }),
    # No agachamento os erros sugeridos pelo fisioterapeuta vão nos dois sentidos (raso demais,
    # fundo demais com o tronco projetado): por isso o desvio conta para os dois lados.
    # A contagem usa a altura do quadril — no plano frontal o ângulo do joelho em 2D é ruidoso.
    6: Exercicio(6, "agachamento", "joelho_medio", False, {
        "joelho_min": ("flexão máxima do joelho (ângulo mínimo)", "°", 0),
        "quadril_min": ("flexão máxima do quadril (ângulo mínimo)", "°", 0),
        "descida_quadril": ("profundidade (descida do quadril / tronco)", "", 0),
        "tronco_max": ("inclinação do tronco à frente", "°", +1),
        "assimetria_joelhos": ("assimetria entre joelhos", "°", +1),
        "duracao_s": ("duração da repetição", "s", 0),
        "picos_extras": ("irregularidade do movimento (picos extras)", "", +1),
    }, sinal_segmentacao="altura_quadril_norm"),
}


def preparar_serie(cin: pd.DataFrame) -> pd.DataFrame:
    cin = cin.copy()
    cin["joelho_medio"] = cin[["joelho_d", "joelho_e"]].mean(axis=1)
    return cin


def _picos_extras(s: np.ndarray, maximo: bool) -> int:
    from scipy.signal import find_peaks

    s = s[~np.isnan(s)]
    if len(s) < 5:
        return 0
    amp = np.nanmax(s) - np.nanmin(s)
    pk, _ = find_peaks(s if maximo else -s, prominence=max(amp * 0.15, 3))
    return max(len(pk) - 1, 0)


def features_repeticao(seg: pd.DataFrame, ex: Exercicio) -> dict[str, float]:
    f: dict[str, float] = {"duracao_s": float(seg["t"].iloc[-1] - seg["t"].iloc[0]) if len(seg) > 1 else 0.0}
    if len(seg) < 3:
        return f
    base = seg.iloc[: max(2, len(seg) // 10)]       # início da repetição = postura de referência
    prim = seg[ex.sinal_primario].to_numpy(dtype=float)
    f["picos_extras"] = float(_picos_extras(prim, ex.pico_e_maximo))
    if ex.id == 1:
        ab = seg["abducao_ombro_d"]
        f["amplitude_abducao"] = float(ab.max())
        topo = seg[ab >= ab.min() + 0.5 * (ab.max() - ab.min())]
        f["cotovelo_min"] = float(topo["cotovelo_d"].min()) if len(topo) else np.nan
        f["tronco_desvio"] = float((seg["inclinacao_tronco"] - base["inclinacao_tronco"].median()).abs().max())
        f["ombros_desvio"] = float((seg["inclinacao_ombros"] - base["inclinacao_ombros"].median()).abs().max())
        f["braco_contralateral"] = float(seg["abducao_ombro_e"].max() - seg["abducao_ombro_e"].min())
        f["punho_altura_max"] = float(seg["punho_acima_ombro_d"].max())
    elif ex.id == 6:
        f["joelho_min"] = float(seg["joelho_medio"].min())
        f["quadril_min"] = float(seg[["quadril_d", "quadril_e"]].mean(axis=1).min())
        tronco_px = float(base["tronco_px"].median()) or np.nan
        f["descida_quadril"] = float((seg["altura_quadril_norm"].max() - base["altura_quadril_norm"].median()) / tronco_px)
        f["tronco_max"] = float(seg["inclinacao_tronco"].max())
        fundo = seg[seg["joelho_medio"] <= seg["joelho_medio"].min() + 15]
        f["assimetria_joelhos"] = float((fundo["joelho_d"] - fundo["joelho_e"]).abs().median()) if len(fundo) else np.nan
    return f


def segmentar_repeticoes(cin: pd.DataFrame, ex: Exercicio, fracao_amplitude: float = 0.35,
                         min_intervalo_s: float = 1.2) -> list[tuple[float, float]]:
    """Segmentação automática: cada pico proeminente do sinal de segmentação é uma repetição,
    delimitada pelos vales vizinhos. A proeminência mínima é adaptativa (35% da amplitude
    p5–p95 do próprio vídeo), o que dispensa ajuste por câmera ou paciente."""
    from scipy.signal import find_peaks

    col = ex.sinal_segmentacao or ex.sinal_primario
    maximo = True if ex.sinal_segmentacao else ex.pico_e_maximo   # altura do quadril: pico = mais baixo
    s = cin[col].interpolate(limit_direction="both").to_numpy(dtype=float)
    if np.isnan(s).all():
        return []
    t = cin["t"].to_numpy()
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.1
    sinal = s if maximo else -s
    amp = float(np.nanpercentile(sinal, 95) - np.nanpercentile(sinal, 5))
    pk, _ = find_peaks(sinal, prominence=max(fracao_amplitude * amp, 1e-6), distance=max(1, int(min_intervalo_s / dt)))
    vales, _ = find_peaks(-sinal, distance=max(1, int(0.5 / dt)))
    reps = []
    for p in pk:
        antes = vales[vales < p]
        depois = vales[vales > p]
        a = antes[-1] if len(antes) else max(0, p - int(2 / dt))
        b = depois[0] if len(depois) else min(len(t) - 1, p + int(2 / dt))
        reps.append((float(t[a]), float(t[b])))
    return reps


# --------------------------------------------------------------------------------------
# Padrão esperado e desvios
# --------------------------------------------------------------------------------------
@dataclass
class PadraoEsperado:
    """Distribuição das features em repetições corretas (mediana e desvio robusto)."""
    exercicio: int
    mediana: dict[str, float]
    escala: dict[str, float]
    n: int

    @classmethod
    def de_repeticoes(cls, reps: pd.DataFrame, ex: Exercicio) -> "PadraoEsperado":
        med, esc = {}, {}
        for c in ex.features:
            v = reps[c].dropna().to_numpy(dtype=float)
            if len(v) == 0:
                continue
            med[c] = float(np.median(v))
            piso = {"duracao_s": 0.4, "picos_extras": 0.5, "punho_altura_max": 0.1,
                    "descida_quadril": 0.1}.get(c, 3.0)
            esc[c] = max(1.4826 * float(np.median(np.abs(v - med[c]))), piso)
        return cls(ex.id, med, esc, len(reps))

    def desvios(self, f: dict[str, float], ex: Exercicio, limiar_z: float = 2.5) -> list[dict]:
        out = []
        for c, (desc, unid, sentido) in ex.features.items():
            if c not in self.mediana or c not in f or f[c] is None or np.isnan(f[c]):
                continue
            z = (f[c] - self.mediana[c]) / self.escala[c]
            ruim = abs(z) >= limiar_z and (sentido == 0 or np.sign(z) == sentido)
            if ruim:
                out.append({"feature": c, "descricao": desc, "valor": round(f[c], 2), "esperado": round(self.mediana[c], 2),
                            "z": round(float(z), 2), "unidade": unid})
        return sorted(out, key=lambda d: -abs(d["z"]))

    def escore(self, f: dict[str, float], ex: Exercicio) -> float:
        """Maior desvio no sentido clinicamente ruim (0 se tudo dentro do padrão)."""
        zs = []
        for c, (_, _, sentido) in ex.features.items():
            if c in self.mediana and c in f and f[c] is not None and not np.isnan(f[c]):
                z = (f[c] - self.mediana[c]) / self.escala[c]
                zs.append(abs(z) if sentido == 0 else max(sentido * z, 0.0))
        return float(max(zs)) if zs else 0.0


def texto_desvio(d: dict) -> str:
    sinal = "acima" if d["z"] > 0 else "abaixo"
    u = d["unidade"]
    return f"{d['descricao']}: {d['valor']:g}{u} ({sinal} do esperado {d['esperado']:g}{u})"


def eventos_sessao(paciente_id: str, reps: pd.DataFrame, t0: float = 0.0, consecutivas_urgente: int = 3) -> list[Evento]:
    """Uma repetição fora do padrão → atenção; N seguidas → urgente (padrão persistente)."""
    eventos, seguidas = [], 0
    for r in reps.itertuples(index=False):
        if r.fora_padrao:
            seguidas += 1
            desv = r.desvios[:3]
            sev = Severidade.URGENTE if seguidas >= consecutivas_urgente else Severidade.ATENCAO
            eventos.append(Evento(
                paciente_id=paciente_id, origem="video", tipo="execucao_fora_do_padrao" if sev < 2 else "padrao_incorreto_persistente",
                severidade=sev, t=t0 + float(r.fim), duracao_s=float(r.fim - r.inicio), detector="pose",
                descricao=(f"Repetição {r.repeticao} fora do padrão esperado" +
                           (f" ({seguidas} seguidas)" if seguidas > 1 else "") + ": " +
                           "; ".join(texto_desvio(d) for d in desv)),
                evidencias={"repeticao": int(r.repeticao), "escore": round(float(r.escore), 2), "desvios": desv}))
        else:
            seguidas = 0
    return eventos
