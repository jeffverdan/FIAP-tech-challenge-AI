"""Fusão multimodal e decisão de alerta — 100% determinística e auditável.

Entrada: :class:`~monitor.events.Evento` de qualquer modalidade, em ordem de tempo.
Saída: :class:`Alerta` para a equipe, cada um com justificativa (as evidências que o originaram),
destinatários e ação sugerida.

Três mecanismos:

1. **Alerta direto** — todo evento ≥ ATENÇÃO vira alerta, com supressão de repetição (mesmo
   paciente + mesmo tipo em ``janela_supressao_s`` só incrementa um contador). É o que evita
   fadiga de alarme sem esconder nada: o evento suprimido continua na linha do tempo.
2. **Regras de correlação entre modalidades** — evidências fracas de fontes *independentes*
   que apontam para o mesmo problema clínico se reforçam (ex.: SpO₂ caindo no monitor + paciente
   dizendo que está sem ar na consulta → insuficiência respiratória provável, crítico).
3. **Escore de risco do paciente** — soma das severidades com decaimento exponencial
   (meia-vida de 30 min), peso menor para vídeo de fisioterapia e sentimento, e bônus por número
   de modalidades distintas ativas: o painel ordena os leitos por esse escore.
"""

from __future__ import annotations

import math
import re
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from monitor.events import Evento, Severidade

DESTINATARIOS = {
    Severidade.ATENCAO: ["enfermagem do leito"],
    Severidade.URGENTE: ["enfermagem do leito", "médico plantonista"],
    Severidade.CRITICO: ["enfermagem do leito", "médico plantonista", "time de resposta rápida"],
}
PRAZO_RESPOSTA = {Severidade.ATENCAO: "avaliar em até 60 min", Severidade.URGENTE: "avaliar em até 15 min",
                  Severidade.CRITICO: "beira-leito imediato"}


@dataclass
class Alerta:
    paciente_id: str
    severidade: Severidade
    t: float
    titulo: str
    justificativa: list[str]
    modalidades: list[str]
    eventos: list[str]
    regra: str
    acao_sugerida: str
    destinatarios: list[str] = field(default_factory=list)
    prazo: str = ""
    repeticoes: int = 1
    id: str = field(default_factory=lambda: f"ALR-{uuid.uuid4().hex[:8].upper()}")

    def __post_init__(self) -> None:
        self.severidade = Severidade(int(self.severidade))
        self.destinatarios = self.destinatarios or DESTINATARIOS.get(self.severidade, [])
        self.prazo = self.prazo or PRAZO_RESPOSTA.get(self.severidade, "")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severidade"] = int(self.severidade)
        d["severidade_rotulo"] = self.severidade.rotulo
        return d


# --------------------------------------------------------------------------------------
# Regras de correlação
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RegraFusao:
    nome: str
    titulo: str
    #: cada grupo é (origens aceitas, prefixos de tipo aceitos, filtro opcional sobre o evento);
    #: todos os grupos precisam ocorrer dentro da janela
    grupos: tuple
    janela_s: float
    severidade: Severidade
    acao: str
    #: True = os grupos precisam ocorrer **nesta ordem** (causa antes do efeito, ex.: a mudança de
    #: prescrição antes da hipotensão)
    ordenada: bool = False
    supressao_s: float = 3600.0

    def casa(self, ev: Evento, grupo) -> bool:
        origens, prefixos, filtro = grupo
        return ev.origem in origens and ev.tipo.startswith(prefixos) and (filtro is None or filtro(ev))


def _g(origens: set[str], *prefixos: str, filtro=None):
    return (frozenset(origens), tuple(prefixos), filtro)


def _droga(padrao: str):
    rx = re.compile(padrao, re.I)
    return lambda ev: bool(rx.search(" ".join(map(str, [ev.evidencias.get("droga", ""), *ev.evidencias.get("drogas", [])]))))


#: fármacos com efeito hemodinâmico direto (anti-hipertensivos, diuréticos, vasodilatadores, sedativos)
HEMOATIVOS = (r"metoprolol|carvedilol|labetalol|esmolol|atenolol|propranolol|amlodipine|nifedipine|nicardipine|"
              r"hydralazine|lisinopril|captopril|enalapril|losartan|valsartan|furosemide|torsemide|bumetanide|"
              r"nitroglycerin|nitroprusside|diltiazem|verapamil|clonidine|propofol|dexmedetomidine|morphine|"
              r"hydromorphone|fentanyl|midazolam|lorazepam")
SEDATIVOS = (r"morphine|hydromorphone|fentanyl|oxycodone|methadone|tramadol|hydrocodone|lorazepam|midazolam|"
             r"diazepam|alprazolam|clonazepam|propofol|dexmedetomidine")

SINAIS_RESP_VITAIS = _g({"vitais"}, "dessaturacao", "desvio_spo2", "desvio_rr", "taquipneia", "bradipneia")
SINAIS_DEPRESSAO_RESP = _g({"vitais"}, "dessaturacao", "bradipneia")
SINAIS_RESP_FALA = _g({"audio"}, "sintoma_relatado_dispneia", "sintoma_relatado_sibilancia", "voz_respiratorio")
SINAIS_HEMO = _g({"vitais"}, "hipotensao", "risco_hipotensao", "bradicardia", "desvio_map", "desvio_hr", "taquicardia")
SINAIS_HIPOTENSAO = _g({"vitais"}, "hipotensao", "bradicardia")
SINAIS_NEURO_FALA = _g({"audio"}, "sintoma_relatado_fala_arrastada", "sintoma_relatado_deficit_neurologico",
                       "sintoma_relatado_confusao", "voz_disartria")
SINAIS_MOTORES = _g({"video", "movimentacao"}, "queda", "padrao_incorreto_persistente", "execucao_fora_do_padrao")
QUEDA = _g({"movimentacao"}, "queda", "imobilidade_pos_queda")
DROGA_SEDATIVA = _g({"prescricao"}, "opioide_benzodiazepinico", "salto_dose_aumento", "dose_atipica",
                    filtro=_droga(SEDATIVOS))
DROGA_HEMO = _g({"prescricao"}, "salto_dose_aumento", "dose_atipica", filtro=_droga(HEMOATIVOS))
FADIGA_FALA = _g({"audio"}, "voz_fadiga", "sintoma_relatado_fadiga")
FISIO_RUIM = _g({"video"}, "padrao_incorreto_persistente", "execucao_fora_do_padrao")

REGRAS_FUSAO: list[RegraFusao] = [
    RegraFusao("respiratoria_multifonte", "Insuficiência respiratória provável",
               (SINAIS_RESP_VITAIS, SINAIS_RESP_FALA), 3600, Severidade.CRITICO,
               "Avaliar via aérea e oxigenação; gasometria; considerar O₂ suplementar e raio-X de tórax"),
    RegraFusao("queda_instabilidade", "Queda com instabilidade hemodinâmica (síncope?)",
               (QUEDA, SINAIS_HEMO), 1800, Severidade.CRITICO,
               "Avaliar trauma (cabeça/quadril), glicemia capilar, ECG e PA em ortostase"),
    RegraFusao("neurologica_multifonte", "Possível evento neurológico agudo",
               (SINAIS_NEURO_FALA, SINAIS_MOTORES), 7200, Severidade.CRITICO,
               "Acionar protocolo de AVC: escala NIHSS, horário do último 'bem', TC de crânio"),
    RegraFusao("depressao_respiratoria_medicamentosa", "Depressão respiratória possivelmente medicamentosa",
               (DROGA_SEDATIVA, SINAIS_DEPRESSAO_RESP), 6 * 3600, Severidade.CRITICO,
               "Revisar opioides/sedativos; avaliar nível de consciência e FR; considerar naloxona",
               ordenada=True, supressao_s=6 * 3600),
    RegraFusao("hemodinamica_medicamentosa", "Hipotensão/bradicardia após mudança de prescrição",
               (DROGA_HEMO, SINAIS_HIPOTENSAO), 6 * 3600, Severidade.URGENTE,
               "Revisar a última alteração de dose com o prescritor; reavaliar PA e FC",
               ordenada=True, supressao_s=6 * 3600),
    RegraFusao("fadiga_fisioterapia", "Fadiga com piora da execução na fisioterapia",
               (FADIGA_FALA, FISIO_RUIM), 24 * 3600, Severidade.ATENCAO,
               "Reduzir carga/repetições da sessão e reavaliar tolerância ao esforço", supressao_s=24 * 3600),
]

PESO_SEV = {0: 0.0, 1: 1.0, 2: 3.0, 3: 6.0}
#: execução de fisioterapia e sentimento indicam qualidade do cuidado, não instabilidade clínica
PESO_ORIGEM = {"video": 0.3, "texto": 0.5}
ACAO_DIRETA = {
    "vitais": "Conferir paciente e sensor; reavaliar sinais vitais",
    "audio": "Revisar a gravação/transcrição da consulta e reavaliar o paciente",
    "texto": "Avaliar sofrimento emocional; considerar apoio psicológico",
    "video": "Revisar a execução com a fisioterapia",
    "movimentacao": "Ir ao quarto do paciente",
    "prescricao": "Conferir a prescrição com farmácia clínica e prescritor",
}


class MotorFusao:
    def __init__(self, janela_supressao_s: float = 900.0, meia_vida_s: float = 1800.0,
                 regras: list[RegraFusao] = REGRAS_FUSAO) -> None:
        self.janela_supressao_s = janela_supressao_s
        self.meia_vida_s = meia_vida_s
        self.regras = regras
        self.eventos: dict[str, list[Evento]] = defaultdict(list)
        self.alertas: list[Alerta] = []
        self._ultimo_alerta: dict[tuple[str, str], Alerta] = {}

    # ---------------------------------------------------------------------------------
    def processar(self, ev: Evento) -> list[Alerta]:
        self.eventos[ev.paciente_id].append(ev)
        novos: list[Alerta] = []
        if ev.severidade >= Severidade.ATENCAO:
            a = self._direto(ev)
            if a:
                novos.append(a)
        novos += self._correlacionar(ev)
        self.alertas += novos
        return novos

    def processar_lote(self, eventos: list[Evento]) -> list[Alerta]:
        out = []
        for ev in sorted(eventos, key=lambda e: e.t):
            out += self.processar(ev)
        return out

    def _suprimir(self, chave: tuple[str, str], t: float, janela: float | None = None) -> bool:
        ant = self._ultimo_alerta.get(chave)
        if ant is not None and t - ant.t < (janela or self.janela_supressao_s):
            ant.repeticoes += 1
            return True
        return False

    def _direto(self, ev: Evento) -> Alerta | None:
        chave = (ev.paciente_id, f"{ev.origem}:{ev.tipo}")
        if self._suprimir(chave, ev.t):
            return None
        a = Alerta(ev.paciente_id, ev.severidade, ev.t, titulo=ev.descricao,
                   justificativa=[f"[{ev.origem}/{ev.detector}] {ev.descricao}"], modalidades=[ev.origem],
                   eventos=[ev.id], regra=f"direto:{ev.tipo}", acao_sugerida=ACAO_DIRETA.get(ev.origem, ""))
        self._ultimo_alerta[chave] = a
        return a

    def _correlacionar(self, ev: Evento) -> list[Alerta]:
        out = []
        hist = self.eventos[ev.paciente_id]
        for r in self.regras:
            if not any(r.casa(ev, g) for g in r.grupos):
                continue
            evidencias = []
            t_min = ev.t - r.janela_s
            for g in r.grupos:
                cand = [e for e in hist if r.casa(e, g) and t_min <= e.t <= ev.t]
                if not cand:
                    break
                escolhido = max(cand, key=lambda e: (int(e.severidade), e.t)) if not r.ordenada else min(cand, key=lambda e: e.t)
                evidencias.append(escolhido)
                if r.ordenada:
                    t_min = escolhido.t            # o próximo grupo tem de vir depois deste
            else:
                if len({e.origem for e in evidencias}) < 2:
                    continue                      # precisa de fontes realmente distintas
                if r.ordenada and not any(e.id == ev.id for e in evidencias):
                    continue                      # só dispara no evento que completa a sequência
                chave = (ev.paciente_id, f"fusao:{r.nome}")
                if self._suprimir(chave, ev.t, r.supressao_s):
                    continue
                a = Alerta(ev.paciente_id, r.severidade, ev.t, titulo=r.titulo,
                           justificativa=[f"[{e.origem}/{e.detector}] {e.descricao}" for e in evidencias],
                           modalidades=sorted({e.origem for e in evidencias}), eventos=[e.id for e in evidencias],
                           regra=f"fusao:{r.nome}", acao_sugerida=r.acao)
                self._ultimo_alerta[chave] = a
                out.append(a)
        return out

    # ---------------------------------------------------------------------------------
    def escore(self, paciente_id: str, t: float, janela_modalidades_s: float = 3600.0) -> dict:
        evs = [e for e in self.eventos.get(paciente_id, []) if e.t <= t]
        s = sum(PESO_SEV[int(e.severidade)] * PESO_ORIGEM.get(e.origem, 1.0) *
                math.exp(-(t - e.t) * math.log(2) / self.meia_vida_s) for e in evs)
        mods = {e.origem for e in evs if t - e.t <= janela_modalidades_s and e.severidade >= Severidade.ATENCAO}
        s *= 1 + 0.25 * max(len(mods) - 1, 0)
        nivel = "vermelho" if s >= 8 else "laranja" if s >= 4 else "amarelo" if s >= 1 else "verde"
        return {"escore": round(s, 2), "nivel": nivel, "modalidades_ativas": sorted(mods)}

    def alertas_paciente(self, paciente_id: str) -> list[Alerta]:
        return [a for a in self.alertas if a.paciente_id == paciente_id]
