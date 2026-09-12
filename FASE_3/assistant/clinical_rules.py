"""
Motor de regras clínicas determinísticas — FASE 3.

Implementa, em Python puro, os gatilhos objetivos dos protocolos institucionais
(PROT-002, PROT-004, PROT-005, PROT-006 e a matriz de alertas do PROT-007).

Por que regras e não a LLM: gatilhos de segurança (alerta crítico, exame pendente,
pendência pré-indução) precisam ser **determinísticos, auditáveis e testáveis**. A LLM
redige e contextualiza; as regras decidem. Essa separação é o que garante que um alerta
crítico nunca deixe de disparar por variação de amostragem do modelo.

Todas as funções recebem dicionários já anonimizados e devolvem estruturas serializáveis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Literal

Severidade = Literal["CRITICO", "ALTO", "MEDIO", "BAIXO"]

# Exames do "Painel SOP" obrigatório (PROT-001, seção 4)
PAINEL_SOP = [
    "Beta-hCG", "TSH", "Prolactina", "17-OH-progesterona", "Testosterona total",
    "SHBG", "FSH", "LH", "Glicemia de jejum", "Insulina de jejum", "Hemoglobina glicada",
]

# Checklist pré-indução (PROT-004, seção 2)
CHECKLIST_PRE_INDUCAO = [
    ("Permeabilidade tubária", "exame"),
    ("Espermograma do parceiro", "exame"),
    ("TSH", "exame"),
    ("Prolactina", "exame"),
    ("Rastreio metabólico atualizado (TOTG ou HbA1c)", "exame"),
    ("Aconselhamento sobre gestação múltipla e hiperestimulação", "registro"),
    ("Termo de consentimento assinado", "registro"),
    ("Ácido fólico iniciado há >= 30 dias", "prescricao"),
]


@dataclass
class Alerta:
    severidade: Severidade
    gatilho: str
    protocolo: str
    detalhe: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dias_desde(iso: str | None, referencia: date | None = None) -> int | None:
    if not iso:
        return None
    try:
        d = datetime.fromisoformat(iso).date()
    except ValueError:
        return None
    return ((referencia or date.today()) - d).days


def _ultimo_valor(exames: list[dict], nome: str) -> dict | None:
    liberados = [
        e for e in exames
        if e.get("nome_exame") == nome and e.get("status") == "liberado" and e.get("valor") is not None
    ]
    if not liberados:
        return None
    return max(liberados, key=lambda e: e.get("data_resultado") or "")


# --------------------------------------------------------------------------------------
# Exames pendentes / vencidos
# --------------------------------------------------------------------------------------

def exames_pendentes(exames: list[dict], hoje: date | None = None) -> list[dict]:
    """Exames solicitados sem resultado liberado (PROT-007: severidade MÉDIO após 60 dias)."""
    saida = []
    for e in exames:
        if e.get("status") != "pendente":
            continue
        dias = _dias_desde(e.get("data_solicitacao"), hoje) or 0
        saida.append(
            {
                "nome_exame": e["nome_exame"],
                "data_solicitacao": e["data_solicitacao"],
                "dias_em_aberto": dias,
                "atrasado": dias > 60,
            }
        )
    return sorted(saida, key=lambda x: -x["dias_em_aberto"])


def painel_sop_faltante(exames: list[dict]) -> list[str]:
    """Itens do Painel SOP que nunca tiveram resultado liberado."""
    liberados = {e["nome_exame"] for e in exames if e.get("status") == "liberado"}
    return [nome for nome in PAINEL_SOP if nome not in liberados]


# --------------------------------------------------------------------------------------
# Matriz de alertas (PROT-007, seção 3)
# --------------------------------------------------------------------------------------

def avaliar_alertas(paciente: dict, exames: list[dict], hoje: date | None = None) -> list[Alerta]:
    alertas: list[Alerta] = []

    hb = _ultimo_valor(exames, "Hemoglobina")
    if hb and hb["valor"] < 8:
        alertas.append(Alerta("CRITICO", "Hemoglobina < 8 g/dL",
                              "PROT-007 §3 / PROT-005 §4",
                              f"Hemoglobina de {hb['valor']} g/dL em {hb['data_resultado']}."))

    if (paciente.get("pa_sistolica") or 0) < 90:
        alertas.append(Alerta("CRITICO", "Instabilidade hemodinâmica", "PROT-007 §3",
                              f"PA sistólica de {paciente['pa_sistolica']} mmHg."))

    tt = _ultimo_valor(exames, "Testosterona total")
    if tt and tt["valor"] > 140:  # 2x o limite superior de referência (70 ng/dL)
        alertas.append(Alerta("ALTO", "Testosterona > 2x LSN", "PROT-003 §5",
                              f"Testosterona total de {tt['valor']} ng/dL — investigar neoplasia."))

    hba1c = _ultimo_valor(exames, "Hemoglobina glicada")
    if hba1c and hba1c["valor"] >= 6.5 and "diabetes" not in (paciente.get("comorbidades") or "").lower():
        alertas.append(Alerta("ALTO", "HbA1c >= 6,5% sem diagnóstico registrado", "PROT-002 §5",
                              f"HbA1c de {hba1c['valor']}% sem diabetes no cadastro."))

    if (paciente.get("pa_sistolica") or 0) >= 140 or (paciente.get("pa_diastolica") or 0) >= 90:
        alertas.append(Alerta("ALTO", "PA >= 140/90", "PROT-002 §5",
                              f"PA de {paciente.get('pa_sistolica')}/{paciente.get('pa_diastolica')} mmHg."))

    eco = _ultimo_valor(exames, "Eco endometrial (USG TV)")
    if eco and eco["valor"] > 12:
        alertas.append(Alerta("ALTO", "Eco endometrial > 12 mm", "PROT-005 §4",
                              f"Eco endometrial de {eco['valor']} mm."))
    elif eco and eco["valor"] > 7 and (paciente.get("sangramentos_12m") or 0) < 4:
        alertas.append(Alerta("MEDIO", "Eco > 7 mm com oligomenorreia", "PROT-005 §2",
                              f"Eco de {eco['valor']} mm e {paciente.get('sangramentos_12m')} sangramentos/ano."))

    for pend in exames_pendentes(exames, hoje):
        if pend["atrasado"]:
            alertas.append(Alerta("MEDIO", "Exame pendente > 60 dias", "PROT-007 §3",
                                  f"{pend['nome_exame']} em aberto há {pend['dias_em_aberto']} dias."))

    dias_multi = _dias_desde(paciente.get("ultimo_retorno_multi"), hoje)
    if dias_multi is not None and dias_multi > 180:
        alertas.append(Alerta("MEDIO", "Sem retorno multiprofissional > 6 meses", "PROT-006 §4",
                              f"Último retorno há {dias_multi} dias."))

    if (paciente.get("sangramentos_12m") or 0) < 4:
        alertas.append(Alerta("MEDIO", "Menos de 4 sangramentos/ano", "PROT-005 §2",
                              f"{paciente.get('sangramentos_12m')} sangramentos nos últimos 12 meses — "
                              "indicada investigação endometrial."))

    if paciente.get("ferriman_gallwey") is None:
        alertas.append(Alerta("BAIXO", "Ferriman-Gallwey ausente", "PROT-007 §3",
                              "Escore não registrado no prontuário."))

    ordem = {"CRITICO": 0, "ALTO": 1, "MEDIO": 2, "BAIXO": 3}
    return sorted(alertas, key=lambda a: ordem[a.severidade])


def severidade_maxima(alertas: list[Alerta]) -> Severidade | None:
    return alertas[0].severidade if alertas else None


# --------------------------------------------------------------------------------------
# Regras específicas por protocolo
# --------------------------------------------------------------------------------------

def encaminhamento_por_imc(imc: float | None) -> str:
    """PROT-006, seção 2."""
    if imc is None:
        return "IMC não registrado — impossível aplicar o PROT-006 §2."
    if imc < 25:
        return "Orientação nutricional na consulta e atividade física estruturada."
    if imc < 30:
        return "Encaminhamento para nutrição ambulatorial e educador físico."
    if imc < 35:
        return "Encaminhamento ao ambulatório multiprofissional (nutrição, educação física, psicologia)."
    return "Ambulatório multiprofissional e avaliação para terapia intensiva de obesidade."


def classificar_hirsutismo(escore: int | None) -> str:
    """PROT-003, seção 2."""
    if escore is None:
        return "não avaliado"
    if escore < 6:
        return "ausente (abaixo do corte institucional de 6)"
    if escore <= 15:
        return "leve"
    if escore <= 25:
        return "moderado"
    return "grave"


def pendencias_pre_inducao(paciente: dict, exames: list[dict], prescricoes: list[dict]) -> list[str]:
    """PROT-004, seção 2 — devolve os itens do checklist que NÃO estão documentados."""
    liberados = {e["nome_exame"] for e in exames if e.get("status") == "liberado"}
    tem_rastreio = bool(liberados & {"TOTG 75g - 120 min", "Hemoglobina glicada"})
    tem_folato = any("fólico" in (p.get("classe_terapeutica") or "") and p.get("ativa") for p in prescricoes)

    pendentes = []
    for item, tipo in CHECKLIST_PRE_INDUCAO:
        if item == "Rastreio metabólico atualizado (TOTG ou HbA1c)":
            ok = tem_rastreio
        elif item == "Ácido fólico iniciado há >= 30 dias":
            ok = tem_folato
        elif tipo == "exame":
            ok = item in liberados
        else:
            ok = False  # registros documentais não são modelados na base sintética
        if not ok:
            pendentes.append(item)
    return pendentes


def indica_investigacao_endometrial(
    paciente: dict, exames: list[dict], hoje: date | None = None
) -> tuple[bool, list[str]]:
    """PROT-005, seção 2 — devolve (indicado, motivos)."""
    motivos = []
    dias_sangramento = _dias_desde(paciente.get("ultimo_sangramento"), hoje)
    if dias_sangramento is not None and dias_sangramento >= 90:
        motivos.append(f"amenorreia de {dias_sangramento} dias (>= 3 meses)")
    if (paciente.get("sangramentos_12m") or 0) < 4:
        motivos.append(f"apenas {paciente.get('sangramentos_12m')} sangramentos nos últimos 12 meses")
    eco = _ultimo_valor(exames, "Eco endometrial (USG TV)")
    if eco and eco["valor"] > 7 and dias_sangramento and dias_sangramento >= 90:
        motivos.append(f"eco endometrial de {eco['valor']} mm em paciente amenorreica")
    if (paciente.get("imc") or 0) >= 35:
        motivos.append(f"IMC de {paciente.get('imc')} kg/m² (fator de risco adicional)")
    return bool(motivos), motivos
