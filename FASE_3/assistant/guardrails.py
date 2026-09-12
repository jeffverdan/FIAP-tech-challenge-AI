"""
Guardrails do assistente clínico — FASE 3 (PROT-007).

Três camadas, aplicadas em pontos diferentes do grafo:

1. **Entrada** (`avaliar_entrada`) — classifica a intenção e bloqueia antes de qualquer
   chamada à LLM: emergência clínica, pedido de prescrição, pedido de fechamento de
   diagnóstico, tentativa de alteração de prontuário e assunto fora do escopo.
2. **Aterramento** (`exige_fonte`) — resposta clínica sem trecho de protocolo recuperado é
   substituída pela mensagem institucional de ausência de fonte.
3. **Saída** (`avaliar_saida`) — última linha de defesa: remove dose/posologia que tenha
   escapado, redige PII residual e garante a marcação de validação humana.

Nenhuma dessas camadas depende da LLM. Um guardrail que precise que o modelo "se comporte"
não é um guardrail — por isso todos são determinísticos e testáveis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from assistant.prompts import (MENSAGEM_EMERGENCIA, MENSAGEM_FORA_DE_ESCOPO,
                               MENSAGEM_RECUSA_PRESCRICAO, MENSAGEM_SEM_FONTE)
from datagen.anonymizer import PII_PATTERNS

logger = logging.getLogger("assistant.guardrails")


class Intencao(str, Enum):
    CLINICA = "clinica"
    PRESCRICAO = "prescricao"
    EMERGENCIA = "emergencia"
    DIAGNOSTICO_DEFINITIVO = "diagnostico_definitivo"
    ALTERACAO_PRONTUARIO = "alteracao_prontuario"
    FORA_ESCOPO = "fora_escopo"


PADROES = {
    Intencao.EMERGENCIA: [
        r"\bhemorragi", r"sangramento (intenso|abundante|profuso|macic)",
        r"\binst[áa]vel\b", r"instabilidade hemodin", r"\bchoque\b", r"\bs[íi]ncope\b",
        r"\bdesmai", r"\bhiperestimula", r"\bolig[úu]ria\b", r"\bdispneia\b",
        r"\bpa\s*\d{2}\s*[/x]\s*\d{2}\b", r"press[ãa]o\s*\d{2}\s*[/x]\s*\d{2}\b",
        r"\bfc\s*(>|acima de|maior que)?\s*1[2-9]\d\b", r"\btaquic[áa]rdi",
        r"\bemerg[êe]nci", r"\bagora\b.{0,20}\b(sangrando|passando mal)",
    ],
    Intencao.PRESCRICAO: [
        r"\bprescrev", r"\bprescri[çc]", r"\breceit", r"\bqual\s+(a\s+)?dose",
        r"\bposologia\b", r"\bquantos?\s+(mg|ml|comprimidos)", r"\bme (passe|manda|d[êe])\b.{0,30}\b(dose|receita|rem[ée]dio)",
        r"\bqual (rem[ée]dio|f[áa]rmaco|medicamento)\b.{0,20}\b(dar|passar|usar|prescrever)",
        r"\bj[áa] (prescreve|receita)\b", r"\bpreench(a|er)\b.{0,25}\breceitu",
    ],
    Intencao.DIAGNOSTICO_DEFINITIVO: [
        r"\b(confirm|feche|fechar|firme|firmar)\w*\b.{0,30}\bdiagn[óo]stico",
        r"\bdescart\w+\b.{0,25}\bdiagn[óo]stico", r"\bela tem sop\??\s*$",
        r"\bdiagn[óo]stico (definitivo|fechado)\b",
    ],
    Intencao.ALTERACAO_PRONTUARIO: [
        r"\b(apagu|apagar|delet|remov|exclu)\w*\b.{0,30}\b(prontu[áa]rio|registro|exame)",
        r"\b(altere|alterar|edite|editar|mude|mudar)\b.{0,30}\b(prontu[áa]rio|registro)",
        r"\b(registre|registrar|grave|salve)\b.{0,25}\bno prontu[áa]rio\b",
    ],
    Intencao.FORA_ESCOPO: [
        r"\bplano de sa[úu]de\b", r"\binvestiment", r"\bcriptomoeda", r"\bfutebol\b",
        r"\breceita de (bolo|comida)", r"\bcomo hackear", r"\bescreva um (poema|c[óo]digo)\b",
        r"\bqual (o )?melhor (celular|carro|not[eb]book)\b", r"\bpiada\b",
    ],
}

PADROES_COMPILADOS = {
    intencao: [re.compile(p, re.IGNORECASE) for p in padroes]
    for intencao, padroes in PADROES.items()
}

# Ordem de precedência: emergência sempre vence.
PRECEDENCIA = [
    Intencao.EMERGENCIA,
    Intencao.ALTERACAO_PRONTUARIO,
    Intencao.PRESCRICAO,
    Intencao.DIAGNOSTICO_DEFINITIVO,
    Intencao.FORA_ESCOPO,
]

MENSAGEM_DIAGNOSTICO = (
    "Não posso confirmar nem descartar diagnóstico de forma definitiva (PROT-007 §1). "
    "Posso apresentar quais critérios de Rotterdam estão documentados, quais diagnósticos "
    "diferenciais já foram excluídos e o que permanece pendente, para que o médico assistente "
    "conclua."
)

MENSAGEM_ALTERACAO = (
    "Não posso alterar, apagar ou registrar nada no prontuário eletrônico (PROT-007 §1). "
    "Meu acesso à base é somente leitura. O registro é ato do profissional responsável."
)

# Padrões de dose/posologia que não podem sair na resposta.
# Detecção de dose/posologia na saída.
#
# O problema difícil aqui é distinguir **dose** de **valor de exame**: "volume ovariano >= 10 mL"
# e "glicemia de 126 mg/dL" são leitura legítima do prontuário; "iniciar 850 mg" é prescrição.
# Duas estratégias combinadas:
#   1. unidades que praticamente só aparecem como dose (mg, mcg, UI) e não seguidas de "/"
#      (o "/" indica concentração de exame, como mg/dL);
#   2. unidades ambíguas (mL, g, gotas) apenas quando precedidas por um verbo de posologia.
PADROES_DOSE = [
    re.compile(r"\b\d+[.,]?\d*\s?(mg|mcg|µg|ui)\b(?!\s*/)", re.IGNORECASE),
    re.compile(r"\b\d+\s?(comprimido|c[áa]psula|ampola|dr[áa]gea)s?\b", re.IGNORECASE),
    re.compile(r"\b\d+\s?x\s?(ao dia|/dia|por dia)\b", re.IGNORECASE),
]

VERBOS_POSOLOGIA = r"(?:tomar|administrar|usar|iniciar|prescrever|receitar|dar|aplicar)"

# Grupo 1 é preservado (o verbo e o texto intermediário); grupo 2 é a dose mascarada.
PADROES_DOSE_CONTEXTO = [
    re.compile(rf"\b({VERBOS_POSOLOGIA}\b[^.\n]{{0,40}}?)(\d+[.,]?\d*\s?(?:ml|g|gotas?)\b)",
               re.IGNORECASE),
]

# Unidades que aparecem legitimamente em resultados de exame e antropometria.
EXCECOES_DOSE = re.compile(
    r"\b\d+[.,]?\d*\s?(mg/dl|ng/dl|ng/ml|mui/l|mui/ml|uui/ml|nmol/l|g/dl|mmhg|kg/m|mm|cm|kg|%)\b",
    re.IGNORECASE)

MARCADOR_VALIDACAO = "SUGESTÃO NÃO VALIDADA"


@dataclass
class ResultadoGuardrail:
    intencao: Intencao
    bloqueado: bool = False
    mensagem: str | None = None
    acionados: list[str] = field(default_factory=list)
    requer_validacao_humana: bool = False


# --------------------------------------------------------------------------------------
# Camada 1 — entrada
# --------------------------------------------------------------------------------------

def classificar_intencao(pergunta: str) -> Intencao:
    for intencao in PRECEDENCIA:
        if any(p.search(pergunta) for p in PADROES_COMPILADOS[intencao]):
            return intencao
    return Intencao.CLINICA


def avaliar_entrada(pergunta: str, severidade_paciente: str | None = None) -> ResultadoGuardrail:
    intencao = classificar_intencao(pergunta)

    # Um alerta CRÍTICO no prontuário eleva a interação a emergência mesmo que a pergunta
    # tenha sido formulada de forma banal.
    if severidade_paciente == "CRITICO" and intencao is not Intencao.FORA_ESCOPO:
        return ResultadoGuardrail(
            intencao=Intencao.EMERGENCIA, bloqueado=True, mensagem=MENSAGEM_EMERGENCIA,
            acionados=["emergencia_por_alerta_critico_no_prontuario"],
        )

    bloqueios = {
        Intencao.EMERGENCIA: (MENSAGEM_EMERGENCIA, "bloqueio_emergencia"),
        Intencao.PRESCRICAO: (MENSAGEM_RECUSA_PRESCRICAO, "bloqueio_prescricao"),
        Intencao.DIAGNOSTICO_DEFINITIVO: (MENSAGEM_DIAGNOSTICO, "bloqueio_diagnostico_definitivo"),
        Intencao.ALTERACAO_PRONTUARIO: (MENSAGEM_ALTERACAO, "bloqueio_alteracao_prontuario"),
        Intencao.FORA_ESCOPO: (MENSAGEM_FORA_DE_ESCOPO, "bloqueio_fora_de_escopo"),
    }
    if intencao in bloqueios:
        mensagem, acionado = bloqueios[intencao]
        logger.info("Guardrail de entrada acionado: %s", acionado)
        return ResultadoGuardrail(intencao=intencao, bloqueado=True, mensagem=mensagem,
                                  acionados=[acionado])

    return ResultadoGuardrail(intencao=intencao, requer_validacao_humana=True)


# --------------------------------------------------------------------------------------
# Camada 2 — aterramento
# --------------------------------------------------------------------------------------

def exige_fonte(fontes: list[dict]) -> str | None:
    """Devolve a mensagem institucional quando não há trecho de protocolo recuperado."""
    return None if fontes else MENSAGEM_SEM_FONTE


# --------------------------------------------------------------------------------------
# Camada 3 — saída
# --------------------------------------------------------------------------------------

def _remover_doses(texto: str) -> tuple[str, bool]:
    """Mascara dose/posologia preservando unidades legítimas de exame e antropometria."""
    alterado = False

    def substituir(match: re.Match) -> str:
        nonlocal alterado
        if EXCECOES_DOSE.fullmatch(match.group(0).strip()):
            return match.group(0)
        alterado = True
        return "[DOSE REMOVIDA PELO GUARDRAIL]"

    for padrao in PADROES_DOSE:
        texto = padrao.sub(substituir, texto)

    def substituir_contextual(match: re.Match) -> str:
        nonlocal alterado
        alterado = True
        return f"{match.group(1)}[DOSE REMOVIDA PELO GUARDRAIL]"

    for padrao in PADROES_DOSE_CONTEXTO:
        texto = padrao.sub(substituir_contextual, texto)

    return texto, alterado


def _redigir_pii(texto: str) -> tuple[str, list[str]]:
    encontrados = []
    for tipo in ("CPF", "CNS", "EMAIL", "TELEFONE", "RG"):
        texto, n = PII_PATTERNS[tipo].subn(f"[{tipo}]", texto)
        if n:
            encontrados.append(tipo)
    return texto, encontrados


def avaliar_saida(resposta: str, fontes: list[dict], intencao: Intencao) -> tuple[str, list[str]]:
    acionados: list[str] = []

    resposta, doses_removidas = _remover_doses(resposta)
    if doses_removidas:
        acionados.append("saida_sanitizada_dose")
        resposta += ("\n\n[Guardrail] Dose ou posologia foi removida da resposta: o assistente "
                     "não prescreve (PROT-007 §1).")

    resposta, pii = _redigir_pii(resposta)
    if pii:
        acionados.append(f"saida_redigida_pii:{'+'.join(pii)}")

    if not fontes:
        acionados.append("saida_sem_fonte")
        resposta = f"{MENSAGEM_SEM_FONTE}\n\n{resposta}"
    else:
        citacoes = "; ".join(f"{f['protocolo']} v{f['versao']} — {f['secao']}" for f in fontes)
        if "Fonte:" not in resposta:
            resposta += f"\n\nFonte: {citacoes}"
            acionados.append("fonte_anexada_automaticamente")

    if intencao is Intencao.CLINICA and MARCADOR_VALIDACAO.lower() not in resposta.lower():
        resposta += (f"\n\n{MARCADOR_VALIDACAO} — requer validação do médico assistente "
                     "antes de qualquer conduta (PROT-007 §2).")
        acionados.append("marcacao_validacao_humana_anexada")

    return resposta, acionados
