"""
Cadeias LangChain (LCEL) do assistente — FASE 3.

A cadeia clínica é a única parte do sistema em que a LLM tem a palavra. Ela recebe blocos
explicitamente delimitados e é instruída a não sair deles:

    PACIENTE:  resumo anonimizado do prontuário (opcional)
    FATOS:     saída do motor determinístico de regras (opcional)
    CONTEXTO:  trechos de protocolo recuperados pelo RAG
    PERGUNTA:  a dúvida do médico

Separar FATOS de CONTEXTO é deliberado: o que é calculado por regra entra como fato dado, e
a LLM não precisa (nem deve) recalcular. Isso reduz a superfície de alucinação numérica —
o erro mais perigoso em um assistente clínico.
"""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from assistant.prompts import SYSTEM_PROMPT_COM_CONTEXTO

logger = logging.getLogger("assistant.chains")

TEMPLATE_HUMANO = """{bloco_paciente}{bloco_fatos}CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}"""

PROMPT_CLINICO = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT_COM_CONTEXTO), ("human", TEMPLATE_HUMANO)]
)


def montar_cadeia_clinica(llm: BaseChatModel) -> Runnable:
    """PROMPT | LLM | parser — a integração da LLM customizada com o LangChain."""
    return PROMPT_CLINICO | llm | StrOutputParser()


def formatar_blocos(pergunta: str, contexto: str, resumo_paciente: str | None = None,
                    fatos: str | None = None) -> dict[str, str]:
    return {
        "pergunta": pergunta.strip(),
        "contexto": contexto.strip() or "(nenhum trecho de protocolo recuperado)",
        "bloco_paciente": f"PACIENTE:\n{resumo_paciente.strip()}\n\n" if resumo_paciente else "",
        "bloco_fatos": f"FATOS:\n{fatos.strip()}\n\n" if fatos else "",
    }


def prompt_renderizado(entradas: dict[str, str]) -> str:
    """Texto exato enviado à LLM — gravado na auditoria para reprodutibilidade."""
    return PROMPT_CLINICO.format(**entradas)
