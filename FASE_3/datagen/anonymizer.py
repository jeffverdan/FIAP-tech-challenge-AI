"""
Anonimização e pseudonimização de dados clínicos — FASE 3.

Estratégia em duas camadas:

1. **Pseudonimização determinística** de identificadores diretos (nome, CPF, prontuário):
   o valor real é substituído por um token estável derivado de HMAC-SHA256 com sal secreto.
   Estável = o mesmo paciente recebe sempre o mesmo pseudônimo, permitindo *joins* entre
   tabelas sem reidentificação.

2. **Redação por padrão** (regex) de PII residual em texto livre: CPF, RG, CNS, telefone,
   e-mail, CEP, endereço, datas de nascimento e nomes próprios conhecidos.

Nenhum dado real é utilizado no projeto — o corpus é integralmente sintético. O módulo existe
para demonstrar o controle exigido pelo enunciado (preprocessing, anonimização e curadoria)
e para ser reutilizável caso a instituição aplique o pipeline sobre dados reais.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)

# O sal deve vir do ambiente em uso real. O default existe apenas para reprodutibilidade
# do dataset sintético versionado no repositório.
DEFAULT_SALT = "aurora-fase3-salt-sintetico"


# --------------------------------------------------------------------------------------
# Padrões de PII
# --------------------------------------------------------------------------------------

PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "CPF": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    "CNS": re.compile(r"\b\d{3}\s?\d{4}\s?\d{4}\s?\d{4}\b"),
    "RG": re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-?[0-9Xx]\b"),
    "TELEFONE": re.compile(r"(?<![\d\w])(?:\+55\s?)?\(?\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "CEP": re.compile(r"\b\d{5}-?\d{3}\b"),
    "DATA": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
    "ENDERECO": re.compile(
        r"\b(?:Rua|Av\.?|Avenida|Travessa|Alameda|Praça|Rodovia)\s+[A-ZÁÂÃÉÊÍÓÔÕÚÇ][^,;\n]{2,60}"
        r"(?:,\s*n?º?\s*\d+)?",
        re.IGNORECASE,
    ),
    "PRONTUARIO": re.compile(r"\b(?:prontu[áa]rio|matr[íi]cula)\s*[:nº#]*\s*\d{4,10}\b", re.IGNORECASE),
}

# Ordem importa: padrões mais específicos primeiro para evitar captura parcial.
PATTERN_ORDER = ["EMAIL", "CPF", "CNS", "PRONTUARIO", "TELEFONE", "CEP", "RG", "DATA", "ENDERECO"]


@dataclass
class AnonymizationReport:
    """Contabiliza o que foi removido — usado nos testes e no relatório técnico."""

    counts: dict[str, int] = field(default_factory=dict)
    pseudonyms: dict[str, str] = field(default_factory=dict)

    def bump(self, kind: str, n: int = 1) -> None:
        self.counts[kind] = self.counts.get(kind, 0) + n

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> dict:
        return {"total_removido": self.total, "por_tipo": dict(sorted(self.counts.items()))}


class Anonymizer:
    """Aplica pseudonimização determinística e redação de PII em texto livre."""

    def __init__(self, salt: str | None = None, name_blocklist: Iterable[str] | None = None) -> None:
        self.salt = (salt or os.getenv("ANONYMIZER_SALT") or DEFAULT_SALT).encode("utf-8")
        self.name_blocklist = sorted(
            {n.strip() for n in (name_blocklist or []) if n and len(n.strip()) > 2},
            key=len,
            reverse=True,
        )
        self.report = AnonymizationReport()

    # -- pseudonimização ---------------------------------------------------------------

    def pseudonymize(self, value: str, prefix: str = "PAC") -> str:
        """Token estável e irreversível sem o sal. Ex.: 'PAC-3F91A2C7'."""
        digest = hmac.new(self.salt, value.strip().lower().encode("utf-8"), hashlib.sha256)
        token = f"{prefix}-{digest.hexdigest()[:8].upper()}"
        self.report.pseudonyms[value] = token
        return token

    # -- redação de texto livre --------------------------------------------------------

    def redact(self, text: str) -> str:
        if not text:
            return text

        redacted = text

        # 1) nomes conhecidos (vindos do cadastro sintético)
        for name in self.name_blocklist:
            pattern = re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)
            redacted, n = pattern.subn("[NOME]", redacted)
            if n:
                self.report.bump("NOME", n)

        # 2) padrões estruturados
        for kind in PATTERN_ORDER:
            pattern = PII_PATTERNS[kind]
            redacted, n = pattern.subn(f"[{kind}]", redacted)
            if n:
                self.report.bump(kind, n)

        return redacted

    # -- registros estruturados --------------------------------------------------------

    def anonymize_record(
        self,
        record: dict,
        direct_identifiers: Iterable[str] = ("nome", "cpf", "cns", "rg"),
        drop_fields: Iterable[str] = ("telefone", "email", "endereco"),
        pseudonym_key: str = "nome",
    ) -> dict:
        """Devolve uma cópia anonimizada de um registro de prontuário."""
        out = dict(record)

        base = str(out.get(pseudonym_key) or out.get("cpf") or out.get("id") or "")
        out["paciente_pseudonimo"] = self.pseudonymize(base)

        for field_name in direct_identifiers:
            if field_name in out:
                out.pop(field_name)
                self.report.bump(field_name.upper())

        for field_name in drop_fields:
            if field_name in out:
                out.pop(field_name)
                self.report.bump(field_name.upper())

        # generalização: data de nascimento -> faixa etária
        if "data_nascimento" in out:
            out.pop("data_nascimento")
            self.report.bump("DATA_NASCIMENTO")

        if "idade" in out:
            out["faixa_etaria"] = faixa_etaria(int(out["idade"]))

        for key, value in list(out.items()):
            if isinstance(value, str):
                out[key] = self.redact(value)

        return out


def faixa_etaria(idade: int) -> str:
    """Generaliza idade em faixas de 5 anos (k-anonimato básico)."""
    if idade < 15:
        return "<15"
    if idade >= 50:
        return "50+"
    low = (idade // 5) * 5
    return f"{low}-{low + 4}"


def scan(text: str) -> dict[str, list[str]]:
    """Diagnóstico: devolve todas as ocorrências de PII encontradas, por tipo."""
    found: dict[str, list[str]] = {}
    for kind in PATTERN_ORDER:
        matches = PII_PATTERNS[kind].findall(text or "")
        if matches:
            found[kind] = matches
    return found


def assert_clean(text: str, context: str = "") -> None:
    """Levanta erro se sobrar PII — usado como portão de qualidade no build do dataset."""
    leftovers = scan(text)
    if leftovers:
        raise ValueError(f"PII residual detectada {context}: {leftovers}")
