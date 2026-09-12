"""
Trilha de auditoria do assistente — FASE 3 (PROT-007, seção 5).

Formato: JSONL *append-only*, um arquivo por dia, em ``logs/audit/aurora-YYYY-MM-DD.jsonl``.

Escolhas de design:
  * **JSONL append-only** — cada linha é um evento imutável; não há update nem delete. Isso é
    o que torna a trilha auditável: reconstruir o estado é reler os eventos em ordem.
  * **Um evento por nó do grafo**, e não um por interação. Assim a auditoria mostra *como* o
    assistente chegou à resposta, e não apenas o que ele respondeu.
  * **Sem PII** — a paciente aparece apenas pelo pseudônimo estável gerado no pipeline de
    anonimização, conforme exigido pelo protocolo.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from assistant.config import CONFIG

logger = logging.getLogger("assistant.audit")

_LOCK = threading.Lock()

CAMPOS_PROIBIDOS = {"nome", "cpf", "cns", "rg", "telefone", "email", "endereco", "data_nascimento"}


def novo_interaction_id() -> str:
    return f"INT-{uuid.uuid4().hex[:12].upper()}"


def _sanitizar(valor: Any) -> Any:
    """Remove chaves com PII em qualquer profundidade — proteção de última linha."""
    if isinstance(valor, dict):
        return {k: _sanitizar(v) for k, v in valor.items() if k.lower() not in CAMPOS_PROIBIDOS}
    if isinstance(valor, list):
        return [_sanitizar(v) for v in valor]
    return valor


class AuditLogger:
    def __init__(self, diretorio: Path | None = None, adapter_hash: str = "n/a",
                 modelo: str = "n/a") -> None:
        self.diretorio = Path(diretorio or CONFIG.audit_log_dir)
        self.diretorio.mkdir(parents=True, exist_ok=True)
        self.adapter_hash = adapter_hash
        self.modelo = modelo

    @property
    def arquivo(self) -> Path:
        return self.diretorio / f"aurora-{date.today().isoformat()}.jsonl"

    def registrar(
        self,
        interaction_id: str,
        no_do_grafo: str,
        entrada: Any = None,
        saida: Any = None,
        *,
        usuario_crm: str = "n/a",
        paciente_pseudonimo: str | None = None,
        fontes: list[dict] | None = None,
        guardrails_acionados: list[str] | None = None,
        severidade_alerta: str | None = None,
        latencia_ms: float | None = None,
        extra: dict | None = None,
    ) -> dict:
        evento = {
            "interaction_id": interaction_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "usuario_crm": usuario_crm,
            "paciente_pseudonimo": paciente_pseudonimo,
            "no_do_grafo": no_do_grafo,
            "entrada": _sanitizar(entrada),
            "saida": _sanitizar(saida),
            "fontes": _sanitizar(fontes or []),
            "guardrails_acionados": guardrails_acionados or [],
            "severidade_alerta": severidade_alerta,
            "latencia_ms": round(latencia_ms, 2) if latencia_ms is not None else None,
            "modelo": self.modelo,
            "adapter_hash": self.adapter_hash,
        }
        linha = json.dumps(evento, ensure_ascii=False, default=str)
        with _LOCK:
            with self.arquivo.open("a", encoding="utf-8") as fh:
                fh.write(linha + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        logger.debug("audit %s | %s", interaction_id, no_do_grafo)
        return evento

    # -- leitura para a interface e para os testes ------------------------------------

    def ler_eventos(self, interaction_id: str | None = None, limite: int = 200,
                    dia: date | None = None) -> list[dict]:
        arquivo = self.diretorio / f"aurora-{(dia or date.today()).isoformat()}.jsonl"
        if not arquivo.exists():
            return []
        eventos = []
        for linha in arquivo.read_text(encoding="utf-8").splitlines():
            if not linha.strip():
                continue
            evento = json.loads(linha)
            if interaction_id and evento["interaction_id"] != interaction_id:
                continue
            eventos.append(evento)
        return eventos[-limite:]

    def arquivos_disponiveis(self) -> list[Path]:
        return sorted(self.diretorio.glob("aurora-*.jsonl"), reverse=True)
