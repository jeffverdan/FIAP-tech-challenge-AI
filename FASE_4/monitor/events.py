"""Contrato único entre os detectores e a camada de fusão.

Todo detector — vídeo, áudio, sinais vitais, prescrições — emite :class:`Evento`. A fusão
(`monitor/fusion`) só enxerga eventos; nunca chama um modelo. Isso mantém a regra da FASE 3:
os modelos produzem *evidências*, a decisão de alertar é determinística e auditável.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable


class Severidade(IntEnum):
    INFO = 0
    ATENCAO = 1
    URGENTE = 2
    CRITICO = 3

    @property
    def rotulo(self) -> str:
        return {0: "info", 1: "atenção", 2: "urgente", 3: "crítico"}[int(self)]


#: Origens válidas — uma por modalidade.
ORIGENS = {"vitais", "prescricao", "audio", "video", "movimentacao", "texto"}


@dataclass
class Evento:
    paciente_id: str
    origem: str
    tipo: str
    severidade: Severidade
    t: float                      # segundos desde o início do monitoramento do paciente
    descricao: str
    detector: str                 # "regra" | "estatistico" | "ml" | "azure" | "pose" | ...
    evidencias: dict[str, Any] = field(default_factory=dict)
    duracao_s: float = 0.0
    id: str = field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:10].upper()}")

    def __post_init__(self) -> None:
        if self.origem not in ORIGENS:
            raise ValueError(f"origem inválida: {self.origem!r}")
        self.severidade = Severidade(int(self.severidade))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severidade"] = int(self.severidade)
        d["severidade_rotulo"] = self.severidade.rotulo
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Evento":
        d = {k: v for k, v in d.items() if k != "severidade_rotulo"}
        return cls(**d)


class EventStore:
    """Armazena eventos em JSONL append-only (um arquivo por execução/paciente)."""

    _lock = threading.Lock()

    def __init__(self, caminho: Path) -> None:
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)

    def gravar(self, eventos: Iterable[Evento]) -> int:
        n = 0
        with self._lock, open(self.caminho, "a", encoding="utf-8") as f:
            for ev in eventos:
                f.write(json.dumps(ev.to_dict(), ensure_ascii=False, default=_json_default) + "\n")
                n += 1
        return n

    def ler(self) -> list[Evento]:
        if not self.caminho.exists():
            return []
        with open(self.caminho, encoding="utf-8") as f:
            return [Evento.from_dict(json.loads(ln)) for ln in f if ln.strip()]


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    return str(o)


def salvar_eventos(eventos: Iterable[Evento], caminho: Path) -> None:
    """Grava uma lista de eventos como JSON (sobrescreve) — usado para resultados de batch."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps([e.to_dict() for e in eventos], ensure_ascii=False, indent=1,
                                  default=_json_default), encoding="utf-8")


def carregar_eventos(caminho: Path) -> list[Evento]:
    return [Evento.from_dict(d) for d in json.loads(Path(caminho).read_text(encoding="utf-8"))]
