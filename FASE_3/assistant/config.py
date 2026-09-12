"""Configuração central do assistente — lida de variáveis de ambiente / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # opcional: o projeto roda sem python-dotenv
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except Exception:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parents[1]


def _bool(nome: str, default: bool = False) -> bool:
    return os.getenv(nome, str(default)).strip().lower() in {"1", "true", "yes", "sim"}


def _caminho(nome: str, default: str) -> Path:
    valor = Path(os.getenv(nome, default))
    return valor if valor.is_absolute() else BASE_DIR / valor


@dataclass
class Config:
    base_model: str = field(default_factory=lambda: os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    adapter_path: Path = field(default_factory=lambda: _caminho("ADAPTER_PATH", "finetune/outputs/adapter"))
    force_fallback: bool = field(default_factory=lambda: _bool("FORCE_FALLBACK", False))
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "auto"))
    max_new_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_NEW_TOKENS", "400")))

    embedding_model: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))
    vectorstore_path: Path = field(default_factory=lambda: _caminho("VECTORSTORE_PATH", "data/vectorstore"))
    documentos_rag: Path = field(default_factory=lambda: _caminho(
        "DOCUMENTOS_RAG", "data/processed/documentos_rag.jsonl"))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))

    hospital_db: Path = field(default_factory=lambda: _caminho("HOSPITAL_DB", "data/hospital.db"))

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    audit_log_dir: Path = field(default_factory=lambda: _caminho("AUDIT_LOG_DIR", "logs/audit"))


CONFIG = Config()
