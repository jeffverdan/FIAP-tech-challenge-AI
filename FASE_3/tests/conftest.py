"""Fixtures compartilhadas dos testes da FASE 3."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))


@pytest.fixture(scope="session")
def base_dir() -> Path:
    return BASE_DIR


@pytest.fixture(scope="session")
def db_sintetico(tmp_path_factory) -> Path:
    """Gera uma base sintética pequena em diretório temporário."""
    from datagen.generate_patients import gerar_base, persistir

    _, tabelas = gerar_base(n_pacientes=12, seed=7)
    caminho = tmp_path_factory.mktemp("dados") / "hospital_teste.db"
    persistir(tabelas, caminho)
    return caminho


@pytest.fixture(scope="session")
def repositorio(db_sintetico):
    from assistant.db_tools import RepositorioClinico

    return RepositorioClinico(db_sintetico)


@pytest.fixture(scope="session")
def retriever():
    from assistant.retriever import RetrieverProtocolos

    return RetrieverProtocolos.construir(forcar_bm25=True)


@pytest.fixture
def assistente(db_sintetico, retriever, tmp_path):
    """Assistente completo em modo fallback, com auditoria isolada por teste."""
    from assistant.audit import AuditLogger
    from assistant.db_tools import RepositorioClinico
    from assistant.graph import AssistenteClinico
    from assistant.llm_provider import ChatFallbackDeterministico

    auditor = AuditLogger(tmp_path / "audit", modelo="fallback-teste")
    return AssistenteClinico(
        llm=ChatFallbackDeterministico(),
        retriever=retriever,
        repositorio=RepositorioClinico(db_sintetico),
        auditor=auditor,
    )
