"""
Recuperação de protocolos institucionais (RAG) — FASE 3.

O RAG é o que dá **explicabilidade** ao assistente: toda resposta cita o trecho de protocolo
efetivamente recuperado, com id, versão e seção — exigência do PROT-007 §4.

Dois backends, escolhidos automaticamente:

``BackendVetorial``
    Embeddings multilíngues (sentence-transformers) indexados em FAISS. Melhor para perguntas
    formuladas com vocabulário diferente do texto do protocolo.

``BackendBM25``
    Okapi BM25 em Python puro, sem dependências pesadas. Além de ser o fallback offline, é
    surpreendentemente forte neste corpus: os protocolos usam vocabulário técnico consistente
    e os médicos perguntam com os mesmos termos ("Ferriman-Gallwey", "TOTG", "eco endometrial").
"""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from assistant.config import CONFIG

logger = logging.getLogger("assistant.retriever")

STOPWORDS = {
    "a", "o", "as", "os", "de", "da", "do", "das", "dos", "e", "em", "no", "na", "nos", "nas",
    "um", "uma", "para", "por", "com", "que", "se", "ao", "aos", "à", "as", "sobre", "qual",
    "quais", "quando", "como", "onde", "ser", "e", "ou", "the", "of", "is", "deve", "devo",
    "pode", "posso", "mais", "menos", "sem", "seu", "sua", "isso", "esta", "este", "essa",
}


def normalizar(texto: str) -> list[str]:
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", texto)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


@dataclass
class Trecho:
    protocolo: str
    versao: str
    titulo: str
    secao: str
    conteudo: str
    score: float

    def como_fonte(self, tamanho_trecho: int = 320) -> dict[str, Any]:
        return {
            "protocolo": self.protocolo,
            "versao": self.versao,
            "secao": self.secao,
            "trecho": re.sub(r"\s+", " ", self.conteudo)[:tamanho_trecho],
            "score": round(self.score, 4),
        }

    def citacao(self) -> str:
        return f"{self.protocolo} v{self.versao} — {self.secao}"


class Backend(Protocol):
    nome: str

    def buscar(self, consulta: str, k: int) -> list[tuple[int, float]]: ...


# --------------------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------------------

class BackendBM25:
    """Okapi BM25 (k1=1.5, b=0.75) sobre os trechos de protocolo."""

    nome = "bm25"

    def __init__(self, documentos: list[dict], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.corpus = [normalizar(f"{d['titulo']} {d['secao']} {d['conteudo']}") for d in documentos]
        self.n = len(self.corpus)
        self.comprimentos = [len(c) for c in self.corpus]
        self.media = sum(self.comprimentos) / max(self.n, 1)
        self.frequencias = [Counter(c) for c in self.corpus]

        df: Counter = Counter()
        for c in self.corpus:
            df.update(set(c))
        self.idf = {
            termo: math.log(1 + (self.n - freq + 0.5) / (freq + 0.5))
            for termo, freq in df.items()
        }

    def buscar(self, consulta: str, k: int) -> list[tuple[int, float]]:
        termos = normalizar(consulta)
        pontuacoes = []
        for i, tf in enumerate(self.frequencias):
            score = 0.0
            for termo in termos:
                if termo not in tf:
                    continue
                f = tf[termo]
                denominador = f + self.k1 * (1 - self.b + self.b * self.comprimentos[i] / max(self.media, 1))
                score += self.idf.get(termo, 0.0) * f * (self.k1 + 1) / denominador
            if score > 0:
                pontuacoes.append((i, score))
        pontuacoes.sort(key=lambda x: -x[1])
        return pontuacoes[:k]


# --------------------------------------------------------------------------------------
# Vetorial (FAISS + sentence-transformers)
# --------------------------------------------------------------------------------------

class BackendVetorial:
    nome = "faiss"

    def __init__(self, documentos: list[dict], modelo: str, cache: Path | None = None) -> None:
        import faiss  # type: ignore
        import numpy as np
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._np = np
        self.encoder = SentenceTransformer(modelo)
        textos = [f"{d['titulo']} — {d['secao']}\n{d['conteudo']}" for d in documentos]

        cache_npy = Path(cache) / "embeddings.npy" if cache else None
        if cache_npy and cache_npy.exists():
            vetores = np.load(cache_npy)
            logger.info("Embeddings carregados do cache %s", cache_npy)
        else:
            vetores = self.encoder.encode(textos, normalize_embeddings=True,
                                          show_progress_bar=False).astype("float32")
            if cache_npy:
                cache_npy.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_npy, vetores)

        self.indice = faiss.IndexFlatIP(vetores.shape[1])  # vetores normalizados => IP = cosseno
        self.indice.add(vetores)

    def buscar(self, consulta: str, k: int) -> list[tuple[int, float]]:
        vetor = self.encoder.encode([consulta], normalize_embeddings=True).astype("float32")
        scores, indices = self.indice.search(vetor, k)
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]


# --------------------------------------------------------------------------------------
# Retriever
# --------------------------------------------------------------------------------------

class RetrieverProtocolos:
    def __init__(self, documentos: list[dict], backend: Backend) -> None:
        self.documentos = documentos
        self.backend = backend

    @classmethod
    def construir(cls, config=CONFIG, forcar_bm25: bool = False) -> "RetrieverProtocolos":
        caminho = Path(config.documentos_rag)
        if not caminho.exists():
            raise FileNotFoundError(
                f"{caminho} não existe. Rode: python datagen/build_finetune_dataset.py"
            )
        documentos = [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]

        if not forcar_bm25:
            try:
                backend: Backend = BackendVetorial(documentos, config.embedding_model,
                                                   config.vectorstore_path)
                logger.info("Retriever: backend vetorial (%s), %d trechos.",
                            config.embedding_model, len(documentos))
                return cls(documentos, backend)
            except ImportError:
                logger.warning("faiss/sentence-transformers indisponíveis — usando BM25.")
            except Exception as erro:  # pragma: no cover
                logger.warning("Backend vetorial falhou (%s) — usando BM25.", erro)

        logger.info("Retriever: backend BM25, %d trechos.", len(documentos))
        return cls(documentos, BackendBM25(documentos))

    def buscar(self, consulta: str, k: int | None = None, score_minimo: float = 0.0) -> list[Trecho]:
        k = k or CONFIG.top_k
        resultados = self.backend.buscar(consulta, k)
        trechos = []
        for indice, score in resultados:
            if score < score_minimo:
                continue
            doc = self.documentos[indice]
            trechos.append(
                Trecho(protocolo=doc["protocolo"], versao=doc["versao"], titulo=doc["titulo"],
                       secao=doc["secao"], conteudo=doc["conteudo"], score=score)
            )
        return trechos

    def montar_contexto(self, trechos: list[Trecho], limite_caracteres: int = 2600) -> str:
        blocos, total = [], 0
        for t in trechos:
            bloco = f"[{t.citacao()}]\n{t.conteudo}"
            if total + len(bloco) > limite_caracteres:
                break
            blocos.append(bloco)
            total += len(bloco)
        return "\n\n".join(blocos)
