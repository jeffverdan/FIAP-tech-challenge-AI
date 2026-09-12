"""
Avaliação do **sistema completo** (grafo LangGraph + guardrails + RAG) — FASE 3.

`evaluate.py` mede o modelo isolado. Este script mede o que o médico realmente recebe: a
resposta depois do RAG, do motor de regras e das três camadas de guardrail.

A distinção importa. Um modelo base sem fine-tuning pode ter conformidade ruim e, ainda
assim, o sistema entregar 100% de conformidade — porque os guardrails não dependem do
modelo. É exatamente esse o desenho, e a avaliação precisa evidenciá-lo.

Uso:
    python finetune/evaluate_assistant.py --fallback
    python finetune/evaluate_assistant.py --limite 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger("finetune.evaluate_assistant")

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"

RE_PACIENTE = re.compile(r"\bPAC-[0-9A-F]{8}\b")
RE_PERGUNTA = re.compile(r"^PERGUNTA:\s*\n?(.*)$", re.DOTALL | re.MULTILINE)


def desmontar(amostra: dict) -> tuple[str, str | None]:
    """Recupera (pergunta, paciente_id) do bloco de usuário da amostra de teste."""
    conteudo = amostra["messages"][1]["content"]
    achado = RE_PERGUNTA.search(conteudo)
    pergunta = (achado.group(1) if achado else conteudo).strip()
    paciente = RE_PACIENTE.search(conteudo)
    return pergunta, paciente.group(0) if paciente else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o assistente completo da FASE 3.")
    parser.add_argument("--test", type=Path, default=BASE_DIR / "data/processed/test.jsonl")
    parser.add_argument("--db", default=None)
    parser.add_argument("--limite", type=int, default=0)
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()

    if args.db:
        os.environ["HOSPITAL_DB"] = args.db
    if args.fallback:
        os.environ["FORCE_FALLBACK"] = "true"

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "ERROR"),
                        format="%(levelname)-8s | %(name)s | %(message)s")

    from assistant.graph import AssistenteClinico
    from finetune import metrics
    from finetune.data_module import ler_jsonl

    amostras = ler_jsonl(args.test)
    if args.limite:
        amostras = amostras[: args.limite]

    assistente = AssistenteClinico()
    registros, bloqueios_corretos, bloqueios_esperados = [], 0, 0

    for i, amostra in enumerate(amostras, 1):
        pergunta, paciente_id = desmontar(amostra)
        resultado = assistente.responder(pergunta, paciente_id, usuario_crm="CRM-AVALIACAO")

        if amostra["categoria"] == "recusa":
            bloqueios_esperados += 1
            bloqueios_corretos += int(resultado["bloqueado"])

        registros.append(
            {
                "categoria": amostra["categoria"],
                "protocolo_esperado": amostra["fonte"]["protocolo"],
                "referencia": amostra["messages"][2]["content"],
                "hipotese": resultado["resposta"],
                "bloqueado": resultado["bloqueado"],
                "fontes_recuperadas": [f["protocolo"] for f in resultado["fontes"]],
                "confianca": resultado["nivel_de_confianca"]["nivel"],
                "latencia_ms": resultado["latencia_ms"],
            }
        )
        if i % 10 == 0:
            logger.info("%d/%d", i, len(amostras))

    agregadas = metrics.avaliar_lote(registros)
    agregadas["taxa_bloqueio_correto_em_recusa"] = (
        round(bloqueios_corretos / bloqueios_esperados, 4) if bloqueios_esperados else None)
    agregadas["latencia_mediana_ms"] = round(
        sorted(r["latencia_ms"] for r in registros)[len(registros) // 2], 1)
    agregadas["taxa_recuperacao_fonte_correta"] = round(
        sum(r["protocolo_esperado"] in r["fontes_recuperadas"] for r in registros
            if not r["bloqueado"]) / max(sum(1 for r in registros if not r["bloqueado"]), 1), 4)
    agregadas["distribuicao_confianca"] = {
        nivel: sum(1 for r in registros if r["confianca"] == nivel)
        for nivel in ("alto", "medio", "baixo")
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "avaliado_em_utc": ts,
        "modo": "fallback" if args.fallback else "modelo",
        "modelo": getattr(assistente.llm, "descricao", "—"),
        "backend_rag": assistente.retriever.backend.nome,
        "n_amostras": len(registros),
        "metricas": agregadas,
    }
    (RESULTS_DIR / f"avaliacao_sistema_{ts}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (RESULTS_DIR / f"predicoes_sistema_{ts}.jsonl").open("w", encoding="utf-8") as fh:
        for registro in registros:
            fh.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("Resultados em %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
