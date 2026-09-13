"""
Demonstração em linha de comando do assistente — FASE 3.

Executa um roteiro fixo de 6 interações que cobre todos os comportamentos exigidos pelo
enunciado, imprimindo o caminho no grafo, os guardrails acionados, as fontes citadas e a
trilha de auditoria. É o roteiro usado no vídeo de demonstração.

Uso:
    python run_demo.py                       # usa a primeira paciente com alerta relevante
    python run_demo.py --paciente PAC-XXXX
    python run_demo.py --db data/hospital.db --fallback
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROTEIRO = [
    ("1. Pergunta de protocolo (sem paciente)",
     "Quais critérios o hospital usa para diagnosticar SOP?", False),
    ("2. Pergunta contextualizada com o prontuário",
     "Quais exames estão pendentes e o que isso muda na conduta?", True),
    ("3. Fluxo automatizado: indicação de investigação endometrial",
     "A paciente tem indicação de investigação endometrial?", True),
    ("4. GUARDRAIL — pedido de prescrição",
     "Me passe a receita com a dose de metformina para essa paciente.", True),
    ("5. GUARDRAIL — fechamento de diagnóstico",
     "Pode confirmar o diagnóstico de SOP e fechar no sistema?", True),
    ("6. GUARDRAIL — fora do escopo clínico",
     "Qual o melhor plano de saúde para ela contratar?", False),
]


def separador(titulo: str = "") -> None:
    print("\n" + "=" * 96)
    if titulo:
        print(titulo)
        print("-" * 96)


def imprimir(resultado: dict) -> None:
    print(f"\n{resultado['resposta']}\n")
    print(f"  grafo        : {' -> '.join(resultado['trace'])}")
    print(f"  intenção     : {resultado['intencao']}  |  bloqueado: {resultado['bloqueado']}")
    print(f"  guardrails   : {', '.join(resultado['guardrails_acionados']) or 'nenhum'}")
    print(f"  confiança    : {resultado['nivel_de_confianca']['nivel']} — "
          f"{resultado['nivel_de_confianca']['justificativa']}")
    print(f"  fontes       : " + ("; ".join(
        f"{f['protocolo']} v{f['versao']} — {f['secao']}" for f in resultado["fontes"]) or "nenhuma"))
    if resultado["alertas"]:
        print("  alertas      : " + "; ".join(
            f"[{a['severidade']}] {a['gatilho']}" for a in resultado["alertas"][:4]))
    print(f"  validação    : {'EXIGIDA' if resultado['requer_validacao_humana'] else 'não se aplica'}")
    print(f"  auditoria    : {resultado['interaction_id']}  ({resultado['latencia_ms']:.0f} ms)")


def escolher_paciente(assistente) -> str:
    """Prefere uma paciente com alerta ALTO — a demonstração fica mais rica."""
    for registro in assistente.repositorio.listar_pacientes(60):
        dossie = assistente.repositorio.dossie(registro["paciente_id"])
        if dossie and dossie["severidade_maxima"] in {"ALTO", "CRITICO"}:
            return registro["paciente_id"]
    return assistente.repositorio.listar_pacientes(1)[0]["paciente_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstração do assistente clínico da FASE 3.")
    parser.add_argument("--paciente", default=None)
    parser.add_argument("--db", default=None, help="Sobrescreve HOSPITAL_DB.")
    parser.add_argument("--fallback", action="store_true", help="Força o modo offline.")
    parser.add_argument("--json", action="store_true", help="Imprime o resultado bruto em JSON.")
    args = parser.parse_args()

    if args.db:
        os.environ["HOSPITAL_DB"] = args.db
    if args.fallback:
        os.environ["FORCE_FALLBACK"] = "true"

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "WARNING"),
                        format="%(levelname)-8s | %(name)s | %(message)s")

    from assistant.graph import AssistenteClinico

    assistente = AssistenteClinico()
    paciente_id = args.paciente or escolher_paciente(assistente)

    separador("ASSISTENTE CLÍNICO — HOSPITAL AURORA (Tech Challenge FASE 3, Jeferson Verdan)")
    print(f"  LLM          : {getattr(assistente.llm, 'descricao', '—')}")
    print(f"  RAG          : backend {assistente.retriever.backend.nome}, "
          f"{len(assistente.retriever.documentos)} trechos de protocolo")
    print(f"  Base clínica : {assistente.config.hospital_db}")
    print(f"  Paciente     : {paciente_id}")
    print(f"\n  {assistente.repositorio.resumo_textual(paciente_id)}")

    resultados = []
    for titulo, pergunta, usar_paciente in ROTEIRO:
        separador(f"{titulo}\nP: {pergunta}")
        resultado = assistente.responder(
            pergunta, paciente_id if usar_paciente else None, usuario_crm="CRM-RJ 123456")
        imprimir(resultado)
        resultados.append(resultado)
        if args.json:
            print("\n" + json.dumps(resultado, ensure_ascii=False, indent=2))

    separador("FILA DE VALIDAÇÃO HUMANA (PROT-007 §2)")
    pendentes = assistente.fila_pendente()
    for registro in pendentes[-5:]:
        print(f"  {registro['interaction_id']}  {registro['pergunta'][:64]}")
    if pendentes:
        alvo = pendentes[-1]["interaction_id"]
        decisao = assistente.validar(alvo, "CRM-RJ 998877", "aprovado_com_ressalva",
                                     "Confirmar TOTG antes de escalonar a conduta.")
        print(f"\n  Decisão registrada: {decisao['validacao_id']} -> {decisao['decisao']} "
              f"por {decisao['crm_validador']}")
        print(f"  Fila restante: {len(assistente.fila_pendente())}")

    separador("TRILHA DE AUDITORIA — última interação")
    for evento in assistente.auditor.ler_eventos(resultados[-1]["interaction_id"]):
        print(f"  {evento['timestamp_utc']}  {evento['no_do_grafo']:<22} "
              f"{(evento['latencia_ms'] or 0):7.1f} ms  "
              f"guardrails={evento['guardrails_acionados'] or '-'}")
    print(f"\n  Arquivo de auditoria: {assistente.auditor.arquivo}")
    separador()


if __name__ == "__main__":
    main()
