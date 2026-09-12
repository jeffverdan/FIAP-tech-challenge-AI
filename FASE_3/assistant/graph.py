"""
Fluxo de decisão automatizado com LangGraph — FASE 3.

    START
      v
    triagem ─────────────(bloqueado)────────────┐
      v (segue)                                 |
    contexto_paciente ───(alerta CRÍTICO)───────┤
      v                                         |
    exames_pendentes                            |
      v                                         |
    recuperar_protocolo                         |
      v                                         |
    sugerir_conduta   (única chamada à LLM)     |
      v                                         |
    guardrails_saida                            |
      v                                         v
    emitir_alertas <────────────────────── resposta_bloqueada
      v
    fila_validacao
      v
     END

Cada nó grava um evento na trilha de auditoria, então a resposta final vem acompanhada do
caminho percorrido — é assim que o sistema explica *como* chegou ao que disse, e não apenas
*o que* disse.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from assistant import guardrails as gr
from assistant.audit import AuditLogger, novo_interaction_id
from assistant.chains import formatar_blocos, montar_cadeia_clinica, prompt_renderizado
from assistant.config import CONFIG
from assistant.db_tools import RepositorioClinico
from assistant.llm_provider import carregar_llm
from assistant.retriever import RetrieverProtocolos

logger = logging.getLogger("assistant.graph")


class EstadoClinico(TypedDict, total=False):
    # entrada
    pergunta: str
    paciente_id: str | None
    usuario_crm: str
    interaction_id: str
    # trabalho
    intencao: str
    bloqueado: bool
    dossie: dict[str, Any] | None
    resumo_paciente: str | None
    fatos: str
    contexto: str
    fontes: list[dict]
    alertas: list[dict]
    severidade: str | None
    # saída
    resposta: str
    guardrails_acionados: list[str]
    requer_validacao_humana: bool
    trace: list[str]


class AssistenteClinico:
    """Encapsula LLM, RAG, repositório, guardrails, auditoria e o grafo."""

    def __init__(
        self,
        llm=None,
        retriever: RetrieverProtocolos | None = None,
        repositorio: RepositorioClinico | None = None,
        auditor: AuditLogger | None = None,
        config=CONFIG,
    ) -> None:
        self.config = config
        self.llm = llm or carregar_llm(config)
        self.retriever = retriever or RetrieverProtocolos.construir(config)
        self.repositorio = repositorio or RepositorioClinico(config.hospital_db)
        self.auditor = auditor or AuditLogger(
            config.audit_log_dir,
            adapter_hash=getattr(self.llm, "_adapter_hash", "n/a"),
            modelo=getattr(self.llm, "descricao", type(self.llm).__name__),
        )
        self.cadeia = montar_cadeia_clinica(self.llm)
        # A fila de validação e as decisões fazem parte da trilha de auditoria: seguem o
        # diretório do auditor, e não a configuração global. Isso mantém cada instância
        # (produção, demonstração, teste) com seu próprio estado de governança.
        self.fila_path = self.auditor.diretorio / "pending_human_validation.jsonl"
        self.validacoes_path = self.auditor.diretorio / "validacoes.jsonl"
        self.grafo = self._construir_grafo()

    # -- nós ---------------------------------------------------------------------------

    def _no_triagem(self, estado: EstadoClinico) -> EstadoClinico:
        inicio = time.perf_counter()
        resultado = gr.avaliar_entrada(estado["pergunta"])
        saida = {
            "intencao": resultado.intencao.value,
            "bloqueado": resultado.bloqueado,
            "resposta": resultado.mensagem or "",
            "guardrails_acionados": list(resultado.acionados),
            "trace": estado.get("trace", []) + ["triagem"],
        }
        self.auditor.registrar(
            estado["interaction_id"], "triagem",
            entrada={"pergunta": estado["pergunta"], "paciente_id": estado.get("paciente_id")},
            saida={"intencao": saida["intencao"], "bloqueado": saida["bloqueado"]},
            usuario_crm=estado.get("usuario_crm", "n/a"),
            paciente_pseudonimo=estado.get("paciente_id"),
            guardrails_acionados=saida["guardrails_acionados"],
            latencia_ms=(time.perf_counter() - inicio) * 1000,
        )
        return saida

    def _no_contexto_paciente(self, estado: EstadoClinico) -> EstadoClinico:
        inicio = time.perf_counter()
        paciente_id = estado.get("paciente_id")
        dossie = self.repositorio.dossie(paciente_id) if paciente_id else None

        saida: EstadoClinico = {
            "dossie": dossie,
            "resumo_paciente": self.repositorio.resumo_textual(paciente_id) if dossie else None,
            "alertas": dossie["alertas"] if dossie else [],
            "severidade": dossie["severidade_maxima"] if dossie else None,
            "trace": estado.get("trace", []) + ["contexto_paciente"],
            "guardrails_acionados": list(estado.get("guardrails_acionados", [])),
        }

        # Reavalia a triagem agora que a criticidade do prontuário é conhecida.
        if saida["severidade"] == "CRITICO":
            reavaliacao = gr.avaliar_entrada(estado["pergunta"], severidade_paciente="CRITICO")
            saida.update(
                intencao=reavaliacao.intencao.value,
                bloqueado=True,
                resposta=reavaliacao.mensagem or "",
                guardrails_acionados=saida["guardrails_acionados"] + reavaliacao.acionados,
            )

        self.auditor.registrar(
            estado["interaction_id"], "contexto_paciente",
            entrada={"paciente_id": paciente_id},
            saida={"encontrado": dossie is not None,
                   "n_alertas": len(saida["alertas"]),
                   "severidade": saida["severidade"]},
            usuario_crm=estado.get("usuario_crm", "n/a"),
            paciente_pseudonimo=paciente_id,
            severidade_alerta=saida["severidade"],
            guardrails_acionados=saida["guardrails_acionados"],
            latencia_ms=(time.perf_counter() - inicio) * 1000,
        )
        return saida

    def _no_exames_pendentes(self, estado: EstadoClinico) -> EstadoClinico:
        inicio = time.perf_counter()
        dossie = estado.get("dossie")
        linhas: list[str] = []

        if dossie:
            pendentes = dossie["exames_pendentes"]
            if pendentes:
                linhas.append("Exames sem resultado liberado: " + "; ".join(
                    f"{p['nome_exame']} ({p['dias_em_aberto']}d"
                    f"{', ATRASADO' if p['atrasado'] else ''})" for p in pendentes[:8]))
            else:
                linhas.append("Nenhum exame pendente.")
            if dossie["painel_sop_faltante"]:
                linhas.append("Itens do Painel SOP sem resultado: "
                              + ", ".join(dossie["painel_sop_faltante"]))
            if dossie["pendencias_pre_inducao"] and dossie["paciente"]["desejo_gestacional"]:
                linhas.append("Pendências do checklist pré-indução (PROT-004 §2): "
                              + "; ".join(dossie["pendencias_pre_inducao"]))
            if dossie["alertas"]:
                linhas.append("Alertas ativos: " + "; ".join(
                    f"[{a['severidade']}] {a['gatilho']}" for a in dossie["alertas"][:6]))

        saida = {"fatos": "\n".join(linhas),
                 "trace": estado.get("trace", []) + ["exames_pendentes"]}
        self.auditor.registrar(
            estado["interaction_id"], "exames_pendentes",
            saida={"n_linhas_de_fato": len(linhas)},
            usuario_crm=estado.get("usuario_crm", "n/a"),
            paciente_pseudonimo=estado.get("paciente_id"),
            latencia_ms=(time.perf_counter() - inicio) * 1000,
        )
        return saida

    def _no_recuperar_protocolo(self, estado: EstadoClinico) -> EstadoClinico:
        inicio = time.perf_counter()
        trechos = self.retriever.buscar(estado["pergunta"], self.config.top_k)
        fontes = [t.como_fonte() for t in trechos]
        contexto = self.retriever.montar_contexto(trechos)

        saida = {"fontes": fontes, "contexto": contexto,
                 "trace": estado.get("trace", []) + ["recuperar_protocolo"]}
        self.auditor.registrar(
            estado["interaction_id"], "recuperar_protocolo",
            entrada={"consulta": estado["pergunta"], "top_k": self.config.top_k},
            saida={"n_trechos": len(fontes), "backend": self.retriever.backend.nome},
            usuario_crm=estado.get("usuario_crm", "n/a"),
            paciente_pseudonimo=estado.get("paciente_id"),
            fontes=fontes,
            latencia_ms=(time.perf_counter() - inicio) * 1000,
        )
        return saida

    def _no_sugerir_conduta(self, estado: EstadoClinico) -> EstadoClinico:
        inicio = time.perf_counter()
        entradas = formatar_blocos(
            pergunta=estado["pergunta"],
            contexto=estado.get("contexto", ""),
            resumo_paciente=estado.get("resumo_paciente"),
            fatos=estado.get("fatos"),
        )
        try:
            resposta = self.cadeia.invoke(entradas)
        except Exception as erro:  # pragma: no cover
            logger.exception("Falha na cadeia LLM")
            resposta = f"Falha ao gerar a resposta ({erro}). Consulte diretamente o protocolo citado."

        saida = {"resposta": resposta,
                 "trace": estado.get("trace", []) + ["sugerir_conduta"]}
        self.auditor.registrar(
            estado["interaction_id"], "sugerir_conduta",
            entrada={"prompt": prompt_renderizado(entradas)},
            saida={"resposta_bruta": resposta},
            usuario_crm=estado.get("usuario_crm", "n/a"),
            paciente_pseudonimo=estado.get("paciente_id"),
            fontes=estado.get("fontes", []),
            latencia_ms=(time.perf_counter() - inicio) * 1000,
        )
        return saida

    def _no_guardrails_saida(self, estado: EstadoClinico) -> EstadoClinico:
        inicio = time.perf_counter()
        intencao = gr.Intencao(estado.get("intencao", gr.Intencao.CLINICA.value))
        resposta, acionados = gr.avaliar_saida(estado.get("resposta", ""),
                                               estado.get("fontes", []), intencao)
        saida = {
            "resposta": resposta,
            "guardrails_acionados": list(estado.get("guardrails_acionados", [])) + acionados,
            "requer_validacao_humana": intencao is gr.Intencao.CLINICA,
            "trace": estado.get("trace", []) + ["guardrails_saida"],
        }
        self.auditor.registrar(
            estado["interaction_id"], "guardrails_saida",
            saida={"resposta_final": resposta},
            usuario_crm=estado.get("usuario_crm", "n/a"),
            paciente_pseudonimo=estado.get("paciente_id"),
            fontes=estado.get("fontes", []),
            guardrails_acionados=saida["guardrails_acionados"],
            latencia_ms=(time.perf_counter() - inicio) * 1000,
        )
        return saida

    def _no_resposta_bloqueada(self, estado: EstadoClinico) -> EstadoClinico:
        inicio = time.perf_counter()
        saida = {
            "resposta": estado.get("resposta", ""),
            "requer_validacao_humana": False,
            "trace": estado.get("trace", []) + ["resposta_bloqueada"],
        }
        self.auditor.registrar(
            estado["interaction_id"], "resposta_bloqueada",
            saida={"resposta_final": saida["resposta"], "intencao": estado.get("intencao")},
            usuario_crm=estado.get("usuario_crm", "n/a"),
            paciente_pseudonimo=estado.get("paciente_id"),
            guardrails_acionados=estado.get("guardrails_acionados", []),
            severidade_alerta=estado.get("severidade"),
            latencia_ms=(time.perf_counter() - inicio) * 1000,
        )
        return saida

    def _no_emitir_alertas(self, estado: EstadoClinico) -> EstadoClinico:
        inicio = time.perf_counter()
        alertas = estado.get("alertas", [])
        criticos = [a for a in alertas if a["severidade"] in {"CRITICO", "ALTO"}]
        for alerta in criticos:
            logger.warning("ALERTA %s | paciente %s | %s",
                           alerta["severidade"], estado.get("paciente_id"), alerta["gatilho"])
        self.auditor.registrar(
            estado["interaction_id"], "emitir_alertas",
            saida={"emitidos": criticos, "total": len(alertas)},
            usuario_crm=estado.get("usuario_crm", "n/a"),
            paciente_pseudonimo=estado.get("paciente_id"),
            severidade_alerta=estado.get("severidade"),
            latencia_ms=(time.perf_counter() - inicio) * 1000,
        )
        return {"trace": estado.get("trace", []) + ["emitir_alertas"]}

    def _no_fila_validacao(self, estado: EstadoClinico) -> EstadoClinico:
        inicio = time.perf_counter()
        if estado.get("requer_validacao_humana"):
            registro = {
                "interaction_id": estado["interaction_id"],
                "criado_em_utc": datetime.now(timezone.utc).isoformat(),
                "paciente_pseudonimo": estado.get("paciente_id"),
                "usuario_crm": estado.get("usuario_crm", "n/a"),
                "pergunta": estado["pergunta"],
                "resposta": estado.get("resposta", ""),
                "fontes": estado.get("fontes", []),
                "status": "pendente",
            }
            self.fila_path.parent.mkdir(parents=True, exist_ok=True)
            with self.fila_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(registro, ensure_ascii=False) + "\n")

        self.auditor.registrar(
            estado["interaction_id"], "fila_validacao",
            saida={"enfileirado": bool(estado.get("requer_validacao_humana"))},
            usuario_crm=estado.get("usuario_crm", "n/a"),
            paciente_pseudonimo=estado.get("paciente_id"),
            latencia_ms=(time.perf_counter() - inicio) * 1000,
        )
        return {"trace": estado.get("trace", []) + ["fila_validacao"]}

    # -- arestas condicionais ----------------------------------------------------------

    @staticmethod
    def _rota_triagem(estado: EstadoClinico) -> str:
        return "bloqueado" if estado.get("bloqueado") else "segue"

    @staticmethod
    def _rota_criticidade(estado: EstadoClinico) -> str:
        return "bloqueado" if estado.get("bloqueado") else "segue"

    # -- montagem ----------------------------------------------------------------------

    def _construir_grafo(self):
        grafo = StateGraph(EstadoClinico)

        grafo.add_node("triagem", self._no_triagem)
        grafo.add_node("contexto_paciente", self._no_contexto_paciente)
        grafo.add_node("exames_pendentes", self._no_exames_pendentes)
        grafo.add_node("recuperar_protocolo", self._no_recuperar_protocolo)
        grafo.add_node("sugerir_conduta", self._no_sugerir_conduta)
        grafo.add_node("guardrails_saida", self._no_guardrails_saida)
        grafo.add_node("resposta_bloqueada", self._no_resposta_bloqueada)
        grafo.add_node("emitir_alertas", self._no_emitir_alertas)
        grafo.add_node("fila_validacao", self._no_fila_validacao)

        grafo.add_edge(START, "triagem")
        grafo.add_conditional_edges(
            "triagem", self._rota_triagem,
            {"bloqueado": "resposta_bloqueada", "segue": "contexto_paciente"},
        )
        grafo.add_conditional_edges(
            "contexto_paciente", self._rota_criticidade,
            {"bloqueado": "resposta_bloqueada", "segue": "exames_pendentes"},
        )
        grafo.add_edge("exames_pendentes", "recuperar_protocolo")
        grafo.add_edge("recuperar_protocolo", "sugerir_conduta")
        grafo.add_edge("sugerir_conduta", "guardrails_saida")
        grafo.add_edge("guardrails_saida", "emitir_alertas")
        grafo.add_edge("resposta_bloqueada", "emitir_alertas")
        grafo.add_edge("emitir_alertas", "fila_validacao")
        grafo.add_edge("fila_validacao", END)

        return grafo.compile()

    # -- API pública -------------------------------------------------------------------

    def responder(self, pergunta: str, paciente_id: str | None = None,
                  usuario_crm: str = "n/a") -> dict[str, Any]:
        estado_inicial: EstadoClinico = {
            "pergunta": pergunta,
            "paciente_id": paciente_id,
            "usuario_crm": usuario_crm,
            "interaction_id": novo_interaction_id(),
            "trace": [],
            "guardrails_acionados": [],
            "fontes": [],
            "alertas": [],
        }
        inicio = time.perf_counter()
        final = self.grafo.invoke(estado_inicial)
        return {
            "interaction_id": final["interaction_id"],
            "resposta": final.get("resposta", ""),
            "intencao": final.get("intencao"),
            "bloqueado": bool(final.get("bloqueado")),
            "fontes": final.get("fontes", []),
            "dados_do_paciente_usados": final.get("resumo_paciente"),
            "alertas": final.get("alertas", []),
            "severidade": final.get("severidade"),
            "guardrails_acionados": final.get("guardrails_acionados", []),
            "requer_validacao_humana": bool(final.get("requer_validacao_humana")),
            "nivel_de_confianca": self._confianca(final),
            "trace": final.get("trace", []),
            "latencia_ms": round((time.perf_counter() - inicio) * 1000, 1),
            "modelo": getattr(self.llm, "descricao", type(self.llm).__name__),
        }

    @staticmethod
    def _confianca(estado: EstadoClinico) -> dict[str, str]:
        """PROT-007 §4 exige nível de confiança com justificativa."""
        fontes = estado.get("fontes", [])
        if not fontes:
            return {"nivel": "baixo", "justificativa": "Nenhum trecho de protocolo recuperado."}
        melhor = max(f.get("score", 0) for f in fontes)
        tem_paciente = estado.get("dossie") is not None
        if melhor >= 5 and tem_paciente:
            return {"nivel": "alto",
                    "justificativa": f"{len(fontes)} trechos recuperados (score máx. {melhor:.2f}) "
                                     "e prontuário da paciente carregado."}
        if melhor >= 2:
            return {"nivel": "medio",
                    "justificativa": f"{len(fontes)} trechos recuperados (score máx. {melhor:.2f})"
                                     + ("." if tem_paciente else "; sem contexto de paciente.")}
        return {"nivel": "baixo",
                "justificativa": f"Aderência fraca ao protocolo (score máx. {melhor:.2f})."}

    # -- fila de validação humana ------------------------------------------------------

    def fila_pendente(self, limite: int = 50) -> list[dict]:
        if not self.fila_path.exists():
            return []
        registros = [json.loads(l) for l in self.fila_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        decididos = {v["interaction_id"] for v in self.validacoes()}
        return [r for r in registros if r["interaction_id"] not in decididos][-limite:]

    def validacoes(self) -> list[dict]:
        if not self.validacoes_path.exists():
            return []
        return [json.loads(l) for l in self.validacoes_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def validar(self, interaction_id: str, crm_validador: str, decisao: str,
                justificativa: str = "") -> dict:
        """Registra a decisão humana (PROT-007 §2). Decisões possíveis: aprovado,
        aprovado_com_ressalva, rejeitado."""
        if decisao not in {"aprovado", "aprovado_com_ressalva", "rejeitado"}:
            raise ValueError(f"Decisão inválida: {decisao}")
        registro = {
            "validacao_id": f"VAL-{interaction_id[-8:]}",
            "interaction_id": interaction_id,
            "crm_validador": crm_validador,
            "decisao": decisao,
            "justificativa": justificativa,
            "criado_em_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.validacoes_path.parent.mkdir(parents=True, exist_ok=True)
        with self.validacoes_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
        self.auditor.registrar(interaction_id, "validacao_humana", saida=registro,
                               usuario_crm=crm_validador)
        return registro
