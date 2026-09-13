"""
Interface de demonstração do Assistente Clínico — FASE 3.

    streamlit run app/streamlit_app.py

Quatro abas, refletindo as quatro exigências do enunciado:
  * **Assistente** — pergunta clínica contextualizada com o prontuário, resposta com fontes.
  * **Painel da paciente** — dossiê, exames pendentes e alertas por severidade.
  * **Validação humana** — fila `pending_human_validation` e registro da decisão médica.
  * **Auditoria** — trilha completa, nó a nó, da interação selecionada.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant.config import CONFIG  # noqa: E402
from assistant.graph import AssistenteClinico  # noqa: E402

st.set_page_config(page_title="Assistente Clínico — Hospital Aurora", page_icon="🩺",
                   layout="wide")

CORES_SEVERIDADE = {"CRITICO": "🔴", "ALTO": "🟠", "MEDIO": "🟡", "BAIXO": "🔵"}

PERGUNTAS_EXEMPLO = [
    "Quais critérios o hospital usa para diagnosticar SOP?",
    "Quais exames estão pendentes e o que isso muda na conduta?",
    "A paciente tem indicação de investigação endometrial?",
    "O que falta para autorizar indução de ovulação?",
    "Para qual serviço devo encaminhar considerando o IMC?",
    "Me passe a receita com a dose de metformina.",          # deve ser bloqueada
    "Qual o melhor plano de saúde para ela contratar?",       # fora de escopo
]


@st.cache_resource(show_spinner="Carregando assistente (LLM, RAG e base clínica)...")
def carregar_assistente() -> AssistenteClinico:
    return AssistenteClinico()


def bloco_fontes(fontes: list[dict]) -> None:
    if not fontes:
        st.warning("Nenhum trecho de protocolo foi recuperado para esta pergunta.")
        return
    st.caption("Fontes recuperadas (explicabilidade — PROT-007 §4)")
    for fonte in fontes:
        with st.expander(f"{fonte['protocolo']} v{fonte['versao']} — {fonte['secao']}  "
                         f"(score {fonte['score']})"):
            st.write(fonte["trecho"])


def aba_assistente(assistente: AssistenteClinico, paciente_id: str | None, crm: str) -> None:
    st.subheader("Consulta ao assistente")

    exemplo = st.selectbox("Perguntas de exemplo", ["(escrever a minha)"] + PERGUNTAS_EXEMPLO)
    padrao = "" if exemplo == "(escrever a minha)" else exemplo
    pergunta = st.text_area("Pergunta clínica", value=padrao, height=90,
                            placeholder="Ex.: quando devo repetir o TOTG nesta paciente?")

    if st.button("Perguntar", type="primary", disabled=not pergunta.strip()):
        with st.spinner("Executando o fluxo LangGraph..."):
            resultado = assistente.responder(pergunta, paciente_id, usuario_crm=crm)
        st.session_state["ultimo_resultado"] = resultado

    resultado = st.session_state.get("ultimo_resultado")
    if not resultado:
        return

    if resultado["bloqueado"]:
        st.error(f"**Resposta bloqueada pelo guardrail — intenção `{resultado['intencao']}`**")
    st.markdown(resultado["resposta"])

    colunas = st.columns(4)
    colunas[0].metric("Confiança", resultado["nivel_de_confianca"]["nivel"].upper())
    colunas[1].metric("Fontes", len(resultado["fontes"]))
    colunas[2].metric("Validação humana", "exigida" if resultado["requer_validacao_humana"] else "não")
    colunas[3].metric("Latência", f"{resultado['latencia_ms']:.0f} ms")
    st.caption(resultado["nivel_de_confianca"]["justificativa"])

    bloco_fontes(resultado["fontes"])

    with st.expander("Caminho percorrido no grafo e guardrails acionados"):
        st.code(" → ".join(resultado["trace"]), language="text")
        st.write("**Guardrails:**", resultado["guardrails_acionados"] or "nenhum")
        st.write("**Dados do paciente usados:**", resultado["dados_do_paciente_usados"] or "nenhum")
        st.write("**Modelo:**", resultado["modelo"])
        st.write("**interaction_id:**", resultado["interaction_id"])


def aba_paciente(assistente: AssistenteClinico, paciente_id: str | None) -> None:
    if not paciente_id:
        st.info("Selecione uma paciente na barra lateral.")
        return

    dossie = assistente.repositorio.dossie(paciente_id)
    if not dossie:
        st.error("Paciente não encontrada.")
        return

    paciente = dossie["paciente"]
    colunas = st.columns(5)
    colunas[0].metric("IMC", f"{paciente['imc']} kg/m²")
    colunas[1].metric("PA", f"{paciente['pa_sistolica']}/{paciente['pa_diastolica']}")
    colunas[2].metric("Ferriman-Gallwey", paciente["ferriman_gallwey"])
    colunas[3].metric("Sangramentos/12m", paciente["sangramentos_12m"])
    colunas[4].metric("Exames pendentes", len(dossie["exames_pendentes"]))

    st.subheader("Alertas ativos")
    if dossie["alertas"]:
        for alerta in dossie["alertas"]:
            icone = CORES_SEVERIDADE.get(alerta["severidade"], "⚪")
            st.write(f"{icone} **{alerta['severidade']}** — {alerta['gatilho']} · "
                     f"{alerta['detalhe']}  \n`{alerta['protocolo']}`")
    else:
        st.success("Nenhum gatilho da matriz de alertas está ativo.")

    esquerda, direita = st.columns(2)
    with esquerda:
        st.subheader("Exames pendentes")
        st.dataframe(dossie["exames_pendentes"], use_container_width=True, hide_index=True)
        st.subheader("Painel SOP faltante")
        st.write(", ".join(dossie["painel_sop_faltante"]) or "completo")
    with direita:
        st.subheader("Últimos resultados liberados")
        liberados = [
            {k: e[k] for k in ("nome_exame", "valor", "unidade", "referencia", "data_resultado")}
            for e in dossie["exames"] if e["status"] == "liberado"
        ][:12]
        st.dataframe(liberados, use_container_width=True, hide_index=True)

    if paciente["desejo_gestacional"]:
        st.subheader("Checklist pré-indução (PROT-004 §2)")
        pendentes = dossie["pendencias_pre_inducao"]
        st.write("\n".join(f"- ❌ {p}" for p in pendentes) if pendentes else "✅ Completo.")

    st.subheader("Consultas recentes")
    st.dataframe(dossie["consultas"], use_container_width=True, hide_index=True)


def aba_validacao(assistente: AssistenteClinico, crm: str) -> None:
    st.subheader("Fila de validação humana (PROT-007 §2)")
    pendentes = assistente.fila_pendente()
    if not pendentes:
        st.success("Nenhuma sugestão aguardando validação.")
    for registro in reversed(pendentes):
        with st.expander(f"{registro['interaction_id']} · "
                         f"{registro.get('paciente_pseudonimo') or 'sem paciente'} · "
                         f"{registro['pergunta'][:70]}"):
            st.markdown(registro["resposta"])
            st.caption("Fontes: " + (", ".join(
                f"{f['protocolo']} — {f['secao']}" for f in registro["fontes"]) or "nenhuma"))
            decisao = st.radio("Decisão", ["aprovado", "aprovado_com_ressalva", "rejeitado"],
                               horizontal=True, key=f"d-{registro['interaction_id']}")
            justificativa = st.text_input("Justificativa",
                                          key=f"j-{registro['interaction_id']}")
            if st.button("Registrar decisão", key=f"b-{registro['interaction_id']}"):
                assistente.validar(registro["interaction_id"], crm, decisao, justificativa)
                st.success("Decisão registrada na trilha de auditoria.")
                st.rerun()

    validadas = assistente.validacoes()
    if validadas:
        st.subheader("Decisões registradas")
        st.dataframe(validadas[::-1], use_container_width=True, hide_index=True)


def aba_auditoria(assistente: AssistenteClinico) -> None:
    st.subheader("Trilha de auditoria (PROT-007 §5)")
    st.caption(f"Arquivo: `{assistente.auditor.arquivo}` — JSONL append-only, sem PII.")

    eventos = assistente.auditor.ler_eventos(limite=500)
    if not eventos:
        st.info("Ainda não há eventos registrados hoje.")
        return

    ids = list(dict.fromkeys(e["interaction_id"] for e in eventos))[::-1]
    escolhido = st.selectbox("Interação", ["(todas)"] + ids)
    filtrados = eventos if escolhido == "(todas)" else [
        e for e in eventos if e["interaction_id"] == escolhido]

    st.dataframe(
        [{"timestamp": e["timestamp_utc"], "nó": e["no_do_grafo"], "crm": e["usuario_crm"],
          "paciente": e["paciente_pseudonimo"], "severidade": e["severidade_alerta"],
          "guardrails": ", ".join(e["guardrails_acionados"]), "latência_ms": e["latencia_ms"]}
         for e in filtrados[::-1]],
        use_container_width=True, hide_index=True,
    )

    if escolhido != "(todas)":
        with st.expander("Eventos completos (JSON)"):
            st.code(json.dumps(filtrados, ensure_ascii=False, indent=2), language="json")


def main() -> None:
    st.title("🩺 Assistente Clínico — Hospital Aurora")
    st.caption("Tech Challenge FASE 3 · Jeferson Verdan · Pós-graduação em IA para Devs (FIAP) — "
               "dados 100% sintéticos")

    assistente = carregar_assistente()

    with st.sidebar:
        st.header("Sessão")
        crm = st.text_input("CRM do usuário", value="CRM-RJ 123456")
        pacientes = assistente.repositorio.listar_pacientes(60)
        opcoes = ["(sem paciente)"] + [p["paciente_id"] for p in pacientes]
        escolhido = st.selectbox("Paciente", opcoes)
        paciente_id = None if escolhido == "(sem paciente)" else escolhido

        st.divider()
        st.caption("**Runtime**")
        st.write("LLM:", getattr(assistente.llm, "descricao", "—"))
        st.write("RAG:", assistente.retriever.backend.nome,
                 f"({len(assistente.retriever.documentos)} trechos)")
        st.write("Base:", CONFIG.hospital_db.name)
        st.divider()
        st.caption("O assistente nunca prescreve, não fecha diagnóstico e não altera o "
                   "prontuário (PROT-007 §1). Toda conduta exige validação médica.")

    abas = st.tabs(["Assistente", "Painel da paciente", "Validação humana", "Auditoria"])
    with abas[0]:
        aba_assistente(assistente, paciente_id, crm)
    with abas[1]:
        aba_paciente(assistente, paciente_id)
    with abas[2]:
        aba_validacao(assistente, crm)
    with abas[3]:
        aba_auditoria(assistente)


if __name__ == "__main__":
    main()
