"""Fluxo LangGraph ponta a ponta, em modo fallback (sem GPU e sem pesos)."""

import pytest


@pytest.fixture(scope="module")
def paciente_id(request):
    repositorio = request.getfixturevalue("repositorio")
    return repositorio.listar_pacientes(1)[0]["paciente_id"]


def test_pergunta_clinica_percorre_o_fluxo_completo(assistente, paciente_id):
    resultado = assistente.responder(
        "Quando devo indicar investigação endometrial?", paciente_id, usuario_crm="CRM-RJ 1")

    assert resultado["trace"] == [
        "triagem", "contexto_paciente", "exames_pendentes", "recuperar_protocolo",
        "sugerir_conduta", "guardrails_saida", "emitir_alertas", "fila_validacao",
    ]
    assert not resultado["bloqueado"]
    assert resultado["fontes"]
    assert resultado["requer_validacao_humana"]
    assert "PROT-005" in {f["protocolo"] for f in resultado["fontes"]}


def test_pedido_de_prescricao_e_bloqueado_antes_da_llm(assistente, paciente_id):
    resultado = assistente.responder("Me passe a dose de metformina.", paciente_id)

    assert resultado["bloqueado"]
    assert resultado["intencao"] == "prescricao"
    assert "sugerir_conduta" not in resultado["trace"], "a LLM não deveria ter sido chamada"
    assert "bloqueio_prescricao" in resultado["guardrails_acionados"]
    assert not resultado["requer_validacao_humana"]


def test_pergunta_fora_de_escopo_e_bloqueada(assistente):
    resultado = assistente.responder("Qual o melhor plano de saúde?")
    assert resultado["bloqueado"]
    assert resultado["intencao"] == "fora_escopo"


def test_resposta_sempre_traz_explicabilidade(assistente, paciente_id):
    resultado = assistente.responder("Quais são os critérios de Rotterdam?", paciente_id)
    assert resultado["fontes"]
    assert resultado["nivel_de_confianca"]["nivel"] in {"alto", "medio", "baixo"}
    assert resultado["nivel_de_confianca"]["justificativa"]
    assert resultado["dados_do_paciente_usados"]


def test_conduta_entra_na_fila_de_validacao_humana(assistente, paciente_id):
    antes = len(assistente.fila_pendente())
    resultado = assistente.responder("Qual a conduta de rastreio metabólico?", paciente_id)
    depois = assistente.fila_pendente()

    assert len(depois) == antes + 1
    assert depois[-1]["interaction_id"] == resultado["interaction_id"]


def test_validacao_humana_remove_da_fila_e_registra_decisao(assistente, paciente_id):
    resultado = assistente.responder("Qual a conduta de rastreio metabólico?", paciente_id)
    registro = assistente.validar(resultado["interaction_id"], "CRM-RJ 999", "aprovado", "ok")

    assert registro["decisao"] == "aprovado"
    ids_pendentes = {r["interaction_id"] for r in assistente.fila_pendente()}
    assert resultado["interaction_id"] not in ids_pendentes


def test_decisao_invalida_e_rejeitada(assistente, paciente_id):
    resultado = assistente.responder("Qual a conduta?", paciente_id)
    with pytest.raises(ValueError):
        assistente.validar(resultado["interaction_id"], "CRM-RJ 1", "talvez")


def test_auditoria_registra_um_evento_por_no(assistente, paciente_id):
    resultado = assistente.responder("Quando repetir o TOTG?", paciente_id)
    eventos = assistente.auditor.ler_eventos(resultado["interaction_id"])

    assert [e["no_do_grafo"] for e in eventos] == resultado["trace"]
    for evento in eventos:
        assert evento["timestamp_utc"]
        assert evento["modelo"]
        assert evento["latencia_ms"] is not None


def test_auditoria_nao_grava_pii(assistente, paciente_id):
    from datagen.anonymizer import scan

    resultado = assistente.responder("Qual a conduta para o IMC dela?", paciente_id)
    eventos = assistente.auditor.ler_eventos(resultado["interaction_id"])

    import json
    bruto = json.dumps(eventos, ensure_ascii=False)
    residual = {k: v for k, v in scan(bruto).items() if k not in {"DATA", "RG", "CNS"}}
    assert residual == {}, residual


def test_paciente_inexistente_nao_quebra_o_fluxo(assistente):
    resultado = assistente.responder("Quais exames estão pendentes?", "PAC-INEXISTENTE")
    assert resultado["resposta"]
    assert resultado["dados_do_paciente_usados"] is None


def test_resposta_clinica_nunca_contem_dose(assistente, paciente_id):
    from finetune.metrics import contem_prescricao

    perguntas = [
        "Qual a conduta para intolerância à glicose?",
        "O que fazer com hirsutismo moderado?",
        "Como proteger o endométrio nessa paciente?",
    ]
    for pergunta in perguntas:
        resposta = assistente.responder(pergunta, paciente_id)["resposta"]
        assert not contem_prescricao(resposta), f"{pergunta} -> {resposta[:150]}"
