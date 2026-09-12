"""A base clínica é somente leitura — o PROT-007 §1 depende disso ser verdade no código."""

import sqlite3

import pytest


def test_conexao_e_somente_leitura(repositorio):
    conexao = sqlite3.connect(f"file:{repositorio.caminho}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conexao.execute("DELETE FROM pacientes")


def test_repositorio_nao_expoe_metodo_de_escrita(repositorio):
    proibidos = {"inserir", "atualizar", "remover", "salvar", "executar_sql", "gravar"}
    assert not proibidos & set(dir(repositorio))


def test_base_inexistente_falha_explicitamente(tmp_path):
    from assistant.db_tools import RepositorioClinico

    with pytest.raises(FileNotFoundError, match="generate_patients"):
        RepositorioClinico(tmp_path / "nao_existe.db")


def test_dossie_reune_tudo_que_o_grafo_precisa(repositorio):
    paciente_id = repositorio.listar_pacientes(1)[0]["paciente_id"]
    dossie = repositorio.dossie(paciente_id)

    for chave in ("paciente", "exames", "prescricoes", "consultas", "exames_pendentes",
                  "painel_sop_faltante", "alertas", "severidade_maxima",
                  "pendencias_pre_inducao"):
        assert chave in dossie


def test_dossie_de_paciente_inexistente_devolve_none(repositorio):
    assert repositorio.dossie("PAC-NAOEXISTE") is None


def test_resumo_textual_nao_contem_identificadores_diretos(repositorio):
    from datagen.anonymizer import scan

    paciente_id = repositorio.listar_pacientes(1)[0]["paciente_id"]
    resumo = repositorio.resumo_textual(paciente_id)

    assert paciente_id in resumo
    assert {k: v for k, v in scan(resumo).items() if k != "DATA"} == {}


def test_tools_sao_invocaveis_pelo_langchain(repositorio, monkeypatch):
    from assistant import db_tools

    monkeypatch.setattr(db_tools, "_REPO", repositorio)
    paciente_id = repositorio.listar_pacientes(1)[0]["paciente_id"]

    assert db_tools.consultar_paciente.invoke({"paciente_id": paciente_id})["paciente_id"] == paciente_id
    assert isinstance(db_tools.listar_exames_pendentes.invoke({"paciente_id": paciente_id}), list)
    assert isinstance(db_tools.avaliar_alertas_paciente.invoke({"paciente_id": paciente_id}), list)


def test_tool_com_paciente_inexistente_devolve_erro_tratado(repositorio, monkeypatch):
    from assistant import db_tools

    monkeypatch.setattr(db_tools, "_REPO", repositorio)
    assert "erro" in db_tools.consultar_paciente.invoke({"paciente_id": "PAC-NAOEXISTE"})


def test_prescricoes_nao_armazenam_posologia(repositorio):
    """A base guarda classe terapêutica, nunca dose — o assistente não pode nem ler uma."""
    paciente_id = repositorio.listar_pacientes(1)[0]["paciente_id"]
    for prescricao in repositorio.prescricoes(paciente_id):
        assert "dose" not in prescricao
        assert "posologia" not in prescricao
