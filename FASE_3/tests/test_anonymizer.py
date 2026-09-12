"""Testes da camada de anonimização (requisito de preprocessing do enunciado)."""

import pytest

from datagen.anonymizer import Anonymizer, assert_clean, faixa_etaria, scan

TEXTO_COM_PII = (
    "Paciente Maria Aparecida Souza, CPF 123.456.789-00, CNS 123 4567 8901 2345, "
    "telefone (21) 98877-6655, e-mail maria@exemplo.com, nascida em 12/03/1994, "
    "residente na Rua das Acácias, 120, CEP 25000-000."
)


def test_redacao_remove_todos_os_tipos_de_pii():
    anon = Anonymizer(name_blocklist=["Maria Aparecida Souza"])
    limpo = anon.redact(TEXTO_COM_PII)

    for tipo in ("CPF", "CNS", "TELEFONE", "EMAIL", "CEP", "ENDERECO", "NOME"):
        assert f"[{tipo}]" in limpo, f"{tipo} não foi redigido"
    assert "Maria" not in limpo
    assert "123.456.789-00" not in limpo


def test_scan_nao_encontra_pii_em_texto_ja_redigido():
    anon = Anonymizer(name_blocklist=["Maria Aparecida Souza"])
    limpo = anon.redact(TEXTO_COM_PII)
    restante = {k: v for k, v in scan(limpo).items() if k != "DATA"}
    assert restante == {}


def test_pseudonimo_e_estavel_e_irreversivel():
    a1 = Anonymizer(salt="sal-fixo")
    a2 = Anonymizer(salt="sal-fixo")
    assert a1.pseudonymize("Maria Souza") == a2.pseudonymize("Maria Souza")
    assert a1.pseudonymize("Maria Souza") != a1.pseudonymize("Ana Souza")
    assert "maria" not in a1.pseudonymize("Maria Souza").lower()


def test_pseudonimo_muda_com_o_sal():
    assert Anonymizer(salt="a").pseudonymize("X") != Anonymizer(salt="b").pseudonymize("X")


def test_registro_anonimizado_nao_carrega_identificadores_diretos():
    anon = Anonymizer(name_blocklist=["Ana Lima"])
    registro = anon.anonymize_record(
        {"id": "1", "nome": "Ana Lima", "cpf": "111.222.333-44", "telefone": "(21) 90000-0000",
         "email": "ana@x.com", "endereco": "Rua A, 1", "data_nascimento": "01/01/1990",
         "idade": 33, "imc": 27.4}
    )
    for proibido in ("nome", "cpf", "telefone", "email", "endereco", "data_nascimento"):
        assert proibido not in registro
    assert registro["paciente_pseudonimo"].startswith("PAC-")
    assert registro["faixa_etaria"] == "30-34"
    assert registro["imc"] == 27.4


@pytest.mark.parametrize("idade,esperado", [(14, "<15"), (16, "15-19"), (33, "30-34"), (51, "50+")])
def test_faixa_etaria(idade, esperado):
    assert faixa_etaria(idade) == esperado


def test_assert_clean_levanta_erro_com_pii():
    with pytest.raises(ValueError):
        assert_clean("meu CPF é 123.456.789-00")
