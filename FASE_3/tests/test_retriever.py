"""O RAG é a base da explicabilidade: precisa trazer o protocolo certo."""

import pytest


@pytest.mark.parametrize("consulta,protocolo_esperado", [
    ("Qual o corte de Ferriman-Gallwey para hirsutismo?", "PROT-003"),
    ("Com que frequência repetir o TOTG?", "PROT-002"),
    ("Quando indicar biópsia endometrial?", "PROT-005"),
    ("O que falta antes de autorizar indução de ovulação?", "PROT-004"),
    ("Para qual serviço encaminho paciente com IMC 32?", "PROT-006"),
    ("Quais critérios de Rotterdam o hospital adota?", "PROT-001"),
    ("Como colher o painel SOP no laboratório?", "MOD-PROC-001"),
])
def test_recupera_o_protocolo_correto_no_topo(retriever, consulta, protocolo_esperado):
    trechos = retriever.buscar(consulta, k=3)
    assert trechos, "nenhum trecho recuperado"
    assert protocolo_esperado in {t.protocolo for t in trechos}


def test_trecho_carrega_metadados_de_fonte(retriever):
    fonte = retriever.buscar("critérios de Rotterdam", k=1)[0].como_fonte()
    for chave in ("protocolo", "versao", "secao", "trecho", "score"):
        assert chave in fonte
    assert fonte["trecho"]


def test_consulta_sem_relacao_nao_traz_trechos(retriever):
    assert retriever.buscar("qual o melhor time de futebol carioca", k=3) == []


def test_contexto_respeita_o_limite_de_caracteres(retriever):
    trechos = retriever.buscar("critérios diagnósticos de SOP", k=6)
    contexto = retriever.montar_contexto(trechos, limite_caracteres=600)
    assert len(contexto) <= 700  # limite + cabeçalho do último bloco aceito


def test_backend_bm25_indexa_todos_os_trechos(retriever):
    assert len(retriever.documentos) >= 40
