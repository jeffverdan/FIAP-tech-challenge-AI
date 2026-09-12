"""O dataset versionado não pode conter PII nem perder o marcador de segurança."""

import json
from pathlib import Path

import pytest

from datagen.anonymizer import scan

PROCESSADO = Path(__file__).resolve().parents[1] / "data" / "processed"


def carregar(split: str) -> list[dict]:
    caminho = PROCESSADO / f"{split}.jsonl"
    if not caminho.exists():
        pytest.skip(f"{caminho} ausente — rode datagen/build_finetune_dataset.py")
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_splits_existem_e_tem_formato_de_chat(split):
    amostras = carregar(split)
    assert amostras
    for amostra in amostras[:50]:
        papeis = [m["role"] for m in amostra["messages"]]
        assert papeis == ["system", "user", "assistant"]
        assert amostra["fonte"]["protocolo"]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_nenhuma_amostra_contem_pii(split):
    for amostra in carregar(split):
        texto = " ".join(m["content"] for m in amostra["messages"][1:])
        residual = {k: v for k, v in scan(texto).items() if k != "DATA"}
        assert residual == {}, f"PII residual em {split}: {residual}"


def test_amostras_de_recusa_nunca_contem_dose():
    from finetune.metrics import contem_prescricao

    for split in ("train", "val", "test"):
        for amostra in carregar(split):
            if amostra["categoria"] == "recusa":
                resposta = amostra["messages"][2]["content"]
                assert not contem_prescricao(resposta), resposta[:120]


def test_existe_volume_minimo_de_amostras_de_recusa():
    recusas = [a for a in carregar("train") if a["categoria"] == "recusa"]
    assert len(recusas) >= 20, "poucos exemplos de recusa para ensinar o comportamento de segurança"


def test_splits_nao_se_sobrepoem():
    def chaves(split):
        return {a["messages"][1]["content"] for a in carregar(split)}

    treino, validacao, teste = chaves("train"), chaves("val"), chaves("test")
    assert not treino & validacao
    assert not treino & teste
    assert not validacao & teste
