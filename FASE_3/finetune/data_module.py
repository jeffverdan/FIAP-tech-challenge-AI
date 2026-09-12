"""
Preparação dos dados para o SFT — FASE 3.

Responsabilidade única: transformar as amostras em formato *chat messages* nos tensores
que o `Trainer` consome, com **mascaramento do prompt**.

Por que mascarar o prompt: se a loss for calculada sobre o system prompt e a pergunta, boa
parte do gradiente é gasta reproduzindo texto que o modelo nunca precisa gerar. Mascarando
(labels = -100 nas posições do prompt), o sinal de treino se concentra na resposta clínica —
que é o comportamento que queremos ajustar.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


def ler_jsonl(caminho: str | Path) -> list[dict]:
    caminho = Path(caminho)
    with caminho.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def montar_textos(tokenizer: Any, exemplo: dict) -> tuple[str, str]:
    """Devolve (texto_completo, texto_do_prompt) já com o chat template do modelo aplicado."""
    mensagens = exemplo["messages"]
    completo = tokenizer.apply_chat_template(mensagens, tokenize=False)
    prompt = tokenizer.apply_chat_template(mensagens[:-1], tokenize=False, add_generation_prompt=True)
    return completo, prompt


def tokenizar(
    tokenizer: Any,
    exemplo: dict,
    max_seq_length: int,
    treinar_apenas_na_resposta: bool = True,
) -> dict[str, list[int]]:
    completo, prompt = montar_textos(tokenizer, exemplo)

    ids = tokenizer(completo, truncation=True, max_length=max_seq_length,
                    add_special_tokens=False)["input_ids"]
    labels = list(ids)

    if treinar_apenas_na_resposta:
        n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for i in range(min(n_prompt, len(labels))):
            labels[i] = IGNORE_INDEX

    return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def preparar_dataset(tokenizer: Any, amostras: Iterable[dict], max_seq_length: int,
                     treinar_apenas_na_resposta: bool = True) -> list[dict]:
    saida, truncadas = [], 0
    for ex in amostras:
        item = tokenizar(tokenizer, ex, max_seq_length, treinar_apenas_na_resposta)
        if len(item["input_ids"]) >= max_seq_length:
            truncadas += 1
        # descarta amostras em que a resposta inteira foi cortada pela truncagem
        if all(l == IGNORE_INDEX for l in item["labels"]):
            continue
        saida.append(item)
    if truncadas:
        logger.warning("%d de %d amostras atingiram max_seq_length=%d.",
                       truncadas, len(saida), max_seq_length)
    return saida


@dataclass
class ColadorCausal:
    """Padding dinâmico por batch. Evita o custo de padronizar tudo em max_seq_length."""

    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        import torch

        maior = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention = [], [], []
        for f in features:
            faltam = maior - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * faltam)
            labels.append(f["labels"] + [IGNORE_INDEX] * faltam)
            attention.append(f["attention_mask"] + [0] * faltam)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
