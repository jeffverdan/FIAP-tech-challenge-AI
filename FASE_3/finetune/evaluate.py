"""
Avaliação do modelo ajustado — FASE 3.

Compara o **modelo base** com o **modelo base + adapter LoRA** no split de teste, em duas
dimensões:

  1. Qualidade textual — ROUGE-L contra a resposta de referência e perplexidade no split.
  2. Conformidade de segurança (PROT-007) — citação de fonte, fonte correta, recusa de
     prescrição, ausência de dose na saída e marcação de validação humana.

Uso:
    python finetune/evaluate.py --adapter finetune/outputs/adapter --comparar-base
    python finetune/evaluate.py --adapter ... --limite 20     # amostra rápida para o vídeo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune import metrics  # noqa: E402
from finetune.data_module import ler_jsonl  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("finetune.evaluate")

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"


def carregar(base_model: str, adapter: Path | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    origem = str(adapter) if adapter else base_model
    tokenizer = AutoTokenizer.from_pretrained(origem, token=os.getenv("HF_TOKEN") or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    modelo = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        token=os.getenv("HF_TOKEN") or None,
    )
    if adapter:
        from peft import PeftModel

        modelo = PeftModel.from_pretrained(modelo, str(adapter))
        logger.info("Adapter carregado de %s", adapter)
    modelo.eval()
    return modelo, tokenizer


def gerar(modelo, tokenizer, mensagens: list[dict], max_new_tokens: int = 320) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(mensagens, tokenize=False, add_generation_prompt=True)
    entradas = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(modelo.device)
    with torch.no_grad():
        saida = modelo.generate(
            **entradas,
            max_new_tokens=max_new_tokens,
            do_sample=False,              # determinístico: avaliação reprodutível
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    gerado = saida[0][entradas["input_ids"].shape[1]:]
    return tokenizer.decode(gerado, skip_special_tokens=True).strip()


def perplexidade(modelo, tokenizer, amostras: list[dict], max_seq_length: int = 1024) -> float:
    import torch

    from finetune.data_module import tokenizar

    total_loss, total_tokens = 0.0, 0
    for ex in amostras:
        item = tokenizar(tokenizer, ex, max_seq_length, treinar_apenas_na_resposta=True)
        ids = torch.tensor([item["input_ids"]], device=modelo.device)
        labels = torch.tensor([item["labels"]], device=modelo.device)
        n = int((labels != -100).sum())
        if n == 0:
            continue
        with torch.no_grad():
            loss = modelo(input_ids=ids, labels=labels).loss
        total_loss += float(loss) * n
        total_tokens += n
    return float(torch.exp(torch.tensor(total_loss / max(total_tokens, 1))))


def avaliar_modelo(modelo, tokenizer, amostras: list[dict], rotulo: str,
                   max_new_tokens: int) -> tuple[dict, list[dict]]:
    resultados = []
    for i, ex in enumerate(amostras, 1):
        mensagens = ex["messages"][:-1]
        referencia = ex["messages"][-1]["content"]
        hipotese = gerar(modelo, tokenizer, mensagens, max_new_tokens)
        resultados.append(
            {
                "categoria": ex["categoria"],
                "protocolo_esperado": ex["fonte"]["protocolo"],
                "pergunta": mensagens[-1]["content"],
                "referencia": referencia,
                "hipotese": hipotese,
            }
        )
        if i % 5 == 0:
            logger.info("[%s] %d/%d amostras geradas", rotulo, i, len(amostras))

    agregadas = metrics.avaliar_lote(resultados)
    agregadas["perplexidade"] = round(perplexidade(modelo, tokenizer, amostras), 3)
    return agregadas, resultados


def tabela_markdown(base: dict | None, tunado: dict) -> str:
    chaves = [
        ("rouge_l_medio", "ROUGE-L médio", "maior é melhor"),
        ("perplexidade", "Perplexidade", "menor é melhor"),
        ("taxa_citacao_fonte", "Cita fonte", "maior é melhor"),
        ("taxa_fonte_correta", "Fonte correta", "maior é melhor"),
        ("taxa_recusa_correta", "Recusa corretamente", "maior é melhor"),
        ("taxa_vazamento_prescricao", "Vazamento de prescrição", "menor é melhor"),
        ("taxa_prescricao_em_recusa", "Prescreve mesmo devendo recusar", "menor é melhor"),
        ("taxa_marcacao_validacao_humana", "Marca validação humana", "maior é melhor"),
    ]
    linhas = ["| Métrica | Base | Fine-tuned | Direção |", "| --- | --- | --- | --- |"]
    for chave, rotulo, direcao in chaves:
        v_base = base.get(chave) if base else None
        v_tun = tunado.get(chave)
        if v_base is None and v_tun is None:
            continue
        linhas.append(
            f"| {rotulo} | {'—' if v_base is None else v_base} | "
            f"{'—' if v_tun is None else v_tun} | {direcao} |"
        )
    return "\n".join(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description="Avalia o modelo ajustado da FASE 3.")
    parser.add_argument("--base-model", default=os.getenv("BASE_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--adapter", type=Path, default=BASE_DIR / "finetune/outputs/adapter")
    parser.add_argument("--test", type=Path, default=BASE_DIR / "data/processed/test.jsonl")
    parser.add_argument("--limite", type=int, default=0, help="0 = split completo.")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--comparar-base", action="store_true",
                        help="Avalia também o modelo base, sem adapter.")
    args = parser.parse_args()

    amostras = ler_jsonl(args.test)
    if args.limite:
        amostras = amostras[: args.limite]
    logger.info("Avaliando %d amostras do split de teste.", len(amostras))

    metricas_base, detalhes_base = None, None
    if args.comparar_base:
        modelo, tokenizer = carregar(args.base_model, None)
        metricas_base, detalhes_base = avaliar_modelo(modelo, tokenizer, amostras, "base",
                                                      args.max_new_tokens)
        del modelo

    modelo, tokenizer = carregar(args.base_model, args.adapter)
    metricas_tunado, detalhes_tunado = avaliar_modelo(modelo, tokenizer, amostras, "fine-tuned",
                                                      args.max_new_tokens)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "avaliado_em_utc": ts,
        "modelo_base": args.base_model,
        "adapter": str(args.adapter),
        "n_amostras": len(amostras),
        "metricas_base": metricas_base,
        "metricas_fine_tuned": metricas_tunado,
    }
    (RESULTS_DIR / f"avaliacao_{ts}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with (RESULTS_DIR / f"predicoes_{ts}.jsonl").open("w", encoding="utf-8") as fh:
        for i, det in enumerate(detalhes_tunado):
            registro = dict(det)
            if detalhes_base:
                registro["hipotese_base"] = detalhes_base[i]["hipotese"]
            fh.write(json.dumps(registro, ensure_ascii=False) + "\n")

    tabela = tabela_markdown(metricas_base, metricas_tunado)
    (RESULTS_DIR / f"avaliacao_{ts}.md").write_text(
        f"# Avaliação do modelo — {ts}\n\n"
        f"- Modelo base: `{args.base_model}`\n- Adapter: `{args.adapter}`\n"
        f"- Amostras: {len(amostras)}\n\n{tabela}\n", encoding="utf-8")

    print("\n" + tabela + "\n")
    logger.info("Resultados em %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
