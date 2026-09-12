"""
Fine-tuning supervisionado (SFT) com LoRA — FASE 3.

Ajusta um LLM aberto aos dados internos do Hospital Aurora: protocolos, FAQs de médicos,
modelos de laudo/receita/procedimento e respostas contextualizadas com o prontuário.

Por que LoRA e não fine-tuning completo:
  * o corpus institucional é pequeno (centenas de amostras) — ajuste completo levaria a
    esquecimento catastrófico do conhecimento linguístico do modelo base;
  * treina ~0,5% dos parâmetros, o que cabe na GPU T4 gratuita do Colab;
  * o adapter resultante tem poucas dezenas de MB e pode ser versionado no repositório,
    ao lado do modelo base baixado do Hugging Face.

Uso:
    python finetune/train_lora.py --config finetune/configs/lora_tinyllama.yaml
    python finetune/train_lora.py --config ... --epocas 1 --smoke-test   # validação rápida
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.data_module import ColadorCausal, ler_jsonl, preparar_dataset  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("finetune.train")

BASE_DIR = Path(__file__).resolve().parents[1]


def carregar_config(caminho: Path) -> dict:
    with Path(caminho).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def escolher_dtype(preferido: str):
    """bfloat16 exige Ampere+. Em T4 (Turing) o fallback para float16 é obrigatório."""
    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA indisponível — treino em CPU será MUITO lento. Use o Colab.")
        return torch.float32
    if preferido == "bfloat16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if preferido == "bfloat16":
        logger.info("GPU sem suporte a bfloat16 (provável T4) — usando float16.")
    return torch.float16


def hash_adapter(adapter_dir: Path) -> str:
    """Hash do adapter, gravado no log de auditoria de toda inferência (PROT-007 §5)."""
    h = hashlib.sha256()
    for arquivo in sorted(adapter_dir.rglob("*")):
        if arquivo.is_file():
            h.update(arquivo.name.encode())
            h.update(arquivo.read_bytes())
    return h.hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA do assistente clínico.")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "finetune/configs/lora_tinyllama.yaml")
    parser.add_argument("--base-model", type=str, default=None, help="Sobrescreve modelo.base.")
    parser.add_argument("--epocas", type=float, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Usa 16 amostras e 1 época — só para validar o pipeline.")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, set_seed)

    cfg = carregar_config(args.config)
    m_cfg, l_cfg, t_cfg, d_cfg, s_cfg = (cfg["modelo"], cfg["lora"], cfg["treino"],
                                         cfg["dados"], cfg["saida"])

    base_model = args.base_model or os.getenv("BASE_MODEL") or m_cfg["base"]
    out_dir = Path(args.out or s_cfg["dir"])
    adapter_dir = Path(s_cfg["adapter_dir"]) if args.out is None else out_dir / "adapter"
    epocas = args.epocas if args.epocas is not None else t_cfg["epocas"]

    set_seed(t_cfg["seed"])
    dtype = escolher_dtype(m_cfg.get("dtype", "bfloat16"))
    logger.info("Modelo base: %s | dtype: %s", base_model, dtype)

    # ---- tokenizer -------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=os.getenv("HF_TOKEN") or None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- dados -----------------------------------------------------------------------
    treino_raw = ler_jsonl(BASE_DIR / d_cfg["train"])
    val_raw = ler_jsonl(BASE_DIR / d_cfg["val"])
    if args.smoke_test:
        treino_raw, val_raw, epocas = treino_raw[:16], val_raw[:4], 1
        logger.warning("SMOKE TEST ativo: 16 amostras de treino, 1 época.")

    treino = preparar_dataset(tokenizer, treino_raw, m_cfg["max_seq_length"],
                              d_cfg.get("treinar_apenas_na_resposta", True))
    val = preparar_dataset(tokenizer, val_raw, m_cfg["max_seq_length"],
                           d_cfg.get("treinar_apenas_na_resposta", True))
    logger.info("Amostras: treino=%d | validação=%d", len(treino), len(val))

    # ---- modelo ----------------------------------------------------------------------
    kwargs: dict = {"torch_dtype": dtype, "token": os.getenv("HF_TOKEN") or None}
    if m_cfg.get("carregar_em_4bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else None

    modelo = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    modelo.config.use_cache = False  # incompatível com gradient checkpointing

    if m_cfg.get("carregar_em_4bit"):
        from peft import prepare_model_for_kbit_training

        modelo = prepare_model_for_kbit_training(
            modelo, use_gradient_checkpointing=t_cfg.get("gradient_checkpointing", True)
        )

    lora = LoraConfig(
        r=l_cfg["r"],
        lora_alpha=l_cfg["alpha"],
        lora_dropout=l_cfg["dropout"],
        bias=l_cfg["bias"],
        task_type=l_cfg["task_type"],
        target_modules=l_cfg["target_modules"],
    )
    modelo = get_peft_model(modelo, lora)
    treinaveis = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    total = sum(p.numel() for p in modelo.parameters())
    logger.info("Parâmetros treináveis: %s de %s (%.3f%%)",
                f"{treinaveis:,}", f"{total:,}", 100 * treinaveis / total)

    # ---- treino ----------------------------------------------------------------------
    argumentos = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epocas,
        per_device_train_batch_size=t_cfg["batch_size"],
        per_device_eval_batch_size=t_cfg["batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=float(t_cfg["learning_rate"]),
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        weight_decay=t_cfg["weight_decay"],
        logging_steps=t_cfg["logging_steps"],
        save_strategy=t_cfg["save_strategy"],
        save_total_limit=t_cfg["save_total_limit"],
        gradient_checkpointing=t_cfg.get("gradient_checkpointing", True),
        optim=t_cfg.get("optim", "adamw_torch"),
        seed=t_cfg["seed"],
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        report_to=[],
        **_eval_kwargs(t_cfg["eval_strategy"]),
    )

    trainer = Trainer(
        model=modelo,
        args=argumentos,
        train_dataset=treino,
        eval_dataset=val,
        data_collator=ColadorCausal(pad_token_id=tokenizer.pad_token_id),
    )

    inicio = datetime.now(timezone.utc)
    resultado = trainer.train()
    metricas_eval = trainer.evaluate()

    # ---- persistência ----------------------------------------------------------------
    adapter_dir.mkdir(parents=True, exist_ok=True)
    modelo.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metadados = {
        "modelo_base": base_model,
        "treinado_em_utc": inicio.isoformat(),
        "duracao_s": round((datetime.now(timezone.utc) - inicio).total_seconds(), 1),
        "epocas": epocas,
        "amostras_treino": len(treino),
        "amostras_val": len(val),
        "parametros_treinaveis": treinaveis,
        "parametros_totais": total,
        "percentual_treinavel": round(100 * treinaveis / total, 4),
        "train_loss": round(float(resultado.training_loss), 4),
        "eval_loss": round(float(metricas_eval.get("eval_loss", float("nan"))), 4),
        "perplexidade_val": round(float(torch.exp(torch.tensor(metricas_eval["eval_loss"]))), 3)
        if "eval_loss" in metricas_eval else None,
        "lora": dict(l_cfg),
        "adapter_hash": hash_adapter(adapter_dir),
    }
    (adapter_dir / "metadados_treino.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Adapter salvo em %s", adapter_dir)
    logger.info("Métricas: %s", json.dumps(
        {k: metadados[k] for k in ("train_loss", "eval_loss", "perplexidade_val", "adapter_hash")},
        ensure_ascii=False))


def _eval_kwargs(estrategia: str) -> dict:
    """`evaluation_strategy` virou `eval_strategy` no transformers 4.46 — suportamos os dois."""
    import inspect

    from transformers import TrainingArguments

    parametros = inspect.signature(TrainingArguments.__init__).parameters
    chave = "eval_strategy" if "eval_strategy" in parametros else "evaluation_strategy"
    return {chave: estrategia}


if __name__ == "__main__":
    main()
