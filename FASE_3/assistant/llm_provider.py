"""
Provedor de LLM do assistente — FASE 3.

Duas implementações compatíveis com a interface ``BaseChatModel`` do LangChain:

``ChatLoRAHuggingFace``
    Carrega o modelo base do Hugging Face e aplica o adapter LoRA treinado na Fase 3.
    É o caminho normal de execução.

``ChatFallbackDeterministico``
    Modo offline, sem pesos e sem GPU. Redige a resposta **de forma extrativa** a partir do
    bloco CONTEXTO (trechos de protocolo recuperados pelo RAG) e do bloco FATOS (saída do
    motor determinístico de regras clínicas), aplicando o mesmo formato de citação de fonte.

Por que manter o fallback: o mesmo padrão adotado na Fase 2. Ele garante que a banca, o
professor ou qualquer integrante do grupo consiga executar o fluxo LangGraph ponta a ponta —
incluindo guardrails, alertas e auditoria — sem GPU e sem baixar 2 GB de pesos. As decisões
clínicas do sistema vêm das regras determinísticas e do RAG, não da amostragem do modelo;
o fine-tuning melhora a redação e a aderência ao formato institucional.

A seleção é automática: ``FORCE_FALLBACK=true``, ausência do adapter ou ausência de
``torch``/``transformers`` levam ao modo fallback, sempre com aviso no log.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from assistant.config import CONFIG
from assistant.prompts import MENSAGEM_SEM_FONTE

logger = logging.getLogger("assistant.llm")


# --------------------------------------------------------------------------------------
# Fallback determinístico
# --------------------------------------------------------------------------------------

class ChatFallbackDeterministico(BaseChatModel):
    """Redige extrativamente a partir dos blocos CONTEXTO/FATOS do prompt. Sem pesos."""

    max_trecho: int = 900

    @property
    def _llm_type(self) -> str:
        return "aurora-fallback-deterministico"

    @property
    def descricao(self) -> str:
        return "fallback determinístico (offline, sem pesos)"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n\n".join(str(m.content) for m in messages if m.type != "system")
        texto = self._redigir(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=texto))])

    # -- montagem da resposta ----------------------------------------------------------

    @staticmethod
    def _extrair_bloco(prompt: str, nome: str) -> str:
        padrao = re.compile(rf"^{nome}:[ \t]*\n?(.*?)(?=\n[A-ZÇÃÕÁÉÍÓÚ]{{4,}}:|\Z)",
                            re.DOTALL | re.MULTILINE)
        achado = padrao.search(prompt)
        return achado.group(1).strip() if achado else ""

    def _redigir(self, prompt: str) -> str:
        contexto = self._extrair_bloco(prompt, "CONTEXTO")
        fatos = self._extrair_bloco(prompt, "FATOS")
        paciente = self._extrair_bloco(prompt, "PACIENTE")
        pergunta = self._extrair_bloco(prompt, "PERGUNTA") or prompt.strip()

        partes: list[str] = []

        if fatos:
            partes.append(fatos)

        if contexto:
            trecho = textwrap.shorten(
                re.sub(r"\s+", " ", contexto), width=self.max_trecho, placeholder=" [...]"
            )
            prefixo = "Complemento do protocolo institucional" if fatos else "Segundo o protocolo institucional"
            partes.append(f"{prefixo}:\n{trecho}")

        if not partes:
            return MENSAGEM_SEM_FONTE

        if paciente:
            partes.append(f"Dados do prontuário considerados: {textwrap.shorten(paciente, 300, placeholder=' [...]')}")

        partes.append(
            "SUGESTÃO NÃO VALIDADA — conteúdo gerado como apoio à decisão; requer validação do "
            "médico assistente antes de qualquer conduta (PROT-007 §2)."
        )
        _ = pergunta  # a pergunta já está refletida na seleção de contexto feita pelo retriever
        return "\n\n".join(partes)


# --------------------------------------------------------------------------------------
# Modelo ajustado (LoRA)
# --------------------------------------------------------------------------------------

class ChatLoRAHuggingFace(BaseChatModel):
    """Modelo base do Hugging Face + adapter LoRA da Fase 3."""

    base_model: str
    adapter_path: str | None = None
    max_new_tokens: int = 400
    temperature: float = 0.1
    device: str = "auto"

    _tokenizer: Any = None
    _modelo: Any = None
    _adapter_hash: str = "n/a"

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **dados: Any) -> None:
        super().__init__(**dados)
        self._carregar()

    @property
    def _llm_type(self) -> str:
        return "aurora-lora-huggingface"

    @property
    def descricao(self) -> str:
        sufixo = f" + adapter {self._adapter_hash}" if self.adapter_path else " (sem adapter)"
        return f"{self.base_model}{sufixo}"

    def _carregar(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        origem_tok = self.adapter_path or self.base_model
        self._tokenizer = AutoTokenizer.from_pretrained(origem_tok)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        cuda = torch.cuda.is_available() and self.device != "cpu"
        dtype = torch.float16 if cuda else torch.float32
        self._modelo = AutoModelForCausalLM.from_pretrained(
            self.base_model, torch_dtype=dtype, device_map="auto" if cuda else None
        )

        if self.adapter_path:
            from peft import PeftModel

            self._modelo = PeftModel.from_pretrained(self._modelo, self.adapter_path)
            meta = Path(self.adapter_path) / "metadados_treino.json"
            if meta.exists():
                self._adapter_hash = json.loads(meta.read_text(encoding="utf-8")).get(
                    "adapter_hash", "n/a")
        self._modelo.eval()
        logger.info("LLM carregada: %s (cuda=%s)", self.descricao, cuda)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        import torch

        papeis = {"system": "system", "human": "user", "ai": "assistant"}
        conversa = [{"role": papeis.get(m.type, "user"), "content": str(m.content)} for m in messages]
        prompt = self._tokenizer.apply_chat_template(conversa, tokenize=False,
                                                     add_generation_prompt=True)
        entradas = self._tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        entradas = {k: v.to(self._modelo.device) for k, v in entradas.items()}

        with torch.no_grad():
            saida = self._modelo.generate(
                **entradas,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        gerado = saida[0][entradas["input_ids"].shape[1]:]
        texto = self._tokenizer.decode(gerado, skip_special_tokens=True).strip()
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=texto))])


# --------------------------------------------------------------------------------------
# Fábrica
# --------------------------------------------------------------------------------------

def carregar_llm(config=CONFIG, forcar_fallback: bool | None = None) -> BaseChatModel:
    """Escolhe a implementação disponível, degradando com aviso explícito no log."""
    if forcar_fallback if forcar_fallback is not None else config.force_fallback:
        logger.warning("FORCE_FALLBACK ativo — usando %s.",
                       ChatFallbackDeterministico().descricao)
        return ChatFallbackDeterministico()

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        logger.warning("torch/transformers indisponíveis — caindo para o fallback determinístico. "
                       "Instale com: pip install -r requirements.txt")
        return ChatFallbackDeterministico()

    adapter = Path(config.adapter_path)
    tem_adapter = (adapter / "adapter_config.json").exists()
    if not tem_adapter:
        logger.warning("Adapter LoRA não encontrado em %s — treine com "
                       "finetune/notebook_colab.ipynb. Usando fallback determinístico.", adapter)
        return ChatFallbackDeterministico()

    try:
        return ChatLoRAHuggingFace(
            base_model=config.base_model,
            adapter_path=str(adapter),
            max_new_tokens=config.max_new_tokens,
            device=config.device,
        )
    except Exception as erro:  # pragma: no cover - depende do ambiente
        logger.error("Falha ao carregar a LLM (%s) — usando fallback determinístico.", erro)
        return ChatFallbackDeterministico()


def hash_do_modelo(llm: BaseChatModel) -> str:
    return getattr(llm, "_adapter_hash", "n/a")
