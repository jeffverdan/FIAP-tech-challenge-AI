"""Cache de respostas Azure — o que torna a conta *trocável* e a demo *reprodutível*.

Toda resposta de um serviço Azure é normalizada em JSON e gravada em
``data/azure_cache/<servico>/<chave>.json``. A chave é o hash do conteúdo de entrada + operação
+ idioma — **não** depende da conta, da região ou da chave de API. Consequências:

* trocar de assinatura não invalida nada do que já foi processado;
* com ``AZURE_MODE=replay`` a demo roda exatamente com as respostas reais já obtidas, mesmo
  depois que a conta expirar (ou sem internet na hora da apresentação);
* o cache vai para o git (é pequeno: JSON de texto), então quem clonar o repositório
  reproduz os resultados sem conta Azure.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from monitor.config import CONFIG

log = logging.getLogger("monitor.azure")


class AzureIndisponivel(RuntimeError):
    """Sem chave configurada, sem cache (em replay) ou serviço recusou a chamada."""


def hash_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def chave(servico: str, operacao: str, conteudo: str, **params: Any) -> str:
    base = json.dumps({"s": servico, "op": operacao, "c": conteudo, "p": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()[:24]


def _arquivo(servico: str, k: str) -> Path:
    return CONFIG.azure_cache_dir / servico / f"{k}.json"


def ler(servico: str, k: str) -> dict | None:
    arq = _arquivo(servico, k)
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    return None


def gravar(servico: str, k: str, resultado: dict, meta: dict | None = None) -> None:
    arq = _arquivo(servico, k)
    arq.parent.mkdir(parents=True, exist_ok=True)
    doc = {"_meta": {"servico": servico, "obtido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     **(meta or {})}, **resultado}
    arq.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")


def executar(servico: str, k: str, ao_vivo: Callable[[], dict], configurado: bool,
             fallback: Callable[[], dict] | None = None, meta: dict | None = None) -> dict:
    """Aplica a política de ``AZURE_MODE`` a uma chamada.

    Retorna sempre um dict com ``fonte`` ∈ {``azure``, ``azure_cache``, ``fallback_local``}.
    """
    modo = CONFIG.azure_mode
    if modo in ("auto", "replay"):
        em_cache = ler(servico, k)
        if em_cache is not None:
            em_cache["fonte"] = "azure_cache"
            return em_cache
        if modo == "replay":
            if fallback is not None:
                log.warning("%s: sem cache para %s em modo replay — usando fallback local", servico, k)
                return {**fallback(), "fonte": "fallback_local"}
            raise AzureIndisponivel(f"{servico}: sem resposta em cache ({k}) e AZURE_MODE=replay")
    if modo in ("auto", "live") and configurado:
        try:
            res = ao_vivo()
            gravar(servico, k, res, meta)
            return {**res, "fonte": "azure"}
        except Exception as e:  # noqa: BLE001 — qualquer falha do serviço cai para o fallback em "auto"
            if modo == "live" or fallback is None:
                raise
            log.warning("%s falhou (%s) — usando fallback local", servico, e)
    elif modo == "live":
        raise AzureIndisponivel(f"{servico}: AZURE_MODE=live mas o serviço não está configurado no .env")
    if fallback is None:
        raise AzureIndisponivel(f"{servico}: indisponível e sem fallback")
    return {**fallback(), "fonte": "fallback_local"}
