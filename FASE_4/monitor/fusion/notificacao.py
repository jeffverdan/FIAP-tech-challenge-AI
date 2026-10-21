"""Entrega dos alertas à equipe e trilha de auditoria.

Canais:
  * **painel** — o dashboard Streamlit lê ``logs/alerts/*.jsonl``;
  * **webhook** — se ``ALERT_WEBHOOK_URL`` estiver no ``.env``, cada alerta ≥ URGENTE é enviado
    como JSON compatível com *incoming webhooks* do Microsoft Teams/Slack (``{"text": ...}``).
    Um Logic App ou Azure Function pode receber o mesmo payload e encaminhar por SMS/e-mail
    (Azure Communication Services) — fora do escopo do código, mas o contrato é este;
  * **auditoria** — JSONL *append-only* com cada evento e cada alerta (quem, quando, por quê),
    sem dados identificáveis, no mesmo padrão da FASE 3.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from monitor.config import CONFIG
from monitor.events import Severidade, _json_default
from monitor.fusion.motor import Alerta

log = logging.getLogger("monitor.alertas")
_LOCK = threading.Lock()


def _append(arq: Path, registro: dict) -> None:
    arq.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(arq, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False, default=_json_default) + "\n")


def auditar(tipo: str, dados: dict, sessao: str = "padrao") -> None:
    dia = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    _append(CONFIG.audit_dir / f"aurora4-{dia}.jsonl",
            {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "sessao": sessao, "tipo": tipo, **dados})


def texto_alerta(a: Alerta) -> str:
    linhas = [f"[{a.severidade.rotulo.upper()}] {a.titulo}", f"Paciente: {a.paciente_id} · t={a.t / 60:.1f} min",
              "Evidências:"] + [f"  • {j}" for j in a.justificativa] + \
             [f"Ação sugerida: {a.acao_sugerida}", f"Para: {', '.join(a.destinatarios)} ({a.prazo})"]
    return "\n".join(linhas)


def enviar_webhook(a: Alerta, url: str | None = None, timeout: float = 5.0) -> bool:
    url = url or CONFIG.alert_webhook_url
    if not url:
        return False
    import requests

    try:
        r = requests.post(url, json={"text": texto_alerta(a), "alerta": a.to_dict()}, timeout=timeout)
        return r.ok
    except requests.RequestException as e:   # notificação nunca derruba o monitor
        log.warning("webhook falhou: %s", e)
        return False


class Notificador:
    def __init__(self, sessao: str = "padrao", webhook_min: Severidade = Severidade.URGENTE) -> None:
        self.sessao = sessao
        self.webhook_min = webhook_min
        self.arquivo = CONFIG.events_dir.parent / "alerts" / f"{sessao}.jsonl"

    def entregar(self, alerta: Alerta) -> dict:
        canais = ["painel"]
        _append(self.arquivo, alerta.to_dict())
        if alerta.severidade >= self.webhook_min and enviar_webhook(alerta):
            canais.append("webhook")
        auditar("alerta", {"alerta": alerta.to_dict(), "canais": canais}, self.sessao)
        return {"id": alerta.id, "canais": canais}
