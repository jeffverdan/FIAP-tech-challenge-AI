"""Configuração central da FASE 4 — lida de variáveis de ambiente / ``.env``.

Tudo que depende da conta Azure fica aqui e no ``.env``. Trocar de assinatura é trocar
quatro variáveis (ou rodar ``scripts/azure_setup.ps1``); nenhum outro arquivo muda.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

try:  # opcional: o projeto roda sem python-dotenv
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except Exception:  # pragma: no cover
    pass


def _caminho(nome: str, default: str) -> Path:
    valor = Path(os.getenv(nome, default))
    return valor if valor.is_absolute() else BASE_DIR / valor


#: Modos de uso dos serviços Azure.
#:  - ``live``    só Azure (erro se não houver chave)
#:  - ``auto``    cache → Azure → fallback local  (padrão)
#:  - ``replay``  só cache (demo depois que a conta expirar)
#:  - ``offline`` só fallback local (sem rede)
AZURE_MODOS = {"live", "auto", "replay", "offline"}


@dataclass
class Config:
    # ---- Azure ----
    azure_mode: str = field(default_factory=lambda: os.getenv("AZURE_MODE", "auto").strip().lower())
    speech_key: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_KEY", "").strip())
    speech_region: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_REGION", "").strip())
    speech_language: str = field(default_factory=lambda: os.getenv("AZURE_SPEECH_LANGUAGE", "en-US").strip())
    language_key: str = field(default_factory=lambda: os.getenv("AZURE_LANGUAGE_KEY", "").strip())
    language_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_LANGUAGE_ENDPOINT", "").strip())
    text_language: str = field(default_factory=lambda: os.getenv("AZURE_TEXT_LANGUAGE", "en").strip())
    azure_cache_dir: Path = field(default_factory=lambda: _caminho("AZURE_CACHE_DIR", "data/azure_cache"))

    # ---- Dados ----
    raw_dir: Path = field(default_factory=lambda: _caminho("RAW_DIR", "data/raw"))
    processed_dir: Path = field(default_factory=lambda: _caminho("PROCESSED_DIR", "data/processed"))
    results_dir: Path = field(default_factory=lambda: _caminho("RESULTS_DIR", "results"))

    # ---- Vídeo ----
    pose_backend: str = field(default_factory=lambda: os.getenv("POSE_BACKEND", "yolo").strip().lower())
    yolo_pose_weights: str = field(default_factory=lambda: os.getenv("YOLO_POSE_WEIGHTS", "yolov8n-pose.pt"))
    yolo_det_weights: str = field(default_factory=lambda: os.getenv("YOLO_DET_WEIGHTS", "yolov8n.pt"))
    video_fps_amostragem: float = field(default_factory=lambda: float(os.getenv("VIDEO_FPS", "10")))

    # ---- Alertas ----
    alert_webhook_url: str = field(default_factory=lambda: os.getenv("ALERT_WEBHOOK_URL", "").strip())
    events_dir: Path = field(default_factory=lambda: _caminho("EVENTS_DIR", "logs/events"))
    audit_dir: Path = field(default_factory=lambda: _caminho("AUDIT_DIR", "logs/audit"))

    def __post_init__(self) -> None:
        if self.azure_mode not in AZURE_MODOS:
            raise ValueError(f"AZURE_MODE inválido: {self.azure_mode!r}. Use um de {sorted(AZURE_MODOS)}")

    # -- conveniências -------------------------------------------------------------
    @property
    def speech_configurado(self) -> bool:
        return bool(self.speech_key and self.speech_region)

    @property
    def language_configurado(self) -> bool:
        return bool(self.language_key and self.language_endpoint)

    def resumo_azure(self) -> dict:
        """Resumo seguro (sem expor chaves) — usado no dashboard e no check_azure."""
        mascara = lambda k: (k[:4] + "…" + k[-2:]) if len(k) > 8 else ("(vazia)" if not k else "…")
        return {
            "modo": self.azure_mode,
            "speech": {"regiao": self.speech_region or "(vazia)", "chave": mascara(self.speech_key),
                       "idioma": self.speech_language},
            "language": {"endpoint": self.language_endpoint or "(vazio)", "chave": mascara(self.language_key),
                         "idioma": self.text_language},
            "cache": str(self.azure_cache_dir),
        }


CONFIG = Config()
