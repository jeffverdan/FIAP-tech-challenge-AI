"""Pipeline de áudio de consulta: Azure Speech → Azure Language → marcadores acústicos → eventos.

    áudio do paciente ──► Azure Speech to Text ──► texto + segmentos com tempo
                                │                        │
                                │                        └──► Azure Text Analytics
                                │                              (sintomas, medicamentos, negação,
                                │                               sentimento, frases-chave)
                                └──► marcadores acústicos nos trechos de fala do paciente
                                       (fluência, pausas, F0, jitter, shimmer, HNR, CPPS)
                                       └──► modelos de voz (disartria, respiratório, fadiga)

Os eventos seguem a regra do projeto: o que o paciente *diz* (sintoma crítico, não negado)
pode chegar a "urgente"/"crítico"; o que a *voz* sugere fica em "atenção" — é um indício
probabilístico, não um diagnóstico, e serve para a fusão confirmar com outras modalidades.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from monitor.audio import acustica
from monitor.audio.modelos import ModeloVoz
from monitor.azure import language, speech
from monitor.config import CONFIG
from monitor.events import Evento, Severidade

#: Sinais de alarme na fala do paciente → (grupo, severidade). Casam com o texto da entidade
#: do TA4H, com o nome normalizado (UMLS) ou, no fallback, com o léxico local.
SINAIS_ALARME: list[tuple[str, str, Severidade]] = [
    (r"chest (pain|tight|discomfort|pressure)|tight chest|angina|dor no peito|dor torácica", "dor_toracica", Severidade.URGENTE),
    (r"short(ness)? of breath|breathless|dyspn|difficulty breathing|hard to breathe|can'?t breathe|"
     r"falta de ar|dispneia", "dispneia", Severidade.URGENTE),
    (r"wheez|chiado|sibil", "sibilancia", Severidade.ATENCAO),
    (r"faint|syncope|passed out|blackout|desmai|síncope", "sincope", Severidade.URGENTE),
    (r"slurred|dysarthri|fala arrastada|disartria", "fala_arrastada", Severidade.URGENTE),
    (r"can'?t move|cannot move|weakness (in|on) (one|the left|the right)|hemipar|"
     r"facial droop|numb face|face (is )?(a bit )?numb|paresia", "deficit_neurologico", Severidade.CRITICO),
    (r"confus|disorient|confus[ãa]o", "confusao", Severidade.URGENTE),
    (r"cough(ing)? (up )?blood|haemoptysis|hemoptysis|hemopt", "hemoptise", Severidade.URGENTE),
    (r"suicid|kill myself|end my life|me matar", "risco_suicidio", Severidade.CRITICO),
    (r"palpitat|racing heart|heart (is )?racing|palpita", "palpitacao", Severidade.ATENCAO),
    (r"tired|fatigue|exhaust|no energy|lack of energy|malaise|letharg|cansa|fadiga|mal-estar", "fadiga", Severidade.ATENCAO),
]

#: Combinações que escalam para crítico (ex.: dor torácica + dispneia → síndrome coronariana/TEP).
COMBINACOES_CRITICAS = [({"dor_toracica", "dispneia"}, "dor torácica com dispneia")]


@dataclass
class ModelosVoz:
    disartria: ModeloVoz | None = None
    respiratorio: ModeloVoz | None = None
    fadiga: ModeloVoz | None = None

    @classmethod
    def carregar(cls, pasta: Path | None = None) -> "ModelosVoz":
        """Prefere a versão com embeddings wav2vec 2.0; sem ``torchaudio``, usa a de marcadores."""
        from monitor.audio import embeddings

        pasta = Path(pasta or CONFIG.processed_dir / "models")
        def _c(nome):
            for arq in ([pasta / f"voz_{nome}.joblib"] if embeddings.disponivel() else []) + \
                       [pasta / f"voz_{nome}_marcadores.joblib", pasta / f"voz_{nome}.joblib"]:
                if arq.exists():
                    m = ModeloVoz.carregar(arq)
                    if not m.usa_wav2vec or embeddings.disponivel():
                        return m
            return None
        return cls(_c("disartria"), _c("respiratorio"), _c("fadiga"))

    @property
    def precisa_wav2vec(self) -> bool:
        return any(m is not None and m.usa_wav2vec for m in (self.disartria, self.respiratorio, self.fadiga))


def features_completas(audio: Path, segmentos: list[tuple[float, float]] | None, wav2vec: bool) -> dict:
    """Marcadores acústicos (+ embedding wav2vec 2.0) nos trechos de fala do paciente."""
    y = acustica.carregar(audio)
    feats = acustica.features_voz(y, segmentos=segmentos or None)
    if wav2vec:
        from monitor.audio import embeddings

        emb = embeddings.embedding_segmentos(y, segmentos or None)
        feats.update({f"w2v_{i}": float(v) for i, v in enumerate(emb)})
    return feats


def sinais_de_alarme(entidades: list[dict]) -> dict[str, dict]:
    """Agrupa as entidades de saúde não negadas em sinais de alarme."""
    achados: dict[str, dict] = {}
    for e in entidades:
        if e.get("negada") or e.get("categoria") not in ("SymptomOrSign", "Diagnosis", "ConditionQualifier", None):
            continue
        alvo = f"{e.get('texto', '')} | {e.get('normalizado') or ''}".lower()
        for padrao, grupo, sev in SINAIS_ALARME:
            if re.search(padrao, alvo):
                a = achados.setdefault(grupo, {"severidade": sev, "mencoes": [], "offset": e.get("offset", 0)})
                a["mencoes"].append(e.get("texto"))
                break
    return achados


def analisar_consulta(audio_paciente: Path, paciente_id: str, t0: float = 0.0, referencia: Path | None = None,
                      modelos: ModelosVoz | None = None, idioma_fala: str | None = None,
                      idioma_texto: str | None = None, dominio: str | None = None,
                      feats: dict | None = None) -> tuple[dict, list[Evento]]:
    """Roda o pipeline completo. ``t0`` posiciona a consulta na linha do tempo do paciente.

    ``dominio`` aplica a calibração de domínio de gravação dos modelos de voz (se existir);
    ``feats`` permite reaproveitar features acústicas já calculadas."""
    from monitor.azure.cache import AzureIndisponivel

    modelos = modelos or ModelosVoz.carregar()
    try:
        stt = speech.transcrever(audio_paciente, idioma=idioma_fala, referencia=referencia)
    except AzureIndisponivel as e:   # sem STT: segue só com a análise acústica
        stt = {"texto": "", "segmentos": [], "palavras": [], "fonte": "indisponivel", "provedor": str(e)}
    nlp = language.analisar_texto(stt["texto"], idioma=idioma_texto) if stt["texto"] else \
        {"sentimento": {"geral": "neutral", "scores": {}, "frases": []}, "frases_chave": [],
         "entidades_saude": [], "relacoes_saude": [], "fonte": "sem_texto", "provedor": None}

    segs = [(s["inicio"], s["fim"]) for s in stt["segmentos"]]
    if feats is None:
        feats = features_completas(audio_paciente, segs, modelos.precisa_wav2vec)
    probs = {}
    for nome in ("disartria", "respiratorio", "fadiga"):
        m: ModeloVoz | None = getattr(modelos, nome)
        if m is not None and feats.get("duracao_s", 0) >= 1:
            probs[nome] = {"prob": round(float(m.prob(feats, dominio)[0]), 3), "limiar": round(m.limiar, 3),
                           "calibrado": bool(dominio and dominio in m.deslocamentos)}

    eventos: list[Evento] = []
    achados = sinais_de_alarme(nlp["entidades_saude"])
    fim_audio = max((s["fim"] for s in stt["segmentos"]), default=0.0)
    for grupo, a in achados.items():
        eventos.append(Evento(
            paciente_id=paciente_id, origem="audio", tipo=f"sintoma_relatado_{grupo}", severidade=a["severidade"],
            t=t0 + fim_audio, detector="azure" if nlp["fonte"].startswith("azure") else "lexico",
            descricao=f"Paciente relata {grupo.replace('_', ' ')}: " + ", ".join(sorted(set(a["mencoes"]))[:4]),
            evidencias={"mencoes": a["mencoes"][:10], "fonte_texto": nlp["fonte"], "fonte_stt": stt["fonte"]}))
    for combo, nome in COMBINACOES_CRITICAS:
        if combo <= set(achados):
            eventos.append(Evento(
                paciente_id=paciente_id, origem="audio", tipo="combinacao_critica_relatada", severidade=Severidade.CRITICO,
                t=t0 + fim_audio, detector="regra", descricao=f"Combinação de sinais de alarme relatada: {nome}",
                evidencias={"grupos": sorted(combo)}))

    frases = nlp["sentimento"]["frases"]
    neg = sum(f["sentimento"] == "negative" for f in frases) / max(len(frases), 1)
    if len(frases) >= 5 and neg >= 0.4:
        eventos.append(Evento(
            paciente_id=paciente_id, origem="texto", tipo="sentimento_negativo", severidade=Severidade.ATENCAO,
            t=t0 + fim_audio, detector="azure" if nlp["fonte"].startswith("azure") else "lexico",
            descricao=f"{neg:.0%} das falas do paciente com sentimento negativo (sofrimento/ansiedade)",
            evidencias={"proporcao_negativa": round(neg, 3), "geral": nlp["sentimento"]["geral"]}))

    rotulos = {"disartria": "padrão vocal compatível com disartria",
               "respiratorio": "padrão vocal compatível com dificuldade respiratória",
               "fadiga": "padrão vocal compatível com fadiga"}
    for nome, p in probs.items():
        if p["prob"] >= p["limiar"]:
            eventos.append(Evento(
                paciente_id=paciente_id, origem="audio", tipo=f"voz_{nome}", severidade=Severidade.ATENCAO,
                t=t0 + fim_audio, detector="ml",
                descricao=f"{rotulos[nome].capitalize()} (p={p['prob']:.0%}, limiar {p['limiar']:.0%})",
                evidencias={**p, **{k: feats.get(k) for k in acustica.MARCADORES if k in feats}}))

    relatorio = {
        "paciente_id": paciente_id, "audio": Path(audio_paciente).name,
        "stt": {"fonte": stt["fonte"], "provedor": stt.get("provedor"), "segmentos": len(stt["segmentos"]),
                "palavras": len(stt.get("palavras", [])), "texto": stt["texto"]},
        "linguagem": {"fonte": nlp["fonte"], "provedor": nlp.get("provedor"), "sentimento": nlp["sentimento"]["geral"],
                      "scores": nlp["sentimento"]["scores"], "frases_chave": nlp["frases_chave"][:15],
                      "entidades": [{k: e[k] for k in ("texto", "categoria", "negada", "normalizado")}
                                    for e in nlp["entidades_saude"]],
                      "relacoes": nlp.get("relacoes_saude", [])[:20]},
        "sinais_alarme": {g: {"severidade": int(a["severidade"]), "mencoes": a["mencoes"]} for g, a in achados.items()},
        "acustica": {k: feats.get(k) for k in acustica.MARCADORES if k in feats} | {"duracao_fala_s": feats.get("duracao_s")},
        "wav2vec2": modelos.precisa_wav2vec,
        "modelos_voz": probs,
        "eventos": [e.to_dict() for e in eventos],
    }
    return relatorio, eventos
