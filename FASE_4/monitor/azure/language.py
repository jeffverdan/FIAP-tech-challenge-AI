"""Azure AI Language (Text Analytics) — sentimento, frases-chave e Text Analytics for Health.

``analisar_texto(texto)`` devolve um dict normalizado:

.. code-block:: python

    {"sentimento": {"geral": "negative", "scores": {...},
                    "frases": [{"texto": ..., "sentimento": ..., "scores": {...}, "opinioes": [...]}]},
     "frases_chave": ["chest pain", ...],
     "entidades_saude": [{"texto": "breathless", "categoria": "SymptomOrSign", "confianca": 0.98,
                          "offset": 120, "negada": False, "normalizado": "Dyspnea", "codigos": [...]}],
     "relacoes_saude": [{"tipo": "DosageOfMedication", "papeis": {...}}]}

O texto é quebrado em blocos de até 5.000 caracteres (limite por documento) respeitando o fim
das frases; os offsets das entidades são remapeados para o texto original.

Fallback local: léxico bilíngue de sintomas/medicamentos com detecção de negação (NegEx
simplificado) e um léxico de sentimento — bem mais pobre, mas mantém o pipeline de pé.
"""

from __future__ import annotations

import re

from monitor.azure import cache
from monitor.config import CONFIG

LIMITE_DOC = 5000
LOTE = 5


def _blocos(texto: str, limite: int = LIMITE_DOC) -> list[tuple[int, str]]:
    """Quebra em blocos ≤ limite, cortando no fim de frase. Retorna (offset, bloco)."""
    blocos, ini = [], 0
    while ini < len(texto):
        fim = min(ini + limite, len(texto))
        if fim < len(texto):
            corte = max(texto.rfind(". ", ini, fim), texto.rfind("? ", ini, fim), texto.rfind("! ", ini, fim))
            fim = corte + 1 if corte > ini else fim
        blocos.append((ini, texto[ini:fim]))
        ini = fim
    return [(o, b) for o, b in blocos if b.strip()]


def _azure_analisar(texto: str, idioma: str) -> dict:
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    cli = TextAnalyticsClient(CONFIG.language_endpoint, AzureKeyCredential(CONFIG.language_key))
    blocos = _blocos(texto)
    docs = [b for _, b in blocos]

    frases, contagem, scores_soma = [], {}, {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    chaves: list[str] = []
    entidades, relacoes = [], []
    for i in range(0, len(docs), LOTE):
        lote, offs = docs[i:i + LOTE], [o for o, _ in blocos[i:i + LOTE]]
        for doc in cli.analyze_sentiment(lote, language=idioma, show_opinion_mining=True):
            if doc.is_error:
                raise cache.AzureIndisponivel(f"sentimento: {doc.error.code} {doc.error.message}")
            contagem[doc.sentiment] = contagem.get(doc.sentiment, 0) + 1
            for k in scores_soma:
                scores_soma[k] += getattr(doc.confidence_scores, k)
            for s in doc.sentences:
                frases.append({"texto": s.text, "sentimento": s.sentiment,
                               "scores": {k: round(getattr(s.confidence_scores, k), 3) for k in scores_soma},
                               "opinioes": [{"alvo": o.target.text, "sentimento": o.target.sentiment,
                                             "avaliacoes": [a.text for a in o.assessments]}
                                            for o in (s.mined_opinions or [])]})
        for doc in cli.extract_key_phrases(lote, language=idioma):
            if not doc.is_error:
                chaves += [k for k in doc.key_phrases if k not in chaves]
        poller = cli.begin_analyze_healthcare_entities(lote, language=idioma)
        for off, doc in zip(offs, poller.result()):
            if doc.is_error:
                raise cache.AzureIndisponivel(f"TA4H: {doc.error.code} {doc.error.message}")
            for e in doc.entities:
                assercao = e.assertion
                entidades.append({
                    "texto": e.text, "categoria": e.category, "subcategoria": e.subcategory,
                    "confianca": round(e.confidence_score, 3), "offset": off + e.offset,
                    "normalizado": e.normalized_text,
                    "negada": bool(assercao and assercao.certainty and "negative" in str(assercao.certainty).lower()),
                    "certeza": str(assercao.certainty) if assercao and assercao.certainty else None,
                    "associacao": str(assercao.association) if assercao and assercao.association else None,
                    "codigos": [f"{d.name}:{d.entity_id}" for d in (e.data_sources or [])][:5],
                })
            for r in doc.entity_relations:
                relacoes.append({"tipo": r.relation_type,
                                 "papeis": {p.name: p.entity.text for p in r.roles}})
    n = max(sum(contagem.values()), 1)
    geral = max(contagem, key=contagem.get) if contagem else "neutral"
    if contagem.get("positive") and contagem.get("negative"):
        geral = "mixed"
    return {"sentimento": {"geral": geral, "scores": {k: round(v / n, 3) for k, v in scores_soma.items()},
                           "frases": frases},
            "frases_chave": chaves, "entidades_saude": entidades, "relacoes_saude": relacoes,
            "idioma": idioma, "provedor": "azure_language"}


# --------------------------------------------------------------------------------------
# Fallback local
# --------------------------------------------------------------------------------------
LEXICO_SAUDE = {
    "SymptomOrSign": [
        "chest pain", "chest tightness", "tight chest", "chest discomfort", "short of breath", "shortness of breath",
        "breathless", "breathlessness", "difficulty breathing", "hard to breathe", "can't breathe", "wheez",
        "cough", "tired", "tiredness", "fatigue", "exhausted", "no energy", "dizzy", "dizziness", "faint",
        "palpitation", "headache", "fever", "sweat", "nausea", "vomit", "diarrh", "pain", "weakness", "numb",
        "slurred", "confus", "swelling", "rash", "blood", "can't move", "cannot move", "face is numb",
        "drooping", "lack of energy", "low energy",
        "dor no peito", "falta de ar", "cansaço", "fadiga", "tontura", "desmaio", "tosse", "febre", "fraqueza",
    ],
    "MedicationName": ["paracetamol", "ibuprofen", "inhaler", "salbutamol", "ventolin", "aspirin", "antibiotic",
                       "amoxicillin", "insulin", "metformin", "statin", "warfarin", "steroid"],
}
NEGACAO = re.compile(r"\b(no|not|never|without|denies|don't|didn't|haven't|hasn't|isn't|sem|nega|não)\b", re.I)
POSITIVO = re.compile(r"\b(good|fine|better|great|thank|ok|okay|well|happy|bem|melhor|obrigad)\w*", re.I)
NEGATIVO = re.compile(r"\b(bad|worse|awful|terrible|scared|worried|anxious|pain|hurt|can't|struggl|horrible|"
                      r"mal|pior|medo|preocupad|dor)\w*", re.I)


def _fallback_analisar(texto: str, idioma: str) -> dict:
    entidades = []
    baixo = texto.lower()
    for cat, termos in LEXICO_SAUDE.items():
        for termo in termos:
            for m in re.finditer(re.escape(termo), baixo):
                inicio_frase = max(baixo.rfind(".", 0, m.start()), baixo.rfind("?", 0, m.start())) + 1
                negada = bool(NEGACAO.search(baixo[max(inicio_frase, m.start() - 40): m.start()]))
                fim = m.end()
                while fim < len(texto) and texto[fim].isalpha():
                    fim += 1
                entidades.append({"texto": texto[m.start():fim], "categoria": cat, "subcategoria": None,
                                  "confianca": 0.5, "offset": m.start(), "normalizado": termo,
                                  "negada": negada, "certeza": "negative" if negada else None,
                                  "associacao": None, "codigos": []})
    frases = []
    for f in re.split(r"(?<=[.!?])\s+", texto):
        if not f.strip():
            continue
        pos, neg = len(POSITIVO.findall(f)), len(NEGATIVO.findall(f))
        s = "positive" if pos > neg else "negative" if neg > pos else "neutral"
        frases.append({"texto": f, "sentimento": s, "scores": {}, "opinioes": []})
    cont = {k: sum(fr["sentimento"] == k for fr in frases) for k in ("positive", "neutral", "negative")}
    n = max(len(frases), 1)
    return {"sentimento": {"geral": max(cont, key=cont.get) if frases else "neutral",
                           "scores": {k: round(v / n, 3) for k, v in cont.items()}, "frases": frases},
            "frases_chave": [], "entidades_saude": sorted(entidades, key=lambda e: e["offset"]),
            "relacoes_saude": [], "idioma": idioma, "provedor": "lexico_local"}


def analisar_texto(texto: str, idioma: str | None = None) -> dict:
    idioma = idioma or CONFIG.text_language
    k = cache.chave("language", "sentimento+frases+ta4h", texto, idioma=idioma)
    return cache.executar(
        "language", k,
        ao_vivo=lambda: _azure_analisar(texto, idioma),
        configurado=CONFIG.language_configurado,
        fallback=lambda: _fallback_analisar(texto, idioma),
        meta={"caracteres": len(texto), "endpoint": CONFIG.language_endpoint},
    )
