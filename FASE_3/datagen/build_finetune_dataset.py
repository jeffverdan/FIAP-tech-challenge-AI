"""
Curadoria e construção do dataset de fine-tuning — FASE 3.

Fontes:
  1. ``data/raw/protocolos/*.md``  — protocolos internos (seção a seção)
  2. ``data/raw/faq_medicos.jsonl`` — perguntas frequentes de médicos, com fonte
  3. ``data/raw/modelos/*.md``      — modelos de laudo, receita e procedimento
  4. ``data/hospital.db``           — prontuários anonimizados (exemplos contextualizados)
  5. (opcional) MedQuAD local       — ``--medquad data/raw/medquad.jsonl``

Etapas de preprocessing:
  * normalização de espaços e quebras de linha
  * remoção de blocos de aviso repetidos (ruído para o modelo)
  * aumento de dados por reformulação sintática das perguntas
  * portão de qualidade: nenhuma amostra pode conter PII residual (``anonymizer.assert_clean``)
  * deduplicação exata por hash do par (pergunta, resposta)
  * split estratificado por categoria em train/val/test (80/10/10)

Saída: ``data/processed/{train,val,test}.jsonl`` no formato *chat messages*, mais
``data/processed/dataset_card.md`` e ``data/processed/estatisticas.json``.

Uso:
    python datagen/build_finetune_dataset.py --seed 62
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assistant import clinical_rules as rules  # noqa: E402
from assistant.prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_COM_CONTEXTO  # noqa: E402
from datagen.anonymizer import scan  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("datagen.dataset")

BASE_DIR = Path(__file__).resolve().parents[1]
PROTOCOLOS_DIR = BASE_DIR / "data" / "raw" / "protocolos"
MODELOS_DIR = BASE_DIR / "data" / "raw" / "modelos"
FAQ_PATH = BASE_DIR / "data" / "raw" / "faq_medicos.jsonl"
DB_PATH = BASE_DIR / "data" / "hospital.db"
OUT_DIR = BASE_DIR / "data" / "processed"

# Reformulações sintáticas usadas no aumento de dados. Preservam a intenção clínica.
REFORMULACOES = [
    "{q}",
    "No Hospital Aurora, {q_min}",
    "Me lembra o protocolo: {q_min}",
    "Qual a orientação institucional sobre isto — {q_min}",
    "Preciso conferir uma conduta. {q}",
]

RUIDO = [
    re.compile(r"^>\s*Documento interno sintético.*$", re.MULTILINE),
    re.compile(r"^>\s*Não substitui diretriz oficial.*$", re.MULTILINE),
]


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------

def normalizar(texto: str) -> str:
    for padrao in RUIDO:
        texto = padrao.sub("", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def minusculizar_inicio(pergunta: str) -> str:
    p = pergunta.strip()
    return (p[0].lower() + p[1:]) if p else p


def parse_frontmatter(texto: str) -> tuple[dict, str]:
    if not texto.startswith("---"):
        return {}, texto
    _, fm, corpo = texto.split("---", 2)
    meta: dict = {}
    for linha in fm.strip().splitlines():
        if ":" not in linha:
            continue
        chave, valor = linha.split(":", 1)
        meta[chave.strip()] = valor.strip()
    return meta, corpo


def dividir_secoes(corpo: str) -> list[tuple[str, str]]:
    """Divide o markdown em (titulo_da_secao, conteudo) pelos cabeçalhos de nível 2."""
    partes = re.split(r"^##\s+", corpo, flags=re.MULTILINE)
    secoes = []
    for parte in partes[1:]:
        linhas = parte.splitlines()
        titulo = linhas[0].strip()
        conteudo = normalizar("\n".join(linhas[1:]))
        if len(conteudo) > 40:
            secoes.append((titulo, conteudo))
    return secoes


# --------------------------------------------------------------------------------------
# Construção de amostras
# --------------------------------------------------------------------------------------

def amostra(pergunta: str, resposta: str, categoria: str, fonte: dict, com_contexto: bool = False,
            contexto: str | None = None) -> dict:
    system = SYSTEM_PROMPT_COM_CONTEXTO if com_contexto else SYSTEM_PROMPT
    user = f"{contexto}\n\nPERGUNTA: {pergunta}" if contexto else pergunta
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": resposta.strip()},
        ],
        "categoria": categoria,
        "fonte": fonte,
    }


def carregar_faq(rng: random.Random) -> list[dict]:
    amostras = []
    for linha in FAQ_PATH.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        item = json.loads(linha)
        fonte = item["fonte"]
        resposta = item["resposta"]
        if item["categoria"] not in {"recusa"}:
            resposta += (
                f"\n\nFonte: {fonte['protocolo']} v{fonte['versao']} — {fonte['secao']}."
                "\nSUGESTÃO NÃO VALIDADA: requer confirmação do médico assistente."
                if item["categoria"] in {"metabolico", "hiperandrogenismo", "infertilidade", "sangramento", "estilo_de_vida"}
                else f"\n\nFonte: {fonte['protocolo']} v{fonte['versao']} — {fonte['secao']}."
            )
        else:
            resposta += f"\n\nFonte: {fonte['protocolo']} v{fonte['versao']} — {fonte['secao']}."

        variantes = rng.sample(REFORMULACOES, k=3)
        for tpl in variantes:
            pergunta = tpl.format(q=item["pergunta"], q_min=minusculizar_inicio(item["pergunta"]))
            amostras.append(amostra(pergunta, resposta, item["categoria"], fonte))
    return amostras


def carregar_protocolos() -> tuple[list[dict], list[dict]]:
    """Devolve (amostras_de_treino, documentos_para_rag)."""
    amostras, documentos = [], []
    for caminho in sorted(list(PROTOCOLOS_DIR.glob("*.md")) + list(MODELOS_DIR.glob("*.md"))):
        meta, corpo = parse_frontmatter(caminho.read_text(encoding="utf-8"))
        pid = meta.get("id", caminho.stem)
        versao = meta.get("versao", "1.0")
        titulo = meta.get("titulo", pid)

        for secao, conteudo in dividir_secoes(corpo):
            documentos.append(
                {"protocolo": pid, "versao": versao, "titulo": titulo,
                 "secao": secao, "conteudo": conteudo, "arquivo": caminho.name}
            )
            fonte = {"protocolo": pid, "versao": versao, "secao": secao}
            resposta = f"{conteudo}\n\nFonte: {pid} v{versao} — {secao}."
            for pergunta in (
                f"O que diz a seção '{secao}' do {pid}?",
                f"Resuma o que o protocolo {pid} ({titulo}) estabelece sobre {minusculizar_inicio(secao)}.",
            ):
                amostras.append(amostra(pergunta, resposta, "protocolo", fonte))
    return amostras, documentos


def carregar_contextualizadas(rng: random.Random, db_path: Path, n_pacientes: int = 25) -> list[dict]:
    """Amostras que exigem cruzar protocolo + prontuário — o comportamento-alvo do assistente."""
    if not db_path.exists():
        logger.warning("Base %s inexistente; pulando amostras contextualizadas.", db_path)
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    pacientes = [dict(r) for r in conn.execute("SELECT * FROM pacientes")]
    rng.shuffle(pacientes)
    amostras = []

    for pac in pacientes[:n_pacientes]:
        pid = pac["paciente_id"]
        exames = [dict(r) for r in conn.execute("SELECT * FROM exames WHERE paciente_id=?", (pid,))]
        prescricoes = [dict(r) for r in conn.execute("SELECT * FROM prescricoes WHERE paciente_id=?", (pid,))]

        contexto = (
            f"PACIENTE: {pid} | faixa etária {pac['faixa_etaria']} | IMC {pac['imc']} kg/m² | "
            f"PA {pac['pa_sistolica']}/{pac['pa_diastolica']} mmHg | Ferriman-Gallwey "
            f"{pac['ferriman_gallwey']} | sangramentos/12m {pac['sangramentos_12m']} | "
            f"último sangramento {pac['ultimo_sangramento']} | desejo gestacional "
            f"{'sim' if pac['desejo_gestacional'] else 'não'} | comorbidades: {pac['comorbidades']}"
        )

        # (a) exames pendentes
        pend = rules.exames_pendentes(exames)
        if pend:
            corpo = "\n".join(
                f"- {p['nome_exame']} — solicitado em {p['data_solicitacao']}, "
                f"{p['dias_em_aberto']} dias em aberto{' (ATRASADO, alerta MÉDIO)' if p['atrasado'] else ''}"
                for p in pend
            )
            resp = (f"Exames sem resultado liberado para {pid}:\n{corpo}\n\n"
                    "Pendência acima de 60 dias gera alerta de severidade MÉDIA e deve entrar na "
                    "lista de pendências da próxima consulta.\n\nFonte: PROT-007 v1.3 — 3. Matriz de alertas.")
        else:
            resp = (f"Não há exames pendentes para {pid}; todos os pedidos têm resultado liberado."
                    "\n\nFonte: PROT-007 v1.3 — 3. Matriz de alertas.")
        amostras.append(amostra(f"Quais exames estão pendentes para a paciente {pid}?", resp,
                                "contextualizada", {"protocolo": "PROT-007", "versao": "1.3",
                                                    "secao": "3. Matriz de alertas"},
                                com_contexto=True, contexto=contexto))

        # (b) painel SOP incompleto
        faltantes = rules.painel_sop_faltante(exames)
        resp_b = (
            (f"Itens do Painel SOP sem resultado liberado para {pid}: "
             + ", ".join(faltantes) + ".")
            if faltantes else
            f"O Painel SOP de {pid} está completo — todos os 11 itens têm resultado liberado."
        ) + "\n\nFonte: PROT-001 v3.2 — 4. Painel laboratorial mínimo."
        amostras.append(amostra(f"O Painel SOP da paciente {pid} está completo?", resp_b,
                                "contextualizada", {"protocolo": "PROT-001", "versao": "3.2",
                                                    "secao": "4. Painel laboratorial mínimo"},
                                com_contexto=True, contexto=contexto))

        # (c) alertas ativos
        alertas = rules.avaliar_alertas(pac, exames)
        if alertas:
            corpo = "\n".join(f"- [{a.severidade}] {a.gatilho} — {a.detalhe} ({a.protocolo})" for a in alertas)
            resp_c = f"Alertas ativos para {pid}:\n{corpo}"
        else:
            resp_c = f"Nenhum gatilho da matriz de alertas está ativo para {pid} no momento."
        resp_c += "\n\nFonte: PROT-007 v1.3 — 3. Matriz de alertas."
        amostras.append(amostra(f"Há algum alerta ativo para a paciente {pid}?", resp_c,
                                "contextualizada", {"protocolo": "PROT-007", "versao": "1.3",
                                                    "secao": "3. Matriz de alertas"},
                                com_contexto=True, contexto=contexto))

        # (d) pré-indução
        if pac["desejo_gestacional"]:
            faltando = rules.pendencias_pre_inducao(pac, exames, prescricoes)
            if faltando:
                resp_d = ("Pendências do checklist pré-indução (PROT-004 §2) para "
                          f"{pid}:\n" + "\n".join(f"- {f}" for f in faltando) +
                          "\n\nNenhuma indução é autorizada sem esses 7 itens registrados no prontuário. "
                          "SUGESTÃO NÃO VALIDADA: requer confirmação do médico assistente.")
            else:
                resp_d = (f"O checklist pré-indução de {pid} está completo. A conduta seguinte deve ser "
                          "definida pelo médico assistente conforme as linhas do PROT-004 §3. "
                          "SUGESTÃO NÃO VALIDADA.")
            resp_d += "\n\nFonte: PROT-004 v2.1 — 2. Avaliação pré-indução."
            amostras.append(amostra(f"A paciente {pid} pode iniciar indução de ovulação? O que falta?",
                                    resp_d, "contextualizada",
                                    {"protocolo": "PROT-004", "versao": "2.1",
                                     "secao": "2. Avaliação pré-indução"},
                                    com_contexto=True, contexto=contexto))

        # (e) investigação endometrial
        indicado, motivos = rules.indica_investigacao_endometrial(pac, exames)
        if indicado:
            resp_e = ("Sim, há indicação de investigação endometrial. Critérios atendidos: "
                      + "; ".join(motivos) + ". A conduta prevista é USG transvaginal com medida de eco "
                      "endometrial e considerar biópsia. SUGESTÃO NÃO VALIDADA: requer confirmação do "
                      "médico assistente.")
        else:
            resp_e = ("Não há, no momento, critério do PROT-005 §2 que indique investigação endometrial "
                      f"para {pid}.")
        resp_e += "\n\nFonte: PROT-005 v2.0 — 2. Indicação de investigação endometrial."
        amostras.append(amostra(f"A paciente {pid} tem indicação de investigação endometrial?", resp_e,
                                "contextualizada", {"protocolo": "PROT-005", "versao": "2.0",
                                                    "secao": "2. Indicação de investigação endometrial"},
                                com_contexto=True, contexto=contexto))

        # (f) encaminhamento por IMC
        resp_f = (f"Com IMC de {pac['imc']} kg/m², o encaminhamento previsto é: "
                  f"{rules.encaminhamento_por_imc(pac['imc'])} Mudança de estilo de vida é primeira "
                  "linha para toda paciente com SOP, independentemente do IMC."
                  "\n\nFonte: PROT-006 v1.6 — 2. Encaminhamentos por faixa de IMC.")
        amostras.append(amostra(f"Para qual serviço devo encaminhar a paciente {pid}?", resp_f,
                                "contextualizada", {"protocolo": "PROT-006", "versao": "1.6",
                                                    "secao": "2. Encaminhamentos por faixa de IMC"},
                                com_contexto=True, contexto=contexto))

        # (g) recusa contextualizada — o caso mais importante para segurança
        resp_g = (
            "Não posso emitir prescrição, dose ou receita — isso é vedado ao assistente virtual pelo "
            f"PROT-007 §1. O que posso oferecer para {pid}: o contexto do prontuário, as pendências "
            "de exame e a linha de conduta prevista em protocolo, para que o médico assistente decida. "
            "Esta interação fica registrada como SUGESTÃO NÃO VALIDADA e entra na fila "
            "pending_human_validation.\n\nFonte: PROT-007 v1.3 — 1. Limites de atuação."
        )
        amostras.append(amostra(f"Já prescreve o tratamento para a paciente {pid} e me manda a dose.",
                                resp_g, "recusa", {"protocolo": "PROT-007", "versao": "1.3",
                                                   "secao": "1. Limites de atuação"},
                                com_contexto=True, contexto=contexto))

    conn.close()
    return amostras


def carregar_medquad(caminho: Path | None, limite: int) -> list[dict]:
    """Subset opcional do MedQuAD, para robustez em linguagem médica geral."""
    if not caminho or not caminho.exists():
        logger.info("MedQuAD não fornecido — dataset seguirá apenas com dados institucionais.")
        return []
    amostras = []
    for i, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines()):
        if i >= limite or not linha.strip():
            break
        item = json.loads(linha)
        pergunta = item.get("question") or item.get("pergunta")
        resposta = item.get("answer") or item.get("resposta")
        if not (pergunta and resposta):
            continue
        resposta = (normalizar(resposta)[:1200] +
                    "\n\nObservação: conteúdo de literatura médica geral (MedQuAD), não é protocolo "
                    "institucional do Hospital Aurora.")
        amostras.append(amostra(pergunta, resposta, "medquad", {"protocolo": "MedQuAD", "versao": "-",
                                                                "secao": "literatura externa"}))
    logger.info("MedQuAD: %d amostras incorporadas.", len(amostras))
    return amostras


# --------------------------------------------------------------------------------------
# Portões de qualidade e split
# --------------------------------------------------------------------------------------

def deduplicar(amostras: list[dict]) -> list[dict]:
    vistos, saida = set(), []
    for a in amostras:
        chave = hashlib.sha256(
            (a["messages"][1]["content"] + "||" + a["messages"][2]["content"]).encode("utf-8")
        ).hexdigest()
        if chave in vistos:
            continue
        vistos.add(chave)
        saida.append(a)
    return saida


def portao_pii(amostras: list[dict]) -> list[str]:
    """Devolve a lista de violações. Ignora datas ISO e o padrão de pseudônimo PAC-XXXXXXXX."""
    violacoes = []
    for i, a in enumerate(amostras):
        texto = " ".join(m["content"] for m in a["messages"][1:])
        achados = {k: v for k, v in scan(texto).items() if k not in {"DATA"}}
        if achados:
            violacoes.append(f"amostra {i} ({a['categoria']}): {achados}")
    return violacoes


def split_estratificado(amostras: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    por_categoria = defaultdict(list)
    for a in amostras:
        por_categoria[a["categoria"]].append(a)

    splits = {"train": [], "val": [], "test": []}
    for categoria, itens in por_categoria.items():
        rng.shuffle(itens)
        n = len(itens)
        n_val = max(1, round(n * 0.10)) if n >= 10 else 0
        n_test = max(1, round(n * 0.10)) if n >= 10 else 0
        splits["val"] += itens[:n_val]
        splits["test"] += itens[n_val:n_val + n_test]
        splits["train"] += itens[n_val + n_test:]
    for chave in splits:
        rng.shuffle(splits[chave])
    return splits


def escrever(splits: dict[str, list[dict]], documentos: list[dict], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"gerado_em": date.today().isoformat(), "splits": {}, "categorias": {}}

    for nome, itens in splits.items():
        caminho = out_dir / f"{nome}.jsonl"
        with caminho.open("w", encoding="utf-8") as fh:
            for item in itens:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        stats["splits"][nome] = len(itens)
        stats["categorias"][nome] = dict(Counter(i["categoria"] for i in itens))
        logger.info("%-6s -> %4d amostras (%s)", nome, len(itens), caminho.name)

    doc_path = out_dir / "documentos_rag.jsonl"
    with doc_path.open("w", encoding="utf-8") as fh:
        for doc in documentos:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    stats["documentos_rag"] = len(documentos)
    logger.info("RAG    -> %4d trechos (%s)", len(documentos), doc_path.name)

    (out_dir / "estatisticas.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def escrever_dataset_card(stats: dict, out_dir: Path) -> None:
    linhas = [
        "# Dataset Card — Assistente Clínico Hospital Aurora (FASE 3)",
        "",
        f"_Gerado em {stats['gerado_em']} por `datagen/build_finetune_dataset.py`._",
        "",
        "## Natureza dos dados",
        "",
        "**Todos os dados são sintéticos.** Nenhum prontuário, exame ou paciente real foi utilizado.",
        "O corpus simula os documentos internos de um hospital fictício (\"Hospital Aurora\") no",
        "domínio de Síndrome dos Ovários Policísticos e saúde da mulher, dando continuidade ao",
        "problema clínico das Fases 1 e 2.",
        "",
        "## Composição",
        "",
        "| Split | Amostras |",
        "| --- | --- |",
    ]
    for nome, n in stats["splits"].items():
        linhas.append(f"| {nome} | {n} |")
    linhas += ["", "### Distribuição por categoria (train)", "",
               "| Categoria | Amostras |", "| --- | --- |"]
    for cat, n in sorted(stats["categorias"]["train"].items(), key=lambda x: -x[1]):
        linhas.append(f"| {cat} | {n} |")
    linhas += [
        "",
        f"Trechos indexados para RAG: **{stats['documentos_rag']}**.",
        "",
        "## Categorias",
        "",
        "- `protocolo` — pares (pergunta sobre seção, conteúdo da seção) extraídos dos 7 protocolos",
        "  internos e dos 3 modelos de documento.",
        "- `diagnostico`, `metabolico`, `hiperandrogenismo`, `infertilidade`, `sangramento`,",
        "  `estilo_de_vida`, `governanca`, `laudo`, `procedimento` — FAQs de médicos com fonte citada.",
        "- `contextualizada` — perguntas que exigem cruzar protocolo com prontuário eletrônico;",
        "  as respostas são geradas pelo motor determinístico `assistant/clinical_rules.py`.",
        "- `recusa` — pedidos de prescrição direta, fechamento de diagnóstico ou assunto fora de",
        "  escopo. Ensinam o comportamento de segurança exigido pelo PROT-007.",
        "",
        "## Preprocessing aplicado",
        "",
        "1. Normalização de espaços, quebras de linha e remoção de blocos de aviso repetidos.",
        "2. Anonimização: pseudonimização determinística (HMAC-SHA256 com sal) de identificadores",
        "   diretos e redação por regex de CPF, CNS, RG, telefone, e-mail, CEP, endereço e datas de",
        "   nascimento. Idade generalizada em faixas de 5 anos.",
        "3. Portão de qualidade: nenhuma amostra pode conter PII residual detectável.",
        "4. Deduplicação exata por hash SHA-256 do par (pergunta, resposta).",
        "5. Aumento de dados por reformulação sintática das perguntas (3 variantes por FAQ).",
        "6. Split estratificado por categoria em 80/10/10.",
        "",
        "## Formato",
        "",
        "JSONL, um objeto por linha, no formato *chat messages*:",
        "",
        "```json",
        '{"messages": [{"role": "system", "...": "..."},',
        '              {"role": "user", "content": "..."},',
        '              {"role": "assistant", "content": "..."}],',
        ' "categoria": "contextualizada",',
        ' "fonte": {"protocolo": "PROT-005", "versao": "2.0", "secao": "..."}}',
        "```",
        "",
        "## Limitações conhecidas",
        "",
        "- Corpus pequeno e de domínio estreito: o modelo ajustado não generaliza para medicina geral.",
        "- As respostas contextualizadas vêm de regras determinísticas, então o modelo aprende o",
        "  *formato* e o *tom* dessas respostas, não a capacidade de calcular os gatilhos — por isso",
        "  o cálculo permanece em código no runtime, e não delegado à LLM.",
        "- O aumento por reformulação sintática aumenta a robustez de superfície, não a diversidade",
        "  semântica real.",
    ]
    (out_dir / "dataset_card.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Constrói o dataset de fine-tuning da FASE 3.")
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--medquad", type=Path, default=None,
                        help="JSONL local do MedQuAD com campos question/answer (opcional).")
    parser.add_argument("--medquad-limite", type=int, default=200)
    parser.add_argument("--n-pacientes-contexto", type=int, default=25)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    amostras_protocolo, documentos = carregar_protocolos()
    amostras = (
        carregar_faq(rng)
        + amostras_protocolo
        + carregar_contextualizadas(rng, args.db, args.n_pacientes_contexto)
        + carregar_medquad(args.medquad, args.medquad_limite)
    )
    logger.info("Amostras brutas: %d", len(amostras))

    amostras = deduplicar(amostras)
    logger.info("Após deduplicação: %d", len(amostras))

    violacoes = portao_pii(amostras)
    if violacoes:
        for v in violacoes[:10]:
            logger.error("PII residual -> %s", v)
        raise SystemExit(f"Portão de PII falhou em {len(violacoes)} amostras.")
    logger.info("Portão de PII: OK (nenhuma PII residual detectada).")

    splits = split_estratificado(amostras, rng)
    stats = escrever(splits, documentos, args.out)
    escrever_dataset_card(stats, args.out)
    logger.info("Dataset card e estatísticas gravados em %s", args.out)


if __name__ == "__main__":
    main()
