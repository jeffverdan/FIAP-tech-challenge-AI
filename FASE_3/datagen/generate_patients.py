"""
Geração da base estruturada sintética do "Hospital Aurora" — FASE 3.

Fluxo:
    1. Gera prontuários *brutos* com PII realista (nome, CPF, telefone, endereço) em
       ``data/raw/prontuarios_brutos.jsonl`` — serve para demonstrar a etapa de anonimização.
    2. Passa cada registro pelo :mod:`datagen.anonymizer`.
    3. Persiste o resultado anonimizado em SQLite (``data/hospital.db``), que é a base
       consultada pelas *tools* do LangChain.

Todos os dados são **sintéticos**. Nenhum paciente real foi utilizado.

Uso:
    python datagen/generate_patients.py --n-pacientes 40 --seed 62
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datagen.anonymizer import Anonymizer, faixa_etaria  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("datagen.pacientes")

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "prontuarios_brutos.jsonl"
DB_PATH = BASE_DIR / "data" / "hospital.db"

PRENOMES = ["Maria", "Ana", "Juliana", "Fernanda", "Beatriz", "Camila", "Larissa", "Patrícia",
            "Renata", "Aline", "Tatiane", "Vanessa", "Bruna", "Carolina", "Débora", "Elaine",
            "Gabriela", "Helena", "Isabela", "Jéssica"]
SOBRENOMES = ["Souza", "Oliveira", "Santos", "Pereira", "Almeida", "Costa", "Rodrigues",
              "Nascimento", "Lima", "Carvalho", "Ferreira", "Ribeiro", "Barbosa", "Monteiro"]
LOGRADOUROS = ["Rua das Acácias", "Avenida Brasil", "Rua Pedro Alves", "Travessa São João",
               "Alameda dos Ipês", "Rua Marechal Deodoro"]

ESPECIALIDADES = ["Ginecologia", "Endocrinologia", "Reprodução Humana",
                  "Ambulatório Multiprofissional", "Nutrição"]

# nome_exame, unidade, prazo_dias, faixa_normal (min, max), gerador
EXAMES = [
    ("Beta-hCG", "mUI/mL", 1, (0, 5)),
    ("TSH", "mUI/L", 1, (0.4, 4.0)),
    ("Prolactina", "ng/mL", 2, (4, 23)),
    ("17-OH-progesterona", "ng/mL", 3, (0.2, 2.0)),
    ("Testosterona total", "ng/dL", 2, (15, 70)),
    ("SHBG", "nmol/L", 2, (30, 90)),
    ("FSH", "mUI/mL", 2, (3, 9)),
    ("LH", "mUI/mL", 2, (2, 12)),
    ("Glicemia de jejum", "mg/dL", 1, (70, 99)),
    ("Insulina de jejum", "uUI/mL", 1, (2, 15)),
    ("Hemoglobina glicada", "%", 1, (4.5, 5.6)),
    ("Colesterol total", "mg/dL", 1, (120, 190)),
    ("HDL", "mg/dL", 1, (50, 80)),
    ("Triglicérides", "mg/dL", 1, (50, 149)),
    ("TOTG 75g - 120 min", "mg/dL", 2, (70, 139)),
    ("Hemoglobina", "g/dL", 1, (12, 16)),
    ("Eco endometrial (USG TV)", "mm", 2, (4, 7)),
]

CLASSES_TERAPEUTICAS = [
    "contraceptivo hormonal combinado",
    "progestagênio cíclico",
    "sensibilizador de insulina",
    "inibidor de aromatase (indução)",
    "suplementação de ácido fólico",
]

SCHEMA = """
DROP TABLE IF EXISTS pacientes;
DROP TABLE IF EXISTS consultas;
DROP TABLE IF EXISTS exames;
DROP TABLE IF EXISTS prescricoes;
DROP TABLE IF EXISTS alertas;
DROP TABLE IF EXISTS validacoes;

CREATE TABLE pacientes (
    paciente_id            TEXT PRIMARY KEY,
    faixa_etaria           TEXT NOT NULL,
    idade_aprox            INTEGER,
    peso_kg                REAL,
    altura_m               REAL,
    imc                    REAL,
    circunferencia_abd_cm  REAL,
    pa_sistolica           INTEGER,
    pa_diastolica          INTEGER,
    ferriman_gallwey       INTEGER,
    diagnostico_sop        INTEGER,
    fenotipo_sop           TEXT,
    data_diagnostico       TEXT,
    desejo_gestacional     INTEGER,
    tabagista              INTEGER,
    sangramentos_12m       INTEGER,
    ultimo_sangramento     TEXT,
    ultimo_retorno_multi   TEXT,
    comorbidades           TEXT
);

CREATE TABLE consultas (
    consulta_id   TEXT PRIMARY KEY,
    paciente_id   TEXT NOT NULL REFERENCES pacientes(paciente_id),
    data          TEXT NOT NULL,
    especialidade TEXT NOT NULL,
    medico_crm    TEXT NOT NULL,
    resumo        TEXT
);

CREATE TABLE exames (
    exame_id          TEXT PRIMARY KEY,
    paciente_id       TEXT NOT NULL REFERENCES pacientes(paciente_id),
    nome_exame        TEXT NOT NULL,
    data_solicitacao  TEXT NOT NULL,
    data_resultado    TEXT,
    valor             REAL,
    unidade           TEXT,
    referencia        TEXT,
    status            TEXT NOT NULL CHECK (status IN ('liberado','pendente'))
);

CREATE TABLE prescricoes (
    prescricao_id       TEXT PRIMARY KEY,
    paciente_id         TEXT NOT NULL REFERENCES pacientes(paciente_id),
    classe_terapeutica  TEXT NOT NULL,
    data_inicio         TEXT NOT NULL,
    ativa               INTEGER NOT NULL,
    protocolo_referencia TEXT,
    validado_por_crm    TEXT
);

CREATE TABLE alertas (
    alerta_id    TEXT PRIMARY KEY,
    paciente_id  TEXT NOT NULL REFERENCES pacientes(paciente_id),
    severidade   TEXT NOT NULL,
    gatilho      TEXT NOT NULL,
    protocolo    TEXT,
    criado_em    TEXT NOT NULL,
    status       TEXT NOT NULL
);

CREATE TABLE validacoes (
    validacao_id   TEXT PRIMARY KEY,
    interaction_id TEXT NOT NULL,
    paciente_id    TEXT,
    crm_validador  TEXT,
    decisao        TEXT,
    justificativa  TEXT,
    criado_em      TEXT NOT NULL
);

CREATE INDEX idx_exames_paciente ON exames(paciente_id);
CREATE INDEX idx_consultas_paciente ON consultas(paciente_id);
CREATE INDEX idx_alertas_paciente ON alertas(paciente_id);
"""


def _valor_exame(rng: random.Random, faixa: tuple[float, float], alterado: bool) -> float:
    lo, hi = faixa
    if alterado:
        valor = hi * rng.uniform(1.15, 2.4)
    else:
        valor = rng.uniform(lo, hi)
    return round(valor, 2)


def gerar_paciente_bruto(rng: random.Random, idx: int, hoje: date) -> dict:
    nome = f"{rng.choice(PRENOMES)} {rng.choice(SOBRENOMES)} {rng.choice(SOBRENOMES)}"
    idade = rng.randint(16, 44)
    altura = round(rng.uniform(1.52, 1.78), 2)
    imc = round(rng.uniform(19.0, 41.0), 1)
    peso = round(imc * altura**2, 1)
    diagnostico = rng.random() < 0.75
    nasc = hoje - timedelta(days=idade * 365 + rng.randint(0, 364))

    return {
        "id": f"BRUTO-{idx:03d}",
        "nome": nome,
        # chave usada para derivar o pseudônimo: garante unicidade mesmo com homônimos
        "chave_pseudonimo": f"{nome}|BRUTO-{idx:03d}",
        "cpf": f"{rng.randint(100, 999)}.{rng.randint(100, 999)}.{rng.randint(100, 999)}-{rng.randint(10, 99)}",
        "cns": f"{rng.randint(100,999)} {rng.randint(1000,9999)} {rng.randint(1000,9999)} {rng.randint(1000,9999)}",
        "telefone": f"({rng.choice(['21','11','31'])}) 9{rng.randint(1000,9999)}-{rng.randint(1000,9999)}",
        "email": f"paciente{idx}@exemplo.com.br",
        "endereco": f"{rng.choice(LOGRADOUROS)}, {rng.randint(10, 900)}",
        "data_nascimento": nasc.strftime("%d/%m/%Y"),
        "idade": idade,
        "peso_kg": peso,
        "altura_m": altura,
        "imc": imc,
        "circunferencia_abd_cm": round(rng.uniform(70, 118), 1),
        "pa_sistolica": rng.choice([110, 116, 120, 124, 128, 132, 138, 142, 146]),
        "pa_diastolica": rng.choice([70, 74, 78, 82, 86, 88, 92]),
        "ferriman_gallwey": rng.randint(2, 28),
        "diagnostico_sop": int(diagnostico),
        "fenotipo_sop": rng.choice(["A", "B", "C", "D"]) if diagnostico else None,
        "data_diagnostico": (hoje - timedelta(days=rng.randint(60, 1400))).isoformat() if diagnostico else None,
        "desejo_gestacional": int(rng.random() < 0.35),
        "tabagista": int(rng.random() < 0.15),
        "sangramentos_12m": rng.choice([0, 2, 3, 4, 6, 8, 11, 12]),
        "ultimo_sangramento": (hoje - timedelta(days=rng.randint(5, 220))).isoformat(),
        "ultimo_retorno_multi": (hoje - timedelta(days=rng.randint(30, 420))).isoformat(),
        "comorbidades": rng.choice(
            ["nenhuma", "hipotireoidismo compensado", "resistência insulínica",
             "dislipidemia", "hipertensão em acompanhamento", "nenhuma"]
        ),
        "observacao_livre": (
            f"Paciente {nome} relata ciclos irregulares. Contato: "
            f"paciente{idx}@exemplo.com.br. Retorno agendado em "
            f"{(hoje + timedelta(days=rng.randint(20, 90))).strftime('%d/%m/%Y')}."
        ),
    }


def gerar_base(n_pacientes: int, seed: int) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    hoje = date.today()

    brutos = [gerar_paciente_bruto(rng, i + 1, hoje) for i in range(n_pacientes)]
    blocklist = [b["nome"] for b in brutos]
    anon = Anonymizer(name_blocklist=blocklist)

    pacientes, consultas, exames, prescricoes, alertas = [], [], [], [], []

    for bruto in brutos:
        registro = anon.anonymize_record(
            bruto,
            direct_identifiers=("nome", "cpf", "cns", "rg", "chave_pseudonimo"),
            pseudonym_key="chave_pseudonimo",
        )
        pid = registro["paciente_pseudonimo"]

        pacientes.append(
            {
                "paciente_id": pid,
                "faixa_etaria": registro["faixa_etaria"],
                "idade_aprox": bruto["idade"],
                "peso_kg": bruto["peso_kg"],
                "altura_m": bruto["altura_m"],
                "imc": bruto["imc"],
                "circunferencia_abd_cm": bruto["circunferencia_abd_cm"],
                "pa_sistolica": bruto["pa_sistolica"],
                "pa_diastolica": bruto["pa_diastolica"],
                "ferriman_gallwey": bruto["ferriman_gallwey"],
                "diagnostico_sop": bruto["diagnostico_sop"],
                "fenotipo_sop": bruto["fenotipo_sop"],
                "data_diagnostico": bruto["data_diagnostico"],
                "desejo_gestacional": bruto["desejo_gestacional"],
                "tabagista": bruto["tabagista"],
                "sangramentos_12m": bruto["sangramentos_12m"],
                "ultimo_sangramento": bruto["ultimo_sangramento"],
                "ultimo_retorno_multi": bruto["ultimo_retorno_multi"],
                "comorbidades": bruto["comorbidades"],
            }
        )

        # --- consultas ---
        for c in range(rng.randint(1, 4)):
            data_c = hoje - timedelta(days=rng.randint(10, 700))
            consultas.append(
                {
                    "consulta_id": f"CON-{pid[-8:]}-{c+1}",
                    "paciente_id": pid,
                    "data": data_c.isoformat(),
                    "especialidade": rng.choice(ESPECIALIDADES),
                    "medico_crm": f"CRM-RJ {rng.randint(100000, 199999)}",
                    "resumo": anon.redact(
                        rng.choice(
                            [
                                "Retorno de rotina. Mantida orientação de estilo de vida (PROT-006).",
                                "Queixa de hirsutismo em progressão lenta. Reaplicado Ferriman-Gallwey.",
                                "Avaliação de ciclo menstrual e revisão do painel laboratorial.",
                                "Consulta de acompanhamento metabólico conforme PROT-002.",
                                "Discussão de desejo gestacional e pendências pré-indução (PROT-004).",
                            ]
                        )
                    ),
                }
            )

        # --- exames ---
        painel = rng.sample(EXAMES, k=rng.randint(8, len(EXAMES)))
        for j, (nome_ex, unidade, prazo, faixa) in enumerate(painel):
            data_sol = hoje - timedelta(days=rng.randint(5, 400))
            pendente = rng.random() < 0.22
            alterado = rng.random() < 0.3
            exames.append(
                {
                    "exame_id": f"EXA-{pid[-8:]}-{j+1}",
                    "paciente_id": pid,
                    "nome_exame": nome_ex,
                    "data_solicitacao": data_sol.isoformat(),
                    "data_resultado": None if pendente else (data_sol + timedelta(days=prazo)).isoformat(),
                    "valor": None if pendente else _valor_exame(rng, faixa, alterado),
                    "unidade": unidade,
                    "referencia": f"{faixa[0]} - {faixa[1]}",
                    "status": "pendente" if pendente else "liberado",
                }
            )

        # --- prescrições ativas ---
        for k in range(rng.randint(0, 2)):
            prescricoes.append(
                {
                    "prescricao_id": f"PRE-{pid[-8:]}-{k+1}",
                    "paciente_id": pid,
                    "classe_terapeutica": rng.choice(CLASSES_TERAPEUTICAS),
                    "data_inicio": (hoje - timedelta(days=rng.randint(20, 500))).isoformat(),
                    "ativa": int(rng.random() < 0.8),
                    "protocolo_referencia": rng.choice(["PROT-002", "PROT-003", "PROT-004", "PROT-005"]),
                    "validado_por_crm": f"CRM-RJ {rng.randint(100000, 199999)}",
                }
            )

    logger.info("Relatório de anonimização: %s", json.dumps(anon.report.as_dict(), ensure_ascii=False))
    return brutos, {
        "pacientes": pacientes,
        "consultas": consultas,
        "exames": exames,
        "prescricoes": prescricoes,
        "alertas": alertas,
    }


def persistir(tabelas: dict, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        for tabela, linhas in tabelas.items():
            if not linhas:
                continue
            cols = list(linhas[0].keys())
            placeholders = ", ".join("?" for _ in cols)
            conn.executemany(
                f"INSERT INTO {tabela} ({', '.join(cols)}) VALUES ({placeholders})",
                [tuple(l[c] for c in cols) for l in linhas],
            )
            logger.info("Tabela %-12s -> %4d registros", tabela, len(linhas))
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera a base sintética do Hospital Aurora.")
    parser.add_argument("--n-pacientes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    brutos, tabelas = gerar_base(args.n_pacientes, args.seed)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("w", encoding="utf-8") as fh:
        for b in brutos:
            fh.write(json.dumps(b, ensure_ascii=False) + "\n")
    logger.info("Prontuários brutos (sintéticos, com PII) -> %s", RAW_PATH)

    persistir(tabelas, args.db)
    logger.info("Base anonimizada gravada em %s", args.db)


if __name__ == "__main__":
    main()
