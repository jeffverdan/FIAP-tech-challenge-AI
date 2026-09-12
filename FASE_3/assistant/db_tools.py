"""
Acesso à base estruturada de prontuários (SQLite) — FASE 3.

**Decisão de segurança:** não usamos um agente com SQL livre (`SQLDatabaseToolkit`). A LLM
nunca escreve SQL. Em vez disso, expomos um conjunto fechado de *tools* LangChain, cada uma
com uma consulta parametrizada e revisada. Motivos:

  * elimina injeção de SQL e consultas acidentalmente destrutivas;
  * a conexão é aberta em modo **somente leitura** (`mode=ro`), então o assistente é
    fisicamente incapaz de alterar o prontuário — o PROT-007 §1 exige isso;
  * cada tool devolve estrutura previsível, o que torna a auditoria e os testes viáveis.

O preço é menos flexibilidade: uma pergunta que exija um recorte não previsto não é atendida.
Em um sistema clínico, é o lado certo do trade-off.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from assistant import clinical_rules as rules
from assistant.config import CONFIG

logger = logging.getLogger("assistant.db")


class RepositorioClinico:
    """Camada de acesso somente-leitura à base anonimizada."""

    def __init__(self, caminho: Path | str | None = None) -> None:
        self.caminho = Path(caminho or CONFIG.hospital_db)
        if not self.caminho.exists():
            raise FileNotFoundError(
                f"{self.caminho} não existe. Rode: python datagen/generate_patients.py"
            )

    def _conectar(self) -> sqlite3.Connection:
        conexao = sqlite3.connect(f"file:{self.caminho}?mode=ro", uri=True)
        conexao.row_factory = sqlite3.Row
        return conexao

    def _consultar(self, sql: str, parametros: tuple = ()) -> list[dict]:
        with self._conectar() as conexao:
            return [dict(r) for r in conexao.execute(sql, parametros)]

    # -- consultas ---------------------------------------------------------------------

    def listar_pacientes(self, limite: int = 50) -> list[dict]:
        return self._consultar(
            "SELECT paciente_id, faixa_etaria, imc, diagnostico_sop, fenotipo_sop, "
            "desejo_gestacional FROM pacientes ORDER BY paciente_id LIMIT ?",
            (limite,),
        )

    def paciente(self, paciente_id: str) -> dict | None:
        linhas = self._consultar("SELECT * FROM pacientes WHERE paciente_id = ?", (paciente_id,))
        return linhas[0] if linhas else None

    def exames(self, paciente_id: str) -> list[dict]:
        return self._consultar(
            "SELECT * FROM exames WHERE paciente_id = ? ORDER BY data_solicitacao DESC",
            (paciente_id,),
        )

    def prescricoes(self, paciente_id: str, apenas_ativas: bool = False) -> list[dict]:
        sql = "SELECT * FROM prescricoes WHERE paciente_id = ?"
        if apenas_ativas:
            sql += " AND ativa = 1"
        return self._consultar(sql + " ORDER BY data_inicio DESC", (paciente_id,))

    def consultas(self, paciente_id: str, limite: int = 5) -> list[dict]:
        return self._consultar(
            "SELECT data, especialidade, medico_crm, resumo FROM consultas "
            "WHERE paciente_id = ? ORDER BY data DESC LIMIT ?",
            (paciente_id, limite),
        )

    # -- visões compostas --------------------------------------------------------------

    def dossie(self, paciente_id: str) -> dict[str, Any] | None:
        """Tudo o que o grafo precisa em uma única leitura."""
        paciente = self.paciente(paciente_id)
        if not paciente:
            return None
        exames = self.exames(paciente_id)
        prescricoes = self.prescricoes(paciente_id)
        alertas = rules.avaliar_alertas(paciente, exames)
        return {
            "paciente": paciente,
            "exames": exames,
            "prescricoes": prescricoes,
            "consultas": self.consultas(paciente_id),
            "exames_pendentes": rules.exames_pendentes(exames),
            "painel_sop_faltante": rules.painel_sop_faltante(exames),
            "alertas": [a.to_dict() for a in alertas],
            "severidade_maxima": rules.severidade_maxima(alertas),
            "pendencias_pre_inducao": rules.pendencias_pre_inducao(paciente, exames, prescricoes),
        }

    def resumo_textual(self, paciente_id: str) -> str:
        """Bloco PACIENTE injetado no prompt — apenas dados anonimizados."""
        dossie = self.dossie(paciente_id)
        if not dossie:
            return f"Paciente {paciente_id} não encontrada na base."
        p = dossie["paciente"]
        return (
            f"{p['paciente_id']} | faixa etária {p['faixa_etaria']} | IMC {p['imc']} kg/m² | "
            f"PA {p['pa_sistolica']}/{p['pa_diastolica']} mmHg | Ferriman-Gallwey "
            f"{p['ferriman_gallwey']} ({rules.classificar_hirsutismo(p['ferriman_gallwey'])}) | "
            f"sangramentos/12m {p['sangramentos_12m']} | último sangramento {p['ultimo_sangramento']} | "
            f"diagnóstico SOP: {'sim' if p['diagnostico_sop'] else 'não'}"
            f"{' (fenótipo ' + p['fenotipo_sop'] + ')' if p['fenotipo_sop'] else ''} | "
            f"desejo gestacional: {'sim' if p['desejo_gestacional'] else 'não'} | "
            f"comorbidades: {p['comorbidades']} | exames pendentes: {len(dossie['exames_pendentes'])}"
        )


# --------------------------------------------------------------------------------------
# Tools LangChain
# --------------------------------------------------------------------------------------

_REPO: RepositorioClinico | None = None


def repositorio() -> RepositorioClinico:
    global _REPO
    if _REPO is None:
        _REPO = RepositorioClinico()
    return _REPO


@tool
def consultar_paciente(paciente_id: str) -> dict:
    """Recupera o cadastro clínico anonimizado de uma paciente pelo pseudônimo (ex.: PAC-1A2B3C4D)."""
    paciente = repositorio().paciente(paciente_id)
    return paciente or {"erro": f"Paciente {paciente_id} não encontrada."}


@tool
def listar_exames_pendentes(paciente_id: str) -> list[dict]:
    """Lista os exames solicitados sem resultado liberado, com dias em aberto e flag de atraso."""
    return rules.exames_pendentes(repositorio().exames(paciente_id))


@tool
def listar_exames_recentes(paciente_id: str, limite: int = 10) -> list[dict]:
    """Lista os exames mais recentes com resultado liberado (nome, valor, unidade, referência)."""
    exames = [e for e in repositorio().exames(paciente_id) if e["status"] == "liberado"]
    return [
        {k: e[k] for k in ("nome_exame", "valor", "unidade", "referencia", "data_resultado")}
        for e in exames[:limite]
    ]


@tool
def listar_prescricoes_ativas(paciente_id: str) -> list[dict]:
    """Lista as classes terapêuticas ativas da paciente (sem dose — a base não armazena posologia)."""
    return [
        {k: p[k] for k in ("classe_terapeutica", "data_inicio", "protocolo_referencia", "validado_por_crm")}
        for p in repositorio().prescricoes(paciente_id, apenas_ativas=True)
    ]


@tool
def historico_consultas(paciente_id: str, limite: int = 5) -> list[dict]:
    """Recupera as últimas consultas registradas: data, especialidade, CRM e resumo anonimizado."""
    return repositorio().consultas(paciente_id, limite)


@tool
def avaliar_alertas_paciente(paciente_id: str) -> list[dict]:
    """Aplica a matriz de alertas do PROT-007 e devolve os gatilhos ativos por severidade."""
    dossie = repositorio().dossie(paciente_id)
    return dossie["alertas"] if dossie else []


@tool
def checklist_pre_inducao(paciente_id: str) -> list[str]:
    """Devolve os itens do checklist pré-indução (PROT-004 §2) ainda não documentados."""
    dossie = repositorio().dossie(paciente_id)
    return dossie["pendencias_pre_inducao"] if dossie else []


FERRAMENTAS = [
    consultar_paciente,
    listar_exames_pendentes,
    listar_exames_recentes,
    listar_prescricoes_ativas,
    historico_consultas,
    avaliar_alertas_paciente,
    checklist_pre_inducao,
]
