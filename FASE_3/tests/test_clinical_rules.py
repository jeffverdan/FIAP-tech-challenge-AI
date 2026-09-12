"""O motor de regras é o que garante determinismo nos gatilhos de segurança."""

from datetime import date, timedelta

import pytest

from assistant import clinical_rules as rules

HOJE = date(2026, 9, 5)


def exame(nome, valor, status="liberado", dias_atras=10):
    data = (HOJE - timedelta(days=dias_atras)).isoformat()
    return {"nome_exame": nome, "valor": valor, "status": status,
            "data_solicitacao": data, "data_resultado": data if status == "liberado" else None}


def paciente(**campos):
    base = {"paciente_id": "PAC-TESTE", "imc": 26.0, "pa_sistolica": 120, "pa_diastolica": 78,
            "ferriman_gallwey": 8, "sangramentos_12m": 12, "comorbidades": "nenhuma",
            "ultimo_sangramento": (HOJE - timedelta(days=20)).isoformat(),
            "ultimo_retorno_multi": (HOJE - timedelta(days=30)).isoformat(),
            "desejo_gestacional": 0}
    base.update(campos)
    return base


# -- alertas ---------------------------------------------------------------------------

def test_hemoglobina_baixa_gera_alerta_critico():
    alertas = rules.avaliar_alertas(paciente(), [exame("Hemoglobina", 7.2)], HOJE)
    assert alertas[0].severidade == "CRITICO"
    assert rules.severidade_maxima(alertas) == "CRITICO"


def test_instabilidade_hemodinamica_gera_critico():
    alertas = rules.avaliar_alertas(paciente(pa_sistolica=82), [], HOJE)
    assert any(a.severidade == "CRITICO" and "hemodin" in a.gatilho.lower() for a in alertas)


def test_hba1c_alta_sem_diabetes_registrado_gera_alto():
    alertas = rules.avaliar_alertas(paciente(), [exame("Hemoglobina glicada", 7.1)], HOJE)
    assert any(a.severidade == "ALTO" and "HbA1c" in a.gatilho for a in alertas)


def test_hba1c_alta_com_diabetes_ja_registrado_nao_gera_alerta():
    alertas = rules.avaliar_alertas(
        paciente(comorbidades="diabetes tipo 2 em acompanhamento"),
        [exame("Hemoglobina glicada", 7.1)], HOJE)
    assert not any("HbA1c" in a.gatilho for a in alertas)


def test_testosterona_acima_de_2x_lsn_gera_alto():
    alertas = rules.avaliar_alertas(paciente(), [exame("Testosterona total", 155)], HOJE)
    assert any("Testosterona" in a.gatilho and a.severidade == "ALTO" for a in alertas)


def test_alertas_saem_ordenados_por_severidade():
    exames = [exame("Hemoglobina", 7.0), exame("Hemoglobina glicada", 7.0),
              exame("TSH", 2.0, status="pendente", dias_atras=200)]
    severidades = [a.severidade for a in rules.avaliar_alertas(paciente(), exames, HOJE)]
    ordem = {"CRITICO": 0, "ALTO": 1, "MEDIO": 2, "BAIXO": 3}
    assert severidades == sorted(severidades, key=lambda s: ordem[s])


def test_paciente_sem_achados_nao_gera_alerta_alto_ou_critico():
    alertas = rules.avaliar_alertas(paciente(), [exame("TSH", 2.0)], HOJE)
    assert all(a.severidade in {"MEDIO", "BAIXO"} for a in alertas)


# -- exames ----------------------------------------------------------------------------

def test_exame_pendente_ha_mais_de_60_dias_e_marcado_como_atrasado():
    pendentes = rules.exames_pendentes([exame("TSH", None, "pendente", 90)], HOJE)
    assert pendentes[0]["atrasado"] is True
    assert pendentes[0]["dias_em_aberto"] == 90


def test_exame_pendente_recente_nao_e_atrasado():
    assert rules.exames_pendentes([exame("TSH", None, "pendente", 10)], HOJE)[0]["atrasado"] is False


def test_painel_sop_faltante_lista_apenas_o_que_falta():
    faltante = rules.painel_sop_faltante([exame("TSH", 2.0), exame("Prolactina", 12)])
    assert "TSH" not in faltante and "Prolactina" not in faltante
    assert "Beta-hCG" in faltante


# -- regras por protocolo ---------------------------------------------------------------

@pytest.mark.parametrize("imc,trecho", [
    (22.0, "nutricional"), (27.0, "educador"), (32.0, "multiprofissional"), (38.0, "intensiva"),
])
def test_encaminhamento_por_imc(imc, trecho):
    assert trecho in rules.encaminhamento_por_imc(imc).lower()


def test_encaminhamento_sem_imc_e_explicito():
    assert "não registrado" in rules.encaminhamento_por_imc(None)


@pytest.mark.parametrize("escore,esperado", [
    (3, "ausente"), (10, "leve"), (20, "moderado"), (30, "grave"),
])
def test_classificacao_de_hirsutismo(escore, esperado):
    assert esperado in rules.classificar_hirsutismo(escore)


def test_checklist_pre_inducao_reconhece_itens_documentados():
    exames = [exame("TSH", 2.0), exame("Prolactina", 12), exame("Hemoglobina glicada", 5.2)]
    prescricoes = [{"classe_terapeutica": "suplementação de ácido fólico", "ativa": 1}]
    pendentes = rules.pendencias_pre_inducao(paciente(desejo_gestacional=1), exames, prescricoes)
    assert "TSH" not in pendentes
    assert "Prolactina" not in pendentes
    assert "Ácido fólico iniciado há >= 30 dias" not in pendentes
    assert "Espermograma do parceiro" in pendentes


def test_amenorreia_prolongada_indica_investigacao_endometrial():
    indicado, motivos = rules.indica_investigacao_endometrial(
        paciente(ultimo_sangramento=(HOJE - timedelta(days=150)).isoformat(), sangramentos_12m=2),
        [], HOJE)
    assert indicado
    assert any("amenorreia" in m for m in motivos)


def test_ciclos_regulares_nao_indicam_investigacao_endometrial():
    indicado, _ = rules.indica_investigacao_endometrial(paciente(sangramentos_12m=12), [], HOJE)
    assert not indicado
