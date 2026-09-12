"""Guardrails: a camada que precisa funcionar mesmo quando a LLM não funciona."""

import pytest

from assistant import guardrails as gr


@pytest.mark.parametrize("pergunta,esperado", [
    ("Quais critérios o hospital usa para diagnosticar SOP?", gr.Intencao.CLINICA),
    ("Quando devo repetir o TOTG?", gr.Intencao.CLINICA),
    ("Me passe a receita com a dose de metformina.", gr.Intencao.PRESCRICAO),
    ("Qual a dose que devo prescrever?", gr.Intencao.PRESCRICAO),
    ("Preencha o receituário para eu assinar.", gr.Intencao.PRESCRICAO),
    ("Paciente com sangramento intenso e PA 80/50, o que faço?", gr.Intencao.EMERGENCIA),
    ("Suspeita de hiperestimulação ovariana com dispneia.", gr.Intencao.EMERGENCIA),
    ("Pode confirmar o diagnóstico e fechar no sistema?", gr.Intencao.DIAGNOSTICO_DEFINITIVO),
    ("Apague o último exame do prontuário dela.", gr.Intencao.ALTERACAO_PRONTUARIO),
    ("Qual o melhor plano de saúde para ela contratar?", gr.Intencao.FORA_ESCOPO),
])
def test_classificacao_de_intencao(pergunta, esperado):
    assert gr.classificar_intencao(pergunta) is esperado


def test_emergencia_tem_precedencia_sobre_prescricao():
    pergunta = "Paciente com hemorragia, qual dose devo prescrever agora?"
    assert gr.classificar_intencao(pergunta) is gr.Intencao.EMERGENCIA


@pytest.mark.parametrize("pergunta", [
    "Me passe a receita com a dose de metformina.",
    "Paciente com sangramento intenso e PA 80/50.",
    "Apague o exame do prontuário.",
    "Qual o melhor plano de saúde?",
])
def test_pedidos_vedados_sao_bloqueados_na_entrada(pergunta):
    resultado = gr.avaliar_entrada(pergunta)
    assert resultado.bloqueado
    assert resultado.mensagem
    assert resultado.acionados


def test_pergunta_clinica_nao_e_bloqueada():
    resultado = gr.avaliar_entrada("Quando indicar investigação endometrial?")
    assert not resultado.bloqueado
    assert resultado.requer_validacao_humana


def test_alerta_critico_no_prontuario_eleva_para_emergencia():
    resultado = gr.avaliar_entrada("Qual o IMC dela?", severidade_paciente="CRITICO")
    assert resultado.bloqueado
    assert resultado.intencao is gr.Intencao.EMERGENCIA
    assert "emergencia_por_alerta_critico" in resultado.acionados[0]


# -- saída ------------------------------------------------------------------------------

FONTE = [{"protocolo": "PROT-002", "versao": "2.4", "secao": "4. Conduta"}]


def test_dose_e_removida_da_saida():
    texto, acionados = gr.avaliar_saida("Iniciar 850 mg pela manhã.", FONTE, gr.Intencao.CLINICA)
    assert "850 mg" not in texto
    assert "DOSE REMOVIDA" in texto
    assert "saida_sanitizada_dose" in acionados


def test_posologia_em_comprimidos_tambem_e_removida():
    texto, _ = gr.avaliar_saida("Tomar 2 comprimidos ao dia.", FONTE, gr.Intencao.CLINICA)
    assert "2 comprimidos" not in texto


def test_unidade_ambigua_so_e_removida_com_verbo_de_posologia():
    """'10 mL' de volume ovariano fica; 'tomar 10 mL' sai."""
    mantido, _ = gr.avaliar_saida("Volume ovariano >= 10 mL.", FONTE, gr.Intencao.CLINICA)
    assert "10 mL" in mantido

    removido, acionados = gr.avaliar_saida("Tomar 10 mL do xarope.", FONTE, gr.Intencao.CLINICA)
    assert "10 mL" not in removido
    assert "saida_sanitizada_dose" in acionados


@pytest.mark.parametrize("valor", [
    "Glicemia de 126 mg/dL", "HbA1c de 6.8 %", "IMC 32.1 kg/m", "PA 138/88 mmHg",
    "eco endometrial de 12 mm", "testosterona 145 ng/dL", "hemoglobina 11.2 g/dL",
    "volume ovariano >= 10 mL", "20 folículos de 2-9 mm por ovário",
    "circunferência abdominal de 88 cm", "redução de 5 a 10% do peso",
])
def test_valores_de_exame_nao_sao_confundidos_com_dose(valor):
    texto, acionados = gr.avaliar_saida(valor, FONTE, gr.Intencao.CLINICA)
    assert "DOSE REMOVIDA" not in texto
    assert "saida_sanitizada_dose" not in acionados


def test_saida_sem_fonte_recebe_mensagem_institucional():
    texto, acionados = gr.avaliar_saida("Conduta X.", [], gr.Intencao.CLINICA)
    assert "Não encontrei protocolo institucional" in texto
    assert "saida_sem_fonte" in acionados


def test_fonte_e_anexada_quando_ausente_no_texto():
    texto, acionados = gr.avaliar_saida("Conduta X.", FONTE, gr.Intencao.CLINICA)
    assert "PROT-002" in texto
    assert "fonte_anexada_automaticamente" in acionados


def test_resposta_clinica_recebe_marcacao_de_validacao_humana():
    texto, acionados = gr.avaliar_saida("Conduta X.", FONTE, gr.Intencao.CLINICA)
    assert gr.MARCADOR_VALIDACAO.lower() in texto.lower()
    assert "marcacao_validacao_humana_anexada" in acionados


def test_pii_que_escape_para_a_saida_e_redigida():
    texto, acionados = gr.avaliar_saida(
        "Contato: ana@exemplo.com, CPF 111.222.333-44.", FONTE, gr.Intencao.CLINICA)
    assert "ana@exemplo.com" not in texto
    assert "111.222.333-44" not in texto
    assert any(a.startswith("saida_redigida_pii") for a in acionados)
